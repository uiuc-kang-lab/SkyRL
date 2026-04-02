"""Model surgery: replace nn.Linear modules with CustomLoraLinear wrappers."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .module import CustomLoraLinear

logger = logging.getLogger(__name__)


def _match_target_modules(
    name: str,
    target_modules: Union[str, List[str]],
    exclude_modules: Optional[str],
) -> bool:
    """Check whether a module name should be adapted."""
    if exclude_modules and re.search(exclude_modules, name):
        return False

    if isinstance(target_modules, str):
        if target_modules == "all-linear":
            return True
        return bool(re.search(target_modules, name))

    # List of exact suffix matches (e.g. ["q_proj", "v_proj"])
    return any(name.endswith(f".{t}") or name == t for t in target_modules)


def apply_custom_lora(model: nn.Module, config) -> nn.Module:
    """Replace target nn.Linear modules with CustomLoraLinear wrappers.

    Args:
        model: The HuggingFace CausalLM model.
        config: A CustomLoraConfig-like object with attributes:
            svd_rank, num_coefficients, target_modules, exclude_modules,
            projection_seed, scheme.

    Returns:
        The modified model (same object, mutated in-place).
    """
    target_modules = config.target_modules
    exclude_modules = config.exclude_modules
    svd_rank = config.svd_rank
    num_coefficients = config.num_coefficients
    seed = config.projection_seed
    scheme_name = config.scheme

    # Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    # Collect modules to replace (avoid mutating dict during iteration)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _match_target_modules(name, target_modules, exclude_modules):
            continue
        # Find parent module and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            attr_name = parts[0]
            parent = model
        replacements.append((parent, attr_name, module))

    if not replacements:
        logger.warning(
            "[CustomLoRA] No matching nn.Linear modules found for "
            f"target_modules={target_modules}, exclude_modules={exclude_modules}"
        )
        return model

    # Perform replacements
    # Use a per-layer seed derived from the global seed + layer index
    # so each layer gets different random projections
    for idx, (parent, attr_name, linear) in enumerate(replacements):
        layer_seed = seed + idx
        custom_linear = CustomLoraLinear(
            original_linear=linear,
            svd_rank=svd_rank,
            num_coefficients=num_coefficients,
            seed=layer_seed,
            scheme_name=scheme_name,
        )
        setattr(parent, attr_name, custom_linear)

    # Count and log
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[CustomLoRA] Applied to {len(replacements)} linear layers | "
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.6f}%) | "
        f"scheme={scheme_name}, svd_rank={svd_rank}, num_coefficients={num_coefficients}"
    )
    for parent, attr_name, _ in replacements[:5]:
        logger.info(f"  [CustomLoRA] {attr_name} -> CustomLoraLinear")
    if len(replacements) > 5:
        logger.info(f"  [CustomLoRA] ... and {len(replacements) - 5} more")

    return model


def merge_custom_lora(model: nn.Module) -> None:
    """Merge all CustomLoraLinear deltas into base weights."""
    for module in model.modules():
        if isinstance(module, CustomLoraLinear):
            module.merge()


def unmerge_custom_lora(model: nn.Module) -> None:
    """Unmerge all CustomLoraLinear deltas from base weights."""
    for module in model.modules():
        if isinstance(module, CustomLoraLinear):
            module.unmerge()


def collect_custom_lora_params(model: nn.Module) -> OrderedDict:
    """Collect only trainable v parameters for checkpointing."""
    params = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.detach().cpu()
    return params
