"""Model surgery: replace nn.Linear modules with CustomLoraLinear wrappers."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
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


@dataclass
class _CustomLoraLayerInfo:
    """Static metadata for one CustomLoraLinear layer.

    Stores only the state_dict key names and the scheme — no references
    to the live module, so it is safe to use outside FSDP forward /
    summon_full_params contexts.
    """

    weight_key: str
    v_key: str
    buffer_keys: dict[str, str]  # scheme buffer name → state_dict key
    scheme_name: str


def build_custom_lora_delta_map(model: nn.Module) -> dict[str, _CustomLoraLayerInfo]:
    """Build a mapping from state_dict weight key → layer info.

    The returned map tells the weight-sync path which extracted tensors
    need a delta applied and where to find the corresponding v / buffers
    in the same extracted state_dict — without ever reading sharded
    module attributes.
    """
    delta_map: dict[str, _CustomLoraLayerInfo] = {}
    buffer_names = ("U_scaled", "V", "P")  # names registered by the scheme
    for name, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            prefix = f"{name}." if name else ""
            info = _CustomLoraLayerInfo(
                weight_key=f"{prefix}weight",
                v_key=f"{prefix}v",
                buffer_keys={
                    buf: f"{prefix}{buf}" for buf in buffer_names
                },
                scheme_name=module.scheme_name,
            )
            delta_map[info.weight_key] = info
    return delta_map


def collect_custom_lora_params(model: nn.Module) -> OrderedDict:
    """Collect trainable v parameters for checkpointing (non-FSDP path).

    For FSDP-sharded models, use ``collect_custom_lora_params_fsdp``
    instead, which gathers full tensors across ranks before saving.
    """
    params = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.detach().cpu()
    return params
