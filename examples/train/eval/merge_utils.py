"""Model-merge helpers shared by the unified eval entry point.

Reconstructs a standard HF model on disk from a base model + a custom-LoRA
v-params safetensors file (the SkyRL "TinyLoRA" format where only the v
coefficients are trained, and the full delta_W is reconstructed from the
base model's SVD + the saved v's). The merged model is written to ``tmp_dir``
so vLLM can load it like any other checkpoint.

The unwrap step uses meta-device parameter transfer (no weight clone) so peak
CPU RAM stays at ~base-model-size rather than ~2× during the merge — the
difference between running and OOMing on 4B+ models.
"""

import gc
import json
import os

import torch
from loguru import logger
from safetensors.torch import load_file


def resolve_custom_lora_path(
    checkpoint_dir: str | None = None,
    adapter_path: str | None = None,
) -> str:
    """Find the custom LoRA v parameters safetensors file."""
    if adapter_path is not None:
        if os.path.isfile(adapter_path):
            return adapter_path
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    if checkpoint_dir is None:
        raise ValueError("Either --checkpoint_dir or --adapter_path must be provided")

    for candidate in [
        os.path.join(checkpoint_dir, "custom_lora", "custom_lora_params.safetensors"),
        os.path.join(os.path.dirname(checkpoint_dir), "custom_lora", "custom_lora_params.safetensors"),
    ]:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find custom_lora_params.safetensors in {checkpoint_dir}. "
        "Expected at <checkpoint_dir>/custom_lora/custom_lora_params.safetensors"
    )


def resolve_model_path(
    base_model_path: str,
    svd_rank: int,
    num_coefficients: int,
    projection_seed: int,
    scheme: str,
    target_modules: str,
    exclude_modules: str | None,
    tmp_dir: str,
    checkpoint_dir: str | None = None,
    adapter_path: str | None = None,
) -> str:
    """Load base model, apply custom LoRA, load v params, merge, save to ``tmp_dir``.

    Returns the path to the saved merged model (a directory containing the
    HF weights + tokenizer + a config.json patched for vLLM).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from skyrl.backends.skyrl_train.custom_lora.apply import apply_custom_lora
    from skyrl.backends.skyrl_train.custom_lora.module import CustomLoraLinear

    # 1. Find saved v parameters
    v_params_path = resolve_custom_lora_path(
        checkpoint_dir=checkpoint_dir, adapter_path=adapter_path,
    )
    logger.info(f"Loading custom LoRA v params from {v_params_path}")
    v_params = load_file(v_params_path)
    logger.info(f"  Loaded {len(v_params)} v tensors, "
                f"total params: {sum(p.numel() for p in v_params.values()):,}")

    # 2. Load base model in float32 on CPU to match training.
    #    During FSDP training, rank 0 loads in float32 on CPU
    #    (bf16=False in fsdp_worker.py:214, cpu_init_weights in fsdp_utils.py:62).
    #    SVD must see identical float32 weights on CPU for correct buffers.
    logger.info(f"Loading base model from {base_model_path} (float32, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # 3. Apply custom LoRA (creates SVD buffers from base weights)
    class _Config:
        pass

    config = _Config()
    config.svd_rank = svd_rank
    config.num_coefficients = num_coefficients
    config.projection_seed = projection_seed
    config.scheme = scheme
    config.target_modules = target_modules
    config.exclude_modules = exclude_modules

    logger.info(f"Applying custom LoRA (svd_rank={svd_rank}, "
                f"num_coefficients={num_coefficients}, scheme={scheme})...")
    model = apply_custom_lora(model, config)

    # 4. Load v parameters into the model
    unexpected = []
    model_sd = model.state_dict()
    n_loaded = 0
    for name, tensor in v_params.items():
        if name in model_sd:
            model_sd[name].copy_(tensor)
            n_loaded += 1
        else:
            unexpected.append(name)

    model.load_state_dict(model_sd, strict=False)

    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint (not in model): {unexpected[:5]}...")
    logger.info(f"  Loaded {n_loaded}/{len(v_params)} v tensors into model")

    # Verify v params are non-zero
    v_norms = []
    for _, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            v_norms.append(module.v.data.abs().sum().item())
    logger.info(f"  v param norms: min={min(v_norms):.6f}, max={max(v_norms):.6f}, "
                f"mean={sum(v_norms)/len(v_norms):.6f}")

    # 5. Merge deltas into base weights
    logger.info("Merging custom LoRA deltas into base weights...")
    n_merged = 0
    for _, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            buffers = {n: b for n, b in module.named_buffers(recurse=False)}
            module.scheme.merge_into(module.weight, module.v, buffers)
            n_merged += 1
    logger.info(f"  Merged {n_merged} custom LoRA layers")

    # 6. Unwrap CustomLoraLinear -> nn.Linear.
    #
    # Memory-critical step: the merged model holds ~16GB of float32 weights
    # for a 4B model, plus per-layer SVD buffers. Naively cloning weights into
    # freshly-allocated nn.Linear modules would double the weight memory
    # (~32GB) and OOM the host before save_pretrained even runs. Instead we:
    #   (a) locate parents in a single pass (avoid O(n²) dict rebuild)
    #   (b) create target Linear on the "meta" device (no storage allocated)
    #   (c) transfer the existing Parameter objects without copying
    #   (d) drop the CustomLoraLinear refs so their SVD buffers can be freed
    logger.info("Unwrapping custom LoRA layers back to nn.Linear...")
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                attr_name = parts[1]
            else:
                parent = model
                attr_name = parts[0]
            to_replace.append((parent, attr_name, module))

    n_replaced = 0
    for parent, attr_name, module in to_replace:
        linear = torch.nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        # Transfer ownership — no copy. The old CustomLoraLinear's weight
        # storage now backs the new Linear; the old module will be GC'd once
        # we clear `to_replace`, releasing its SVD buffers and v param.
        linear.weight = module.weight
        if module.bias is not None:
            linear.bias = module.bias
        setattr(parent, attr_name, linear)
        n_replaced += 1

    to_replace.clear()
    logger.info(f"  Unwrapped {n_replaced} layers")

    # 7. Save merged model
    merged_path = os.path.join(tmp_dir, "merged_model")
    logger.info(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="2GB")

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)

    _patch_qwen35_config_for_vllm(merged_path)

    # Explicit teardown: drop references + force GC so the ~16GB fp32 base
    # weights and SVD buffers are returned to the OS before vLLM starts
    # staging its own model (otherwise peak CPU RAM ~= base + vLLM staging).
    del model, model_sd, v_params, to_replace
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged_path


def _patch_qwen35_config_for_vllm(merged_path: str) -> None:
    """Restructure Qwen3.5 text config as VL config for vLLM compatibility.

    vLLM normalizes Qwen3_5ForCausalLM → Qwen3_5ForConditionalGeneration (the
    VL architecture) and then expects Qwen3_5Config (VL format) rather than
    Qwen3_5TextConfig. After PEFT merge, save_pretrained writes the text-only
    config. This function wraps it in the VL structure so vLLM's multimodal
    registry type-check passes. With language_model_only=True the vision tower
    is stubbed out, so the dummy vision_config is never materialised.
    """
    config_path = os.path.join(merged_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("model_type") != "qwen3_5_text":
        return

    top_level_keys = {
        "architectures", "model_type", "transformers_version",
        "torch_dtype", "auto_map", "_commit_hash",
        "_attn_implementation_autoset",
    }
    text_config = {k: v for k, v in cfg.items() if k not in top_level_keys}
    text_config["model_type"] = "qwen3_5_text"

    vl_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": text_config,
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 27, "hidden_size": 1152, "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304, "num_heads": 16, "in_channels": 3,
            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 3584, "num_position_embeddings": 2304,
        },
        "image_token_id": 248056, "video_token_id": 248057,
        "vision_start_token_id": 248053, "vision_end_token_id": 248054,
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
    }
    if "transformers_version" in cfg:
        vl_config["transformers_version"] = cfg["transformers_version"]

    with open(config_path, "w") as f:
        json.dump(vl_config, f, indent=2)
    logger.info("Patched config.json to VL format for vLLM compatibility")
