import argparse
import json
import os
import torch
from loguru import logger
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Checkpoint resolution: custom LoRA merge
# ---------------------------------------------------------------------------

def resolve_custom_lora_path(adapter_path: str) -> str:
    """Find the custom LoRA v parameters safetensors file."""
    if os.path.isfile(adapter_path):
        return adapter_path
    raise FileNotFoundError(f"Adapter file not found: {adapter_path}")



def resolve_model_path(
    base_model_path: str,
    svd_rank: int,
    num_coefficients: int,
    projection_seed: int,
    scheme: str,
    target_modules: str,
    exclude_modules: str | None,
    merged_path: str,
    adapter_path: str | None = None,
) -> str:
    """Load base model, apply custom LoRA, load v params, merge, and save."""
    from transformers import AutoModelForCausalLM
    from skyrl.backends.skyrl_train.custom_lora.apply import apply_custom_lora
    from skyrl.backends.skyrl_train.custom_lora.module import CustomLoraLinear

    # 1. Find saved v parameters
    v_params_path = resolve_custom_lora_path(adapter_path=adapter_path)
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

    # 6. Unwrap CustomLoraLinear -> nn.Linear
    logger.info("Unwrapping custom LoRA layers back to nn.Linear...")
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr_name = parts[1]
            else:
                parent = model
                attr_name = parts[0]
            linear = torch.nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            linear.weight = torch.nn.Parameter(module.weight.data.clone(), requires_grad=False)
            if module.bias is not None:
                linear.bias = torch.nn.Parameter(module.bias.data.clone(), requires_grad=False)
            replacements.append((parent, attr_name, linear))

    for parent, attr_name, linear in replacements:
        setattr(parent, attr_name, linear)
    logger.info(f"  Unwrapped {len(replacements)} layers")

    # 7. Save merged model
    logger.info(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)

    # Copy tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)

    _patch_qwen35_config_for_vllm(merged_path)

    del model
    return merged_path


def _patch_qwen35_config_for_vllm(merged_path: str) -> None:
    """Restructure Qwen3.5 text config as VL config for vLLM compatibility."""
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
    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a custom LoRA checkpoint on EURUS math")
    p.add_argument("--adapter_path", default=None,
                   help="Direct path to any custom LoRA .safetensors file (takes precedence)")
    p.add_argument("--base_model_path", required=True,
                   help="Base HF model (same one used during training)")
    p.add_argument("--output_path", default=None, help="Write JSON results to this path")

    # Custom LoRA config
    p.add_argument("--svd_rank", type=int, default=16)
    p.add_argument("--num_coefficients", type=int, default=16)
    p.add_argument("--projection_seed", type=int, default=42)
    p.add_argument("--scheme", default="svd_random_projection")
    p.add_argument("--target_modules", default="all-linear")
    p.add_argument("--exclude_modules", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    
    model_path = resolve_model_path(
        base_model_path=args.base_model_path,
        svd_rank=args.svd_rank,
        num_coefficients=args.num_coefficients,
        projection_seed=args.projection_seed,
        scheme=args.scheme,
        target_modules=args.target_modules,
        exclude_modules=args.exclude_modules,
        merged_path=args.output_path,
        adapter_path=args.adapter_path,
    )
    
if __name__ == "__main__":
    main()
