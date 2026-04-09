"""Evaluate a custom LoRA checkpoint on a random subset of the EURUS math dataset.

Unlike standard LoRA (PEFT), custom LoRA stores only the trainable v
coefficients in ``custom_lora/custom_lora_params.safetensors``.  This script
reconstructs the full delta_W from the base model's SVD + the saved v params,
merges into the base weights, and evaluates via vLLM.

Usage examples
--------------
uv run -m examples.train.eurus_rl_math.eval_custom_lora_checkpoint \\
    --checkpoint_dir /path/to/global_step_60/policy \\
    --base_model_path Qwen/Qwen3.5-4B \\
    --num_problems 200 --seed 42 --num_gpus 2

# With explicit custom LoRA config overrides:
uv run -m examples.train.eurus_rl_math.eval_custom_lora_checkpoint \\
    --checkpoint_dir /path/to/global_step_60/policy \\
    --base_model_path Qwen/Qwen3.5-4B \\
    --svd_rank 16 --num_coefficients 16 \\
    --data_path ~/data/eurus_rl_math/validation.parquet
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import datasets as hf_datasets
import ray
import torch
from loguru import logger
from safetensors.torch import load_file

from examples.train.eurus_rl_math.env import MathEnv
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.train.config import InferenceEngineConfig, SkyRLLoraConfig
from skyrl.utils.tok import get_tokenizer


# ---------------------------------------------------------------------------
# Checkpoint resolution: custom LoRA merge
# ---------------------------------------------------------------------------

def resolve_custom_lora_path(checkpoint_dir: str) -> str:
    """Find the custom_lora_params.safetensors file in a checkpoint directory."""
    # Try direct path
    direct = os.path.join(checkpoint_dir, "custom_lora", "custom_lora_params.safetensors")
    if os.path.exists(direct):
        return direct

    # Try one level up (checkpoint_dir might be the policy/ subdir)
    parent = os.path.join(os.path.dirname(checkpoint_dir), "custom_lora", "custom_lora_params.safetensors")
    if os.path.exists(parent):
        return parent

    raise FileNotFoundError(
        f"Could not find custom_lora_params.safetensors in {checkpoint_dir}. "
        "Expected at <checkpoint_dir>/custom_lora/custom_lora_params.safetensors"
    )


def resolve_model_path(
    checkpoint_dir: str,
    base_model_path: str,
    svd_rank: int,
    num_coefficients: int,
    projection_seed: int,
    scheme: str,
    target_modules: str,
    exclude_modules: str | None,
    tmp_dir: str,
) -> str:
    """Load base model, apply custom LoRA, load v params, merge, and save."""
    from transformers import AutoModelForCausalLM

    from skyrl.backends.skyrl_train.custom_lora.apply import apply_custom_lora
    from skyrl.backends.skyrl_train.custom_lora.schemes import get_scheme

    # 1. Find saved v parameters
    v_params_path = resolve_custom_lora_path(checkpoint_dir)
    logger.info(f"Loading custom LoRA v params from {v_params_path}")
    v_params = load_file(v_params_path)
    logger.info(f"  Loaded {len(v_params)} v tensors, "
                f"total params: {sum(p.numel() for p in v_params.values()):,}")

    # 2. Load base model on CPU
    logger.info(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
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
    missing, unexpected = [], []
    model_sd = model.state_dict()
    for name, tensor in v_params.items():
        if name in model_sd:
            model_sd[name].copy_(tensor)
        else:
            unexpected.append(name)

    # Load via state_dict to handle buffer/param distinction
    model.load_state_dict(model_sd, strict=False)

    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint (not in model): {unexpected}")

    # 5. Merge deltas into base weights
    logger.info("Merging custom LoRA deltas into base weights...")
    from skyrl.backends.skyrl_train.custom_lora.module import CustomLoraLinear

    n_merged = 0
    for name, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            scheme_obj = module.scheme
            buffers = {n: b for n, b in module.named_buffers(recurse=False)}
            scheme_obj.merge_into(module.weight, module.v, buffers)
            n_merged += 1
    logger.info(f"  Merged {n_merged} custom LoRA layers")

    # 6. Extract base model (unwrap CustomLoraLinear -> nn.Linear)
    logger.info("Unwrapping custom LoRA layers back to nn.Linear...")
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, CustomLoraLinear):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = model
            # Create a plain nn.Linear with the merged weight
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
    merged_path = os.path.join(tmp_dir, "merged_model")
    logger.info(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)

    # Copy tokenizer
    tokenizer = get_tokenizer(base_model_path, trust_remote_code=True, padding_side="left")
    tokenizer.save_pretrained(merged_path)

    _patch_qwen35_config_for_vllm(merged_path)

    del model
    torch.cuda.empty_cache()

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
            "depth": 27,
            "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 3584,
            "num_position_embeddings": 2304,
        },
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
    }
    if "transformers_version" in cfg:
        vl_config["transformers_version"] = cfg["transformers_version"]

    with open(config_path, "w") as f:
        json.dump(vl_config, f, indent=2)

    logger.info("Patched config.json to VL format for vLLM compatibility")


# ---------------------------------------------------------------------------
# Dataset loading (shared with eval_checkpoint.py)
# ---------------------------------------------------------------------------

def load_eurus_dataset(data_path: str | None) -> hf_datasets.Dataset:
    if data_path:
        logger.info(f"Loading dataset from {data_path}")
        return hf_datasets.load_dataset("parquet", data_files=data_path, keep_in_memory=True)["train"]

    logger.info("Downloading EURUS dataset from HF hub (PRIME-RL/Eurus-2-RL-Data)...")
    ds = hf_datasets.load_dataset("PRIME-RL/Eurus-2-RL-Data", "default", split="train")
    ds = ds.filter(lambda x: x["ability"] == "math" and x["reward_model"]["style"] == "rule")

    def _to_parquet_schema(ex):
        prompt = [{"role": "user", "content": ex["prompt"][-1]["content"]}]
        return {
            "prompt": prompt,
            "reward_spec": {"method": "rule", "ground_truth": ex["reward_model"]["ground_truth"]},
            "extra_info": {"question": ex["prompt"][-1]["content"]},
        }

    ds = ds.map(_to_parquet_schema, remove_columns=ds.column_names)
    return ds


def random_subset(ds: hf_datasets.Dataset, num_problems: int, seed: int) -> hf_datasets.Dataset:
    if num_problems >= len(ds):
        logger.info(f"Requested {num_problems} problems but dataset only has {len(ds)}; using all.")
        return ds
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), num_problems))
    return ds.select(indices)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def run_eval(args: argparse.Namespace, model_path: str) -> dict:
    tokenizer = get_tokenizer(model_path, trust_remote_code=True, padding_side="left")

    ds_full = load_eurus_dataset(args.data_path)
    ds = random_subset(ds_full, args.num_problems, args.seed)
    logger.info(f"Evaluating on {len(ds)} problems (seed={args.seed})")

    logger.info(f"Initializing vLLM engine (TP={args.num_gpus}, model={model_path})")
    enable_prefix_caching = args.enable_prefix_caching if args.enable_prefix_caching is not None else False
    if args.n_samples > 1 and not enable_prefix_caching:
        logger.warning("n_samples > 1 without prefix caching: prompt KV computed separately per sample.")

    ie_cfg = InferenceEngineConfig(
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_engine=True,
        enforce_eager=False,
        backend="vllm",
        enable_prefix_caching=enable_prefix_caching,
    )
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=args.num_gpus,
        model_dtype=ie_cfg.model_dtype,
        pretrain=model_path,
        seed=args.seed,
        vllm_v1_disable_multiproc=ie_cfg.vllm_v1_disable_multiproc,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_engine=True,
        tokenizer=tokenizer,
        backend="vllm",
        inference_engine_enable_sleep=False,
        engine_init_kwargs={
            "language_model_only": True,
            "max_model_len": args.max_prompt_length + args.max_generate_length,
        },
    )
    client = InferenceEngineClient(engines, tokenizer, model_path, SkyRLLoraConfig(), ie_cfg)

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_generate_length,
        "top_p": 1.0,
    }

    batch_size = args.batch_size if args.batch_size is not None else args.n_samples

    # Tokenize + filter
    logger.info("Tokenizing prompts...")
    all_valid: list[tuple[int, list[int], dict, dict]] = []
    n_skipped = 0
    for j in range(len(ds)):
        row = ds[j]
        ids = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, return_dict=False)
        if len(ids) <= args.max_prompt_length:
            all_valid.append((j, ids, row["reward_spec"], row.get("extra_info", {})))
        else:
            n_skipped += 1
    logger.info(f"Valid: {len(all_valid)}, skipped (too long): {n_skipped}")

    # Expand by n_samples
    expanded = [(pid, ids, rs, info) for (pid, ids, rs, info) in all_valid for _ in range(args.n_samples)]

    results_by_problem: dict[int, list[bool]] = defaultdict(list)
    n_batches = (len(expanded) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(expanded), batch_size)):
        batch = expanded[start : start + batch_size]
        pids, ids_list, rs_list, _ = zip(*batch)

        logger.info(f"Batch {batch_idx + 1}/{n_batches}: {len(ids_list)} completions")

        input_batch: InferenceEngineInput = {
            "prompt_token_ids": list(ids_list),
            "sampling_params": sampling_params,
            "session_ids": list(range(len(ids_list))),
        }
        output = await client.generate(input_batch)

        for response, pid, reward_spec in zip(output["responses"], pids, rs_list):
            env = MathEnv(extras={"reward_spec": reward_spec}, math_verify_timeout=10)
            step_out = env.step(response)
            results_by_problem[pid].append(float(step_out["reward"]) > 0.0)

        print(f"  pass@1 so far: {sum(v[0] for v in results_by_problem.values()) / len(results_by_problem):.4f}")

    # Metrics
    n_total = len(results_by_problem)
    pass_at_1 = sum(v[0] for v in results_by_problem.values()) / n_total
    metrics: dict = {
        "checkpoint_dir": args.checkpoint_dir,
        "base_model_path": args.base_model_path,
        "svd_rank": args.svd_rank,
        "num_coefficients": args.num_coefficients,
        "n_problems": n_total,
        "n_skipped": n_skipped,
        "seed": args.seed,
        "temperature": args.temperature,
        "pass_at_1": pass_at_1,
    }

    if args.n_samples > 1:
        pass_at_k = sum(any(v) for v in results_by_problem.values()) / n_total
        mean_acc = sum(sum(v) / len(v) for v in results_by_problem.values()) / n_total
        metrics.update({"n_samples": args.n_samples, "pass_at_k": pass_at_k, "mean_sample_acc": mean_acc})
        logger.info(
            f"pass@1={pass_at_1:.4f}  pass@{args.n_samples}={pass_at_k:.4f}  "
            f"mean_acc={mean_acc:.4f}  (n={n_total})"
        )
    else:
        logger.info(f"pass@1={pass_at_1:.4f}  (n={n_total})")

    if args.output_path:
        metrics["per_problem"] = {
            str(pid): {
                "samples": samples,
                "question": all_valid[pid][3].get("question", ""),
            }
            for pid, samples in sorted(results_by_problem.items())
        }
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results written to {args.output_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a custom LoRA checkpoint on EURUS math")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to the checkpoint directory containing custom_lora/custom_lora_params.safetensors")
    p.add_argument("--base_model_path", required=True,
                   help="Base HF model (same one used during training)")
    p.add_argument("--data_path", default=None,
                   help="Local parquet file. Omit to download from HF hub.")
    p.add_argument("--num_problems", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_gpus", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--max_prompt_length", type=int, default=5120)
    p.add_argument("--max_generate_length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy. Set >0 with --n_samples for pass@k.")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Independent samples per problem")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--enable_prefix_caching", type=lambda x: x.lower() != "false",
                   default=None, metavar="BOOL")
    p.add_argument("--output_path", default=None, help="Write JSON results to this path")

    # Custom LoRA config
    p.add_argument("--svd_rank", type=int, default=16)
    p.add_argument("--num_coefficients", type=int, default=16)
    p.add_argument("--projection_seed", type=int, default=42)
    p.add_argument("--scheme", default="svd_random_projection")
    p.add_argument("--target_modules", default="all-linear")
    p.add_argument("--exclude_modules", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples > 1 and args.temperature == 0.0:
        logger.warning("n_samples > 1 with temperature=0 gives identical samples; consider --temperature 0.6")

    ray.init(ignore_reinit_error=True)

    tmp_dir = tempfile.mkdtemp(prefix="eval_custom_lora_")
    try:
        model_path = resolve_model_path(
            checkpoint_dir=args.checkpoint_dir,
            base_model_path=args.base_model_path,
            svd_rank=args.svd_rank,
            num_coefficients=args.num_coefficients,
            projection_seed=args.projection_seed,
            scheme=args.scheme,
            target_modules=args.target_modules,
            exclude_modules=args.exclude_modules,
            tmp_dir=tmp_dir,
        )
        metrics = asyncio.run(run_eval(args, model_path))
        print(f"\npass@1: {metrics['pass_at_1']:.4f}  (n={metrics['n_problems']} problems)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
