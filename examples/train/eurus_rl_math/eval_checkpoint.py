"""Evaluate a LoRA adapter checkpoint on a random subset of the EURUS math dataset.

The adapter is merged into the base model at runtime; the merged weights are
written to a temporary directory and discarded after evaluation.

Usage examples
--------------
uv run -m examples.train.eurus_rl_math.eval_checkpoint \\
    --lora_adapter /path/to/lora_adapter \\
    --base_model_path Qwen/Qwen3.5-2B \\
    --num_problems 200 --seed 42 --num_gpus 1

# With a local parquet:
uv run -m examples.train.eurus_rl_math.eval_checkpoint \\
    --lora_adapter /path/to/lora_adapter \\
    --base_model_path Qwen/Qwen3.5-2B \\
    --data_path ~/data/eurus_rl_math/validation.parquet \\
    --num_problems 200
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
from loguru import logger

from examples.train.eurus_rl_math.env import MathEnv
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.train.config import InferenceEngineConfig, SkyRLLoraConfig
from skyrl.utils.tok import get_tokenizer

# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_model_path(lora_adapter: str, base_model_path: str, tmp_dir: str) -> str:
    """Merge a LoRA adapter into the base model and return the merged model path."""
    if not (Path(lora_adapter) / "adapter_config.json").exists():
        raise ValueError(
            f"No adapter_config.json found in {lora_adapter}. "
            "Pass the LoRA adapter directory directly."
        )

    logger.info(f"Merging LoRA adapter at {lora_adapter} into {base_model_path}...")

    # Import here to keep top-level imports light
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, lora_adapter)
    merged = merged.merge_and_unload()

    merged_path = os.path.join(tmp_dir, "merged_model")
    logger.info(f"Saving merged model to {merged_path}")
    merged.save_pretrained(merged_path, safe_serialization=True)

    # Copy tokenizer from base model
    tokenizer = get_tokenizer(base_model_path, trust_remote_code=True, padding_side="left")
    tokenizer.save_pretrained(merged_path)

    _patch_qwen35_config_for_vllm(merged_path)

    return merged_path


def _patch_qwen35_config_for_vllm(merged_path: str) -> None:
    """Restructure Qwen3.5 text config as VL config for vLLM compatibility.

    vLLM normalizes Qwen3_5ForCausalLM → Qwen3_5ForConditionalGeneration (the
    VL architecture) and then expects Qwen3_5Config (VL format) rather than
    Qwen3_5TextConfig.  After PEFT merge, save_pretrained writes the text-only
    config.  This function wraps it in the VL structure so vLLM's multimodal
    registry type-check passes.  With language_model_only=True the vision tower
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
# Dataset loading
# ---------------------------------------------------------------------------

def load_eurus_dataset(data_path: str | None) -> hf_datasets.Dataset:
    """Load the EURUS math dataset, either from a local parquet or HF hub."""
    if data_path:
        logger.info(f"Loading dataset from {data_path}")
        return hf_datasets.load_dataset("parquet", data_files=data_path, keep_in_memory=True)["train"]

    logger.info("Downloading EURUS dataset from HF hub (PRIME-RL/Eurus-2-RL-Data)...")
    ds = hf_datasets.load_dataset("PRIME-RL/Eurus-2-RL-Data", "default", split="train")
    ds = ds.filter(lambda x: x["ability"] == "math" and x["reward_model"]["style"] == "rule")

    # Normalise to the same schema used by the local parquet files
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

    # Dataset
    ds_full = load_eurus_dataset(args.data_path)
    ds = random_subset(ds_full, args.num_problems, args.seed)
    logger.info(f"Evaluating on {len(ds)} problems (seed={args.seed})")

    # Inference engine
    logger.info(f"Initializing vLLM engine (TP={args.num_gpus}, model={model_path})")
    # Prefix caching reuses prompt KV across all n_samples of the same problem,
    # which is critical for large n_samples (e.g. 256). The expanded list is
    # ordered [p0×n, p1×n, ...] so batches of size n_samples hit the same prefix.
    # Prefix caching is OFF by default: vLLM 0.17's Mamba prefix cache
    # ("align" mode) is experimental and crashes deterministically on hybrid
    # Mamba-Attention models like Qwen3.5.  Explicitly pass
    # --enable_prefix_caching true to opt in at your own risk.
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

    # Metrics
    n_total = len(results_by_problem)
    pass_at_1 = sum(v[0] for v in results_by_problem.values()) / n_total
    metrics: dict = {
        "checkpoint_path": args.checkpoint_path,
        "model_path": model_path,
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
    p = argparse.ArgumentParser(description="Evaluate a LoRA adapter checkpoint on a random EURUS subset")
    p.add_argument("--lora_adapter", required=True,
                   help="Path to the LoRA adapter directory (must contain adapter_config.json)")
    p.add_argument("--base_model_path", required=True,
                   help="Base HF model to merge the LoRA adapter into")
    p.add_argument("--data_path", default=None,
                   help="Local parquet file (train.parquet / validation.parquet). "
                        "Omit to download from HF hub.")
    p.add_argument("--num_problems", type=int, default=200,
                   help="Number of problems to randomly sample for evaluation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subset sampling and generation")
    p.add_argument("--num_gpus", type=int, default=1,
                   help="Tensor parallel size for the vLLM engine")
    p.add_argument("--max_prompt_length", type=int, default=5120)
    p.add_argument("--max_generate_length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (pass@1). Set >0 with --n_samples for pass@k.")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Independent samples per problem (requires --temperature > 0 for diversity)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Prompts per generate() call. Defaults to n_samples so each batch covers "
                        "exactly one problem (maximises prefix-cache hits). Override if needed.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--enable_prefix_caching", type=lambda x: x.lower() != "false",
                   default=None, metavar="BOOL",
                   help="Enable vLLM prefix caching. Defaults to True when n_samples > 1.")
    p.add_argument("--output_path", default=None,
                   help="Write JSON results to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples > 1 and args.temperature == 0.0:
        logger.warning("n_samples > 1 with temperature=0 gives identical samples; consider --temperature 0.6")

    ray.init(ignore_reinit_error=True)

    tmp_dir = tempfile.mkdtemp(prefix="eval_ckpt_")
    try:
        model_path = resolve_model_path(args.lora_adapter, args.base_model_path, tmp_dir)
        metrics = asyncio.run(run_eval(args, model_path))
        print(f"\npass@1: {metrics['pass_at_1']:.4f}  (n={metrics['n_problems']} problems)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
