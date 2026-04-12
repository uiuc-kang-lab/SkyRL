"""Evaluate a custom LoRA checkpoint on a subset of the EURUS math dataset.

Unlike standard LoRA (PEFT), custom LoRA stores only the trainable v
coefficients in ``custom_lora_params.safetensors``.  This script
reconstructs the full delta_W from the base model's SVD + the saved v params,
merges into the base weights, and evaluates via vLLM (direct, no Ray).

Usage examples
--------------
python -m examples.train.eurus_rl_math.eval_custom_lora_checkpoint \\
    --adapter_path ../models/Qwen3.5-4B_tinylora_opd/custom_lora_params.safetensors \\
    --base_model_path Qwen/Qwen3.5-4B \\
    --data_path /workspace/data/eurus_fixed/train.parquet \\
    --num_problems 2048 --n_samples 256 --temperature 0.6 --num_gpus 1

# From a SkyRL checkpoint:
python -m examples.train.eurus_rl_math.eval_custom_lora_checkpoint \\
    --checkpoint_dir /path/to/global_step_60/policy \\
    --base_model_path Qwen/Qwen3.5-4B \\
    --data_path /workspace/data/eurus_fixed/train.parquet
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets as hf_datasets
import torch
from loguru import logger
from safetensors.torch import load_file

from examples.train.eurus_rl_math.env import MathEnv


# ---------------------------------------------------------------------------
# Checkpoint resolution: custom LoRA merge
# ---------------------------------------------------------------------------

def resolve_custom_lora_path(checkpoint_dir: str | None = None, adapter_path: str | None = None) -> str:
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
    """Load base model, apply custom LoRA, load v params, merge, and save."""
    from transformers import AutoModelForCausalLM
    from skyrl.backends.skyrl_train.custom_lora.apply import apply_custom_lora
    from skyrl.backends.skyrl_train.custom_lora.module import CustomLoraLinear

    # 1. Find saved v parameters
    v_params_path = resolve_custom_lora_path(checkpoint_dir=checkpoint_dir, adapter_path=adapter_path)
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
    merged_path = os.path.join(tmp_dir, "merged_model")
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


# ---------------------------------------------------------------------------
# Dataset loading
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
# Grading (parallel via process pool)
# ---------------------------------------------------------------------------

def _grade_one(args_tuple):
    """Grade a single response in a subprocess (math_verify needs SIGALRM)."""
    import signal

    response, reward_spec, timeout = args_tuple

    # Hard process-level timeout as safety net: kill grading if it exceeds
    # 2× the math_verify timeout.  math_verify's internal SIGALRM sometimes
    # fails to fire if the LaTeX parser hangs before the alarm is armed.
    hard_timeout = (timeout * 2) if timeout else 30

    def _alarm_handler(signum, frame):
        raise TimeoutError("Grading hard timeout")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(hard_timeout)
    try:
        env = MathEnv(extras={"reward_spec": reward_spec}, math_verify_timeout=timeout)
        step_out = env.step(response)
        return float(step_out["reward"]) > 0.0
    except (TimeoutError, Exception):
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace, model_path: str) -> dict:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Dataset
    ds_full = load_eurus_dataset(args.data_path)
    ds = random_subset(ds_full, args.num_problems, args.seed)
    logger.info(f"Evaluating on {len(ds)} problems (seed={args.seed})")

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

    # Initialize vLLM directly — no Ray, no SkyRL wrapper, native n>1 support
    logger.info(f"Initializing vLLM engine (TP={args.num_gpus}, model={model_path})")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_generate_length,
        trust_remote_code=True,
        seed=args.seed,
        enable_prefix_caching=bool(args.enable_prefix_caching),
        enforce_eager=False,
        language_model_only=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_generate_length,
        top_p=1.0,
        n=args.n_samples,
    )

    # Batch problems (not expanded samples — vLLM handles n internally)
    batch_size = args.batch_size if args.batch_size is not None else 64
    n_batches = (len(all_valid) + batch_size - 1) // batch_size

    results_by_problem: dict[int, list[bool]] = defaultdict(list)
    grade_pool = ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

    for batch_idx, start in enumerate(range(0, len(all_valid), batch_size)):
        batch = all_valid[start : start + batch_size]
        pids = [b[0] for b in batch]
        prompts = [{"prompt_token_ids": b[1]} for b in batch]
        rs_list = [b[2] for b in batch]

        logger.info(f"Batch {batch_idx + 1}/{n_batches}: "
                    f"{len(prompts)} problems × {args.n_samples} samples")

        outputs = llm.generate(prompts, sampling_params=sampling_params)

        # Grade in parallel via process pool
        grade_args = []
        grade_pids = []
        for output, pid, reward_spec in zip(outputs, pids, rs_list):
            for completion in output.outputs:
                grade_args.append((completion.text, reward_spec, 10))
                grade_pids.append(pid)

        grade_results = list(grade_pool.map(_grade_one, grade_args))

        for pid, correct in zip(grade_pids, grade_results):
            results_by_problem[pid].append(correct)

        n_done = len(results_by_problem)
        if n_done > 0:
            p1 = sum(v[0] for v in results_by_problem.values()) / n_done
            if args.n_samples > 1:
                pk = sum(any(v) for v in results_by_problem.values()) / n_done
                ma = sum(sum(v) / len(v) for v in results_by_problem.values()) / n_done
                print(f"  [{n_done} problems] pass@1={p1:.4f}  pass@{args.n_samples}={pk:.4f}  mean_acc={ma:.4f}")
            else:
                print(f"  [{n_done} problems] pass@1={p1:.4f}")

    grade_pool.shutdown()

    # Final metrics
    n_total = len(results_by_problem)
    pass_at_1 = sum(v[0] for v in results_by_problem.values()) / n_total
    metrics: dict = {
        "adapter_path": args.adapter_path,
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
    p.add_argument("--checkpoint_dir", default=None,
                   help="SkyRL checkpoint directory (searches for custom_lora/custom_lora_params.safetensors)")
    p.add_argument("--adapter_path", default=None,
                   help="Direct path to any custom LoRA .safetensors file (takes precedence)")
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
                   help="Independent samples per problem (uses vLLM native n parameter)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Problems per generate() call (default: 64)")
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

    if args.checkpoint_dir is None and args.adapter_path is None:
        raise ValueError("Either --checkpoint_dir or --adapter_path must be provided")

    tmp_dir = tempfile.mkdtemp(prefix="eval_custom_lora_")
    try:
        model_path = resolve_model_path(
            base_model_path=args.base_model_path,
            svd_rank=args.svd_rank,
            num_coefficients=args.num_coefficients,
            projection_seed=args.projection_seed,
            scheme=args.scheme,
            target_modules=args.target_modules,
            exclude_modules=args.exclude_modules,
            tmp_dir=tmp_dir,
            checkpoint_dir=args.checkpoint_dir,
            adapter_path=args.adapter_path,
        )
        metrics = run_eval(args, model_path)
        print(f"\npass@1: {metrics['pass_at_1']:.4f}  (n={metrics['n_problems']} problems)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
