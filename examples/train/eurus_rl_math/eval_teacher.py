"""Evaluate a teacher (or any) model on the eurus_rl_math dataset using SkyRL's vLLM inference.

Generates responses greedy (temperature=0) by default for a clean pass@1 estimate.
Set --temperature > 0 with --n_samples > 1 to estimate pass@k via independent samples.

Usage:
    uv run -m examples.train.eurus_rl_math.eval_teacher \\
        --model_path Qwen/Qwen2.5-7B-Instruct \\
        --data_path $HOME/data/eurus_rl_math/validation.parquet \\
        --num_gpus 4 \\
        --output_path results.json
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict

import datasets as hf_datasets
import ray
from loguru import logger
from tqdm import tqdm

from examples.train.eurus_rl_math.env import MathEnv
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.train.config import InferenceEngineConfig, SkyRLLoraConfig
from skyrl.utils.tok import get_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a teacher model on eurus_rl_math")
    p.add_argument("--model_path", required=True, help="HF repo or local path of the model to evaluate")
    p.add_argument("--data_path", required=True, help="Path to validation.parquet (or train_small.parquet)")
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs; used as tensor_parallel_size for the single vLLM engine",
    )
    p.add_argument("--max_prompt_length", type=int, default=5120, help="Skip prompts longer than this")
    p.add_argument("--max_generate_length", type=int, default=4096, help="Max new tokens to generate")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 = greedy decoding (best for pass@1)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Independent samples per problem. >1 requires --temperature > 0. Reports pass@1 and pass@k.",
    )
    p.add_argument("--batch_size", type=int, default=64, help="Inference batch size (prompts per call)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--output_path", default=None, help="Optional path to write per-problem JSON results")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems (for quick tests)")
    return p.parse_args()


async def run_eval(args: argparse.Namespace) -> float:
    if args.n_samples > 1 and args.temperature == 0.0:
        logger.warning("n_samples > 1 with temperature=0 gives identical greedy samples. Consider --temperature 0.6.")

    # ------------------------------------------------------------------ #
    # Tokenizer                                                            #
    # ------------------------------------------------------------------ #
    logger.info(f"Loading tokenizer: {args.model_path}")
    tokenizer = get_tokenizer(args.model_path, trust_remote_code=True, padding_side="left")

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    logger.info(f"Loading dataset: {args.data_path}")
    ds = hf_datasets.load_dataset("parquet", data_files=args.data_path, keep_in_memory=True)["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info(f"Problems to evaluate: {len(ds)}")

    # ------------------------------------------------------------------ #
    # Inference engine                                                     #
    # ------------------------------------------------------------------ #
    logger.info(f"Initializing vLLM engine (TP={args.num_gpus})")
    ie_cfg = InferenceEngineConfig(
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_engine=True,
        # For eval we want CUDA graphs for speed; they're safe outside training.
        enforce_eager=False,
        backend="vllm",
    )
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=args.num_gpus,
        model_dtype=ie_cfg.model_dtype,
        pretrain=args.model_path,
        seed=42,
        vllm_v1_disable_multiproc=ie_cfg.vllm_v1_disable_multiproc,
        enable_prefix_caching=ie_cfg.enable_prefix_caching,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_engine=True,
        tokenizer=tokenizer,
        backend="vllm",
        # No sleep/weight-sync needed for eval-only
        inference_engine_enable_sleep=False,
    )
    lora_cfg = SkyRLLoraConfig()  # rank=0, no LoRA
    client = InferenceEngineClient(engines, tokenizer, args.model_path, lora_cfg, ie_cfg)

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_generate_length,
        "top_p": 1.0,
        # Stop on </s> or common EOS tokens; model's own EOS is handled by vLLM.
    }

    # ------------------------------------------------------------------ #
    # Generation + grading                                                 #
    # ------------------------------------------------------------------ #
    # For n_samples > 1 we expand each prompt into n_samples copies and
    # deduplicate results by problem index afterward.
    results_by_problem: dict[int, list[bool]] = defaultdict(list)
    n_skipped = 0

    all_indices = list(range(len(ds)))
    for batch_start in tqdm(range(0, len(all_indices), args.batch_size), desc="Evaluating"):
        batch_idx = all_indices[batch_start : batch_start + args.batch_size]
        rows = ds[batch_idx]

        prompts = rows["prompt"]
        reward_specs = rows["reward_spec"]

        # Tokenize
        token_ids_all = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, return_dict=False)
            for p in prompts
        ]

        # Filter prompts that exceed max_prompt_length
        valid_entries = [
            (batch_idx[j], ids, rspec)
            for j, (ids, rspec) in enumerate(zip(token_ids_all, reward_specs))
            if len(ids) <= args.max_prompt_length
        ]
        n_skipped += len(batch_idx) - len(valid_entries)
        if not valid_entries:
            continue

        problem_ids, token_ids, rs_list = zip(*valid_entries)

        # Expand by n_samples: repeat each prompt n_samples times
        expanded_ids = [ids for ids in token_ids for _ in range(args.n_samples)]
        expanded_problem_ids = [pid for pid in problem_ids for _ in range(args.n_samples)]
        expanded_rs = [rs for rs in rs_list for _ in range(args.n_samples)]

        input_batch: InferenceEngineInput = {
            "prompt_token_ids": expanded_ids,
            "sampling_params": sampling_params,
            "session_ids": list(range(len(expanded_ids))),
        }
        output = await client.generate(input_batch)

        for response, pid, reward_spec in zip(output["responses"], expanded_problem_ids, expanded_rs):
            env = MathEnv(extras={"reward_spec": reward_spec})
            step_out = env.step(response)
            correct = float(step_out["reward"]) > 0.0
            results_by_problem[pid].append(correct)

    if n_skipped > 0:
        logger.warning(f"Skipped {n_skipped} prompts exceeding {args.max_prompt_length} tokens")

    # ------------------------------------------------------------------ #
    # Aggregate metrics                                                    #
    # ------------------------------------------------------------------ #
    n_total = len(results_by_problem)
    if n_total == 0:
        logger.error("No problems were evaluated!")
        return 0.0

    # pass@1: mean of per-problem first-sample correctness
    pass_at_1 = sum(v[0] for v in results_by_problem.values()) / n_total

    metrics: dict = {"model_path": args.model_path, "data_path": args.data_path, "n_total": n_total,
                     "pass_at_1": pass_at_1}

    if args.n_samples > 1:
        # pass@k: fraction of problems with at least one correct sample
        pass_at_k = sum(any(v) for v in results_by_problem.values()) / n_total
        mean_acc = sum(sum(v) / len(v) for v in results_by_problem.values()) / n_total
        metrics.update({"n_samples": args.n_samples, "pass_at_k": pass_at_k, "mean_sample_acc": mean_acc})
        logger.info(
            f"pass@1={pass_at_1:.4f}  pass@{args.n_samples}={pass_at_k:.4f}  "
            f"mean_sample_acc={mean_acc:.4f}  ({n_total} problems)"
        )
    else:
        logger.info(f"pass@1={pass_at_1:.4f}  ({n_total} problems)")

    if args.output_path:
        metrics["per_problem"] = {
            str(pid): {"samples": samples} for pid, samples in sorted(results_by_problem.items())
        }
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")

    return pass_at_1


def main() -> None:
    args = parse_args()
    ray.init(ignore_reinit_error=True)
    pass_at_1 = asyncio.run(run_eval(args))
    print(f"\nFinal pass@1: {pass_at_1:.4f}")


if __name__ == "__main__":
    main()
