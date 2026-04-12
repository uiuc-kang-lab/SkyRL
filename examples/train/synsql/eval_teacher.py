"""Evaluate a teacher (or any) model on a text-to-SQL dataset using SkyRL's vLLM inference.

Supports multi-turn evaluation: the model generates SQL tool calls, observes execution
results, and iterates up to --max_turns before submitting a final <solution>.

By default, uses single-turn chat template mode (all turns in one assistant message with
observations encoded directly), matching `generator.use_conversation_multi_turn=false` in
the standard training config. Pass --conversation_multi_turn to use the multi-turn chat
template variant instead.

Generates responses greedy (temperature=0) by default for a clean pass@1 estimate.
Set --temperature > 0 with --n_samples > 1 to estimate pass@k via independent samples.

Usage:
    uv run -m examples.train.synsql.eval_teacher \\
        --model_path Qwen/Qwen2.5-Coder-7B-Instruct \\
        --data_path $HOME/data/synsql/train.parquet \\
        --db_path $HOME/data/sql_data \\
        --num_gpus 4 \\
        --output_path results.json
"""

import argparse
import asyncio
import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import datasets as hf_datasets
import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.train.config import InferenceEngineConfig, SkyRLLoraConfig
from skyrl.utils.tok import get_tokenizer
from skyrl_gym.envs.sql.env import SQLEnv, Text2SQLEnvConfig
from skyrl_gym.envs.sql.utils import verify_format_and_extract


def _prompt_hash(prompt) -> str:
    """16-char hex prefix of SHA-256 over the JSON-serialised prompt.

    Used as a lightweight fingerprint to verify that uid→problem alignment
    is consistent between eval_teacher.py and training.
    """
    blob = json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class _ProblemState:
    """Mutable state for one problem (or one sample of a problem) during multi-turn eval."""

    pid: int  # problem id (index into all_valid)
    input_ids: List[int]  # running token sequence for generation
    conversation: list  # chat messages (only used in conversation_multi_turn mode)
    env: SQLEnv
    done: bool = False
    reward: float = 0.0
    n_turns_used: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a teacher model on a text-to-SQL dataset")
    p.add_argument("--model_path", required=True, help="HF repo or local path of the model to evaluate")
    p.add_argument("--data_path", required=True, help="Path to parquet dataset")
    p.add_argument("--db_path", required=True, help="Root path to database files (e.g. containing SynSQL-2.5M/)")
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs; used as tensor_parallel_size for the single vLLM engine",
    )
    p.add_argument("--max_prompt_length", type=int, default=29000, help="Skip prompts longer than this")
    p.add_argument("--max_generate_length", type=int, default=3000, help="Max new tokens to generate per turn")
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
    p.add_argument("--batch_size", type=int, default=32, help="Inference batch size (prompts per call)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--output_path", default=None, help="Optional path to write per-problem JSON results")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems (for quick tests)")
    p.add_argument("--max_turns", type=int, default=6, help="Maximum conversation turns per problem")
    p.add_argument(
        "--conversation_multi_turn",
        action="store_true",
        default=False,
        help="Use multi-turn chat template (separate user/assistant messages per turn). "
        "Default is single-turn mode (all turns in one assistant message), matching "
        "generator.use_conversation_multi_turn=false in the standard training config.",
    )
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
        inference_engine_enable_sleep=False,
    )
    lora_cfg = SkyRLLoraConfig()
    client = InferenceEngineClient(engines, tokenizer, args.model_path, lora_cfg, ie_cfg)

    stop_tokens = ["</tool_call>", "</solution>"]
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_generate_length,
        "top_p": 1.0,
        "stop": stop_tokens,
        "include_stop_str_in_output": True,
        # Must be False for models that encode <think>/<​/think> as special tokens
        # (e.g. Qwen3/3.5). With True (the default), these tags are stripped from
        # the decoded text, causing verify_format_and_extract to fail.
        "skip_special_tokens": False,
    }

    # ------------------------------------------------------------------ #
    # Pre-tokenize and filter prompts                                      #
    # ------------------------------------------------------------------ #
    logger.info("Tokenizing prompts...")
    env_cfg = Text2SQLEnvConfig(db_path=args.db_path)

    all_valid: list[tuple[int, dict, list[int]]] = []  # (seq_id, row_data, initial_ids)
    problem_meta: dict[int, dict] = {}
    n_skipped = 0
    n_db_missing = 0

    for j in range(len(ds)):
        row = ds[j]
        ids = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, return_dict=False)
        if len(ids) > args.max_prompt_length:
            n_skipped += 1
            continue
        # Verify database file exists before adding
        try:
            _test_env = SQLEnv(
                env_config=env_cfg,
                extras={
                    "db_id": row["db_id"],
                    "reward_spec": row["reward_spec"],
                    "data": row["data"],
                    "max_turns": args.max_turns,
                },
            )
            del _test_env
        except FileNotFoundError:
            n_db_missing += 1
            continue

        seq_id = len(all_valid)
        all_valid.append((seq_id, row, ids))
        problem_meta[seq_id] = {"raw_idx": j, "prompt_hash": _prompt_hash(row["prompt"])}

    logger.info(
        f"Valid prompts: {len(all_valid)}, skipped (too long): {n_skipped}, "
        f"skipped (db missing): {n_db_missing}"
    )

    # ------------------------------------------------------------------ #
    # Multi-turn generation + grading                                      #
    # ------------------------------------------------------------------ #
    # For n_samples > 1, each sample gets an independent SQLEnv and state.
    results_by_problem: dict[int, list[bool]] = defaultdict(list)

    # Precompute generation_prompt_ids for conversation_multi_turn mode
    if args.conversation_multi_turn:
        # Build a base conversation to compute the generation prompt tokens
        # (e.g., for Qwen: "<|im_start|>assistant\n")
        base_conversation = [{"role": "user", "content": ""}]
        base_ids = tokenizer.apply_chat_template(
            base_conversation, add_generation_prompt=True, return_dict=False
        )
        base_no_gen = tokenizer.apply_chat_template(
            base_conversation, add_generation_prompt=False, return_dict=False
        )
        generation_prompt_ids = base_ids[len(base_no_gen) :]

    # Build initial states: expand by n_samples
    all_states: list[_ProblemState] = []
    for seq_id, row, initial_ids in all_valid:
        for _ in range(args.n_samples):
            env = SQLEnv(
                env_config=env_cfg,
                extras={
                    "db_id": row["db_id"],
                    "reward_spec": row["reward_spec"],
                    "data": row["data"],
                    "max_turns": args.max_turns,
                },
            )
            all_states.append(
                _ProblemState(
                    pid=seq_id,
                    input_ids=list(initial_ids),
                    conversation=copy.deepcopy(row["prompt"]) if args.conversation_multi_turn else [],
                    env=env,
                )
            )

    active_indices = list(range(len(all_states)))
    n_problems_done = 0
    n_correct_so_far = 0

    for turn in range(args.max_turns):
        if not active_indices:
            break

        logger.info(
            f"Turn {turn + 1}/{args.max_turns}: {len(active_indices)} active samples "
            f"(problems done: {n_problems_done}/{len(all_valid)})"
        )

        # Process active samples in batches
        for batch_start in range(0, len(active_indices), args.batch_size):
            batch_indices = active_indices[batch_start : batch_start + args.batch_size]

            batch_token_ids = [all_states[si].input_ids for si in batch_indices]

            input_batch: InferenceEngineInput = {
                "prompt_token_ids": batch_token_ids,
                "sampling_params": sampling_params,
                "session_ids": list(range(len(batch_token_ids))),
            }
            output = await client.generate(input_batch)

            for local_idx, si in enumerate(batch_indices):
                state = all_states[si]
                response = output["responses"][local_idx]
                response_ids = output["response_ids"][local_idx]

                step_out = state.env.step(response)
                state.n_turns_used += 1

                if step_out["done"]:
                    state.done = True
                    state.reward = float(step_out["reward"])

                    # Diagnostic: log details for the first few completed problems
                    if sum(len(v) for v in results_by_problem.values()) < 3:
                        is_valid, thoughts, pred_sql, _ = verify_format_and_extract(response)
                        logger.info(
                            f"[DIAG] pid={state.pid} reward={state.reward:.1f} "
                            f"format_valid={is_valid} has_think={thoughts is not None and len(thoughts) > 0} "
                            f"pred_sql={pred_sql!r:.200}"
                        )
                        logger.info(f"[DIAG] response (first 500 chars): {response[:500]!r}")
                else:
                    # Build observation token IDs and extend input_ids for next turn
                    observations = step_out["observations"]

                    if args.conversation_multi_turn:
                        # Multi-turn mode: use chat template for proper formatting
                        state.conversation.append({"role": "assistant", "content": response})
                        for obs in observations:
                            state.conversation.append(obs)
                        state.input_ids = tokenizer.apply_chat_template(
                            state.conversation, add_generation_prompt=True, return_dict=False
                        )
                    else:
                        # Single-turn mode (default): token-in-token-out
                        # All turns stay in one assistant message. Observations are
                        # encoded directly without chat template markers.
                        #
                        # Strip EOS from response IDs since we're continuing the
                        # same assistant message (matches training behavior in
                        # _update_agent_loop_state_with_singleturn_chat_template).
                        resp_ids = list(response_ids)
                        if resp_ids and resp_ids[-1] == tokenizer.eos_token_id:
                            resp_ids = resp_ids[:-1]

                        # Encode observation content directly (no chat template)
                        obs_ids = []
                        for obs in observations:
                            obs_ids.extend(tokenizer.encode(obs["content"], add_special_tokens=False))

                        state.input_ids = state.input_ids + resp_ids + obs_ids

                    # Check if conversation has grown too long
                    if len(state.input_ids) > args.max_prompt_length:
                        logger.debug(
                            f"Problem {state.pid} exceeded max_prompt_length at turn {turn + 1}, "
                            f"force-terminating."
                        )
                        state.done = True
                        state.reward = 0.0

        # Update active set and track completed problems
        new_active = []
        for si in active_indices:
            state = all_states[si]
            if state.done:
                correct = state.reward > 0.0
                results_by_problem[state.pid].append(correct)
                if len(results_by_problem[state.pid]) == args.n_samples:
                    n_problems_done += 1
                    n_correct_so_far += int(any(results_by_problem[state.pid]))
            else:
                new_active.append(si)
        active_indices = new_active

        if n_problems_done > 0:
            logger.info(
                f"Turn {turn + 1}/{args.max_turns} done. "
                f"Running pass@k: {n_correct_so_far}/{n_problems_done} "
                f"= {n_correct_so_far / n_problems_done:.4f}"
            )

    # Any samples still active after all turns are marked as failures
    for si in active_indices:
        state = all_states[si]
        if not state.done:
            state.done = True
            state.reward = 0.0
            results_by_problem[state.pid].append(False)
            if len(results_by_problem[state.pid]) == args.n_samples:
                n_problems_done += 1
                n_correct_so_far += int(any(results_by_problem[state.pid]))

    # ------------------------------------------------------------------ #
    # Reward distribution diagnostic                                       #
    # ------------------------------------------------------------------ #
    all_rewards = [s.reward for s in all_states]
    from collections import Counter
    reward_dist = Counter(r for r in all_rewards)
    logger.info(f"[DIAG] Reward distribution: {dict(sorted(reward_dist.items()))}")
    logger.info(
        f"[DIAG] Turns used distribution: "
        f"{dict(Counter(s.n_turns_used for s in all_states))}"
    )

    # ------------------------------------------------------------------ #
    # Aggregate metrics                                                    #
    # ------------------------------------------------------------------ #
    n_total = len(results_by_problem)
    if n_total == 0:
        logger.error("No problems were evaluated!")
        return 0.0

    pass_at_1 = sum(v[0] for v in results_by_problem.values()) / n_total

    metrics: dict = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "db_path": args.db_path,
        "max_prompt_length": args.max_prompt_length,
        "max_turns": args.max_turns,
        "conversation_multi_turn": args.conversation_multi_turn,
        "n_total": n_total,
        "pass_at_1": pass_at_1,
    }

    if args.n_samples > 1:
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
        # Build per-problem results with actual reward values for debugging
        rewards_by_problem: dict[int, list[float]] = defaultdict(list)
        for s in all_states:
            rewards_by_problem[s.pid].append(s.reward)
        metrics["per_problem"] = {
            str(pid): {"samples": samples, "rewards": rewards_by_problem[pid], **problem_meta[pid]}
            for pid, samples in sorted(results_by_problem.items())
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
