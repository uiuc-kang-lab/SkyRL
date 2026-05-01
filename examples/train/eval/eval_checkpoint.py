"""Unified eval entry point for TinyLoRA adapters across all four SkyRL tasks.

Consolidates the four domain-specific eval scripts (eurus_rl_math, mix_code,
mix_general, synsql) into one ``--task``-dispatched driver. Supports arbitrary
HF base models with or without a TinyLoRA adapter: if neither ``--lora_adapter``
nor ``--checkpoint_dir`` is given, the base model is evaluated as-is (useful
for establishing baselines).

Usage
-----
    # With TinyLoRA adapter:
    uv run -m examples.train.eval.eval_checkpoint \\
        --task {math,code,general,sql} \\
        --base_model_path Qwen/Qwen3.5-4B \\
        --lora_adapter /path/to/custom_lora_params.safetensors \\
        --num_problems 1024 --n_samples 4 --temperature 0.6 \\
        --output_path results.json

    # Base model only (no adapter):
    uv run -m examples.train.eval.eval_checkpoint \\
        --task math --base_model_path Qwen/Qwen3.5-4B \\
        --num_problems 1024 --n_samples 4 --temperature 0.6 \\
        --output_path baseline_math.json

    # SQL adds multi-turn flags:
    uv run -m examples.train.eval.eval_checkpoint \\
        --task sql --base_model_path Qwen/Qwen3.5-4B \\
        --lora_adapter <adapter.safetensors> \\
        --db_path ./databases --max_turns 6

Metrics reported in the output JSON:
    - pass_at_1           : mean(reward[0] > 0) across problems
    - pass_at_k           : mean(any(r > 0)) across problems  (n_samples > 1)
    - mean_sample_acc     : mean over problems of (frac of samples with r > 0)
    - avg_reward          : mean over problems of (mean raw reward)
    - reward_distribution : Counter of all raw reward values seen

For SQL, rewards are in {-1.0, 0.0, 1.0} (malformed, wrong, correct), so
``avg_reward`` is distinct from ``mean_sample_acc`` (which thresholds at >0).
For math/code/general, rewards are in {0, 1} and the two are equal.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Subprocess helpers need the project root on sys.path when re-entered via
# ``python -u .../eval_checkpoint.py --_grade_helper`` so that
# ``examples.train.*`` imports resolve.
_PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import datasets as hf_datasets
import torch
from loguru import logger
from tqdm import tqdm

from examples.train.eval import merge_utils
from examples.train.eval.tasks import (
    TASK_CHOICES, TaskAdapter, get_adapter, random_subset,
)


# ---------------------------------------------------------------------------
# Subprocess grading helper mode
# ---------------------------------------------------------------------------

def _run_grade_helper_mode() -> None:
    """Read {task, response, reward_spec, extras, timeout} from stdin,
    grade once, emit ``{"reward": float}`` to stdout.

    Used by math + general tasks (sympy/math_verify hang risk) via
    ``subprocess.run(..., timeout=hard_timeout)`` so there's a real OS-level
    SIGKILL deadline. Code and SQL grade inline.
    """
    try:
        payload = json.loads(sys.stdin.read())
        adapter = get_adapter(payload["task"])
        reward = adapter.grade_one(
            payload["response"], payload["reward_spec"], payload.get("extras", {}),
        )
    except BaseException:
        reward = 0.0
    sys.stdout.write(json.dumps({"reward": float(reward)}) + "\n")
    sys.stdout.flush()


def _grade_via_subprocess(
    task: str, response: str, reward_spec: dict, extras: dict, timeout: int,
) -> float:
    """Grade one response in a fresh subprocess; OS-level SIGKILL on timeout.

    In-process SIGALRM is unreliable for math_verify (it arms its own
    SIGALRM and clears it on exit), and sympy/LaTeX paths run in C code that
    doesn't deliver Python signals. Subprocess with a real deadline sidesteps
    both.
    """
    hard_timeout = max(timeout * 3 if (timeout and timeout > 0) else 30, 30)
    payload = json.dumps({
        "task": task,
        "response": response,
        "reward_spec": reward_spec,
        "extras": extras,
        "timeout": timeout,
    })
    try:
        res = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--_grade_helper"],
            input=payload,
            capture_output=True,
            timeout=hard_timeout,
            text=True,
            cwd=_PROJ_ROOT,
        )
        if res.returncode != 0 or not res.stdout:
            return 0.0
        for line in reversed(res.stdout.strip().splitlines()):
            try:
                obj = json.loads(line)
                return float(obj.get("reward", 0.0))
            except json.JSONDecodeError:
                continue
        return 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except BaseException:
        return 0.0


# ---------------------------------------------------------------------------
# Tokenize + filter
# ---------------------------------------------------------------------------

def _tokenize_and_filter(
    ds: hf_datasets.Dataset,
    adapter: TaskAdapter,
    tokenizer,
    max_prompt_length: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[int, list[int], dict, dict, dict]], int]:
    """Return (all_valid, n_skipped).

    Each element of all_valid is (pid, input_ids, reward_spec, extras, full_row)
    where ``full_row`` is kept only when the adapter is multi-turn (SQL needs
    row["prompt"] for conversation_multi_turn mode; others discard it).
    """
    logger.info("Tokenizing prompts...")
    all_valid: list[tuple[int, list[int], dict, dict, dict]] = []
    n_skipped = 0
    keep_row = adapter.is_multi_turn
    for j in range(len(ds)):
        row = ds[j]
        ids = tokenizer.apply_chat_template(
            adapter.row_prompt(row), add_generation_prompt=True, return_dict=False,
        )
        if len(ids) > max_prompt_length:
            n_skipped += 1
            continue
        # Multi-turn adapters may need to reject rows where the env can't
        # be constructed (e.g. SQL db file missing). Cheap pre-flight.
        if adapter.is_multi_turn:
            try:
                _probe = adapter.make_env(row, args)
                del _probe
            except FileNotFoundError:
                n_skipped += 1
                continue
        all_valid.append((
            j, ids, row["reward_spec"], adapter.row_extras(row),
            row if keep_row else {},
        ))
    logger.info(f"Valid: {len(all_valid)}, skipped: {n_skipped}")
    return all_valid, n_skipped


# ---------------------------------------------------------------------------
# Single-turn driver
# ---------------------------------------------------------------------------

def _run_single_turn(
    adapter: TaskAdapter,
    all_valid: list,
    llm,
    sampling_params,
    args: argparse.Namespace,
) -> dict[int, list[float]]:
    """Batched ``llm.generate`` with native n=n_samples; grade each completion."""
    from vllm import SamplingParams  # noqa: F401 — reassurance for type checker

    batch_size = args.batch_size
    n_batches = (len(all_valid) + batch_size - 1) // batch_size

    rewards_by_problem: dict[int, list[float]] = defaultdict(list)

    # Subprocess pool is only created if the adapter actually needs it.
    grade_pool: ThreadPoolExecutor | None = None
    if adapter.uses_subprocess_grading and not args.skip_eval:
        grade_pool = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4),
        )

    try:
        for batch_idx, start in enumerate(range(0, len(all_valid), batch_size)):
            batch = all_valid[start : start + batch_size]
            pids = [b[0] for b in batch]
            prompts = [{"prompt_token_ids": b[1]} for b in batch]
            rs_list = [b[2] for b in batch]
            extras_list = [b[3] for b in batch]

            logger.info(
                f"Batch {batch_idx + 1}/{n_batches}: "
                f"{len(prompts)} problems × {args.n_samples} samples"
            )
            outputs = llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False,
            )

            if args.skip_eval:
                if args.dump_gen_dir:
                    adapter.write_dump_batch(
                        args.dump_gen_dir, batch_idx, pids, outputs, rs_list, extras_list,
                    )
                continue

            if grade_pool is not None:
                # Queue every completion through the subprocess grader.
                submissions = []  # (pid, future)
                for output, pid, reward_spec, extras in zip(
                    outputs, pids, rs_list, extras_list,
                ):
                    for completion in output.outputs:
                        fut = grade_pool.submit(
                            _grade_via_subprocess,
                            adapter.name, completion.text, reward_spec,
                            extras, 10,
                        )
                        submissions.append((pid, fut))
                for pid, fut in submissions:
                    rewards_by_problem[pid].append(float(fut.result()))
            else:
                # Inline grading (code: pytest; safe from sympy hangs).
                for output, pid, reward_spec, extras in zip(
                    outputs, pids, rs_list, extras_list,
                ):
                    for completion in output.outputs:
                        reward = adapter.grade_one(
                            completion.text, reward_spec, extras,
                        )
                        rewards_by_problem[pid].append(float(reward))

            if rewards_by_problem:
                n_done = len(rewards_by_problem)
                p1 = sum(v[0] > 0 for v in rewards_by_problem.values()) / n_done
                logger.info(
                    f"  [{n_done} problems so far] pass@1={p1:.4f}"
                )
    finally:
        if grade_pool is not None:
            grade_pool.shutdown(wait=False, cancel_futures=True)

    return rewards_by_problem


# ---------------------------------------------------------------------------
# Multi-turn driver (SQL)
# ---------------------------------------------------------------------------

@dataclass
class _ProblemState:
    """Mutable state for one (problem, sample) during multi-turn eval."""
    pid: int
    input_ids: list[int]
    conversation: list          # only populated in conversation_multi_turn mode
    env: Any                    # SQLEnv
    done: bool = False
    reward: float = 0.0
    n_turns_used: int = 0
    final_response: str = ""    # last assistant message (for optional dump)


def _run_multi_turn_sql(
    adapter: TaskAdapter,
    all_valid: list,
    llm,
    sampling_params,
    tokenizer,
    args: argparse.Namespace,
) -> dict[int, list[float]]:
    """Multi-turn SQL eval: lift of synsql's state-machine loop, with
    ``final_response`` tracking for optional dump round-trip compatibility.
    """
    # Precompute generation_prompt_ids for conversation_multi_turn mode
    if args.conversation_multi_turn:
        base_conversation = [{"role": "user", "content": ""}]
        base_ids = tokenizer.apply_chat_template(
            base_conversation, add_generation_prompt=True, return_dict=False,
        )
        base_no_gen = tokenizer.apply_chat_template(
            base_conversation, add_generation_prompt=False, return_dict=False,
        )
        # generation_prompt_ids is the delta (unused below — kept for parity
        # with the original script). Retained so the block's side-effects are
        # preserved if the tokenizer ever relies on mutating internal caches.
        _ = base_ids[len(base_no_gen):]

    # Build initial states: expand by n_samples
    all_states: list[_ProblemState] = []
    per_state_meta: list[dict] = []   # parallel list for dump writing (reward_spec + extras + row)
    for seq_id, ids, reward_spec, extras, row in all_valid:
        for _ in range(args.n_samples):
            all_states.append(
                _ProblemState(
                    pid=seq_id,
                    input_ids=list(ids),
                    conversation=copy.deepcopy(row["prompt"]) if args.conversation_multi_turn else [],
                    env=adapter.make_env(row, args),
                )
            )
            per_state_meta.append({
                "reward_spec": reward_spec, "extras": extras, "row": row,
            })

    active_indices = list(range(len(all_states)))

    for turn in range(args.max_turns):
        if not active_indices:
            break
        logger.info(
            f"Turn {turn + 1}/{args.max_turns}: {len(active_indices)} active samples"
        )

        for batch_start in tqdm(range(0, len(active_indices), args.batch_size)):
            batch_indices = active_indices[batch_start : batch_start + args.batch_size]
            batch_token_ids = [all_states[si].input_ids for si in batch_indices]

            prompts = [{"prompt_token_ids": toks} for toks in batch_token_ids]
            outputs = llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False,
            )

            for local_idx, si in enumerate(batch_indices):
                state = all_states[si]
                completion = outputs[local_idx].outputs[0]
                response = completion.text
                response_ids = completion.token_ids

                step_out = state.env.step(response)
                state.n_turns_used += 1

                if step_out["done"]:
                    state.done = True
                    state.reward = float(step_out["reward"])
                    state.final_response = response
                else:
                    observations = step_out["observations"]
                    if args.conversation_multi_turn:
                        state.conversation.append({"role": "assistant", "content": response})
                        for obs in observations:
                            state.conversation.append(obs)
                        state.input_ids = tokenizer.apply_chat_template(
                            state.conversation, add_generation_prompt=True, return_dict=False,
                        )
                    else:
                        resp_ids = list(response_ids)
                        if resp_ids and resp_ids[-1] == tokenizer.eos_token_id:
                            resp_ids = resp_ids[:-1]
                        obs_ids = []
                        for obs in observations:
                            obs_ids.extend(
                                tokenizer.encode(obs["content"], add_special_tokens=False)
                            )
                        state.input_ids = state.input_ids + resp_ids + obs_ids

                    if len(state.input_ids) > args.max_prompt_length:
                        logger.debug(
                            f"Problem {state.pid} exceeded max_prompt_length at turn "
                            f"{turn + 1}, force-terminating."
                        )
                        state.done = True
                        state.reward = 0.0
                        state.final_response = response

        active_indices = [si for si in active_indices if not all_states[si].done]

    # Any samples still active after all turns are marked as failures
    for si in active_indices:
        state = all_states[si]
        if not state.done:
            state.done = True
            state.reward = 0.0

    # Dump final responses per sample (if requested).
    if args.skip_eval and args.dump_gen_dir:
        from examples.train.eval.tasks import SQLTaskAdapter
        # Group states back into batches of problems (same batching as synsql
        # used historically: simply sharded by SAMPLE index, batch_size states
        # per dump file). The downstream grade_dumps.py reads all *.jsonl in a
        # dir anyway, so batch boundaries are cosmetic.
        from collections import defaultdict as _dd
        by_pid: dict[int, list[_ProblemState]] = _dd(list)
        for s in all_states:
            by_pid[s.pid].append(s)

        records = []
        for pid in sorted(by_pid.keys()):
            for sample_idx, state in enumerate(by_pid[pid]):
                meta_idx = all_states.index(state)  # O(n) but n is small
                meta = per_state_meta[meta_idx]
                row = meta["row"]
                records.append({
                    "pid": int(pid),
                    "sample_idx": sample_idx,
                    "response": state.final_response,
                    "reward_spec": meta["reward_spec"],
                    "db_id": row["db_id"],
                    "data": row["data"],
                    "max_turns": args.max_turns,
                })
        SQLTaskAdapter.write_sql_dump_batch(args.dump_gen_dir, 0, records)

    # Collect rewards per problem
    rewards_by_problem: dict[int, list[float]] = defaultdict(list)
    for state in all_states:
        rewards_by_problem[state.pid].append(float(state.reward))

    # Diagnostic
    all_rewards = [s.reward for s in all_states]
    dist = Counter(all_rewards)
    logger.info(f"[DIAG] Reward distribution: {dict(sorted(dist.items()))}")
    turns_dist = Counter(s.n_turns_used for s in all_states)
    logger.info(f"[DIAG] Turns used distribution: {dict(sorted(turns_dist.items()))}")

    return rewards_by_problem


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(
    rewards_by_problem: dict[int, list[float]],
    n_samples: int,
) -> dict:
    n_total = len(rewards_by_problem)
    if n_total == 0:
        return {"n_problems": 0}

    pass_at_1 = sum(v[0] > 0 for v in rewards_by_problem.values()) / n_total
    avg_reward = sum(sum(v) / len(v) for v in rewards_by_problem.values()) / n_total
    reward_distribution = Counter(r for v in rewards_by_problem.values() for r in v)

    out: dict = {
        "n_problems": n_total,
        "pass_at_1": pass_at_1,
        "avg_reward": avg_reward,
        "reward_distribution": {str(k): v for k, v in sorted(reward_distribution.items())},
    }
    if n_samples > 1:
        pass_at_k = sum(any(r > 0 for r in v) for v in rewards_by_problem.values()) / n_total
        mean_sample_acc = sum(
            sum(r > 0 for r in v) / len(v) for v in rewards_by_problem.values()
        ) / n_total
        out.update({
            "n_samples": n_samples,
            "pass_at_k": pass_at_k,
            "mean_sample_acc": mean_sample_acc,
        })
    return out


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def _build_arg_parser(adapter: TaskAdapter) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified eval entry point for TinyLoRA adapters.",
    )
    # --task was already consumed by the two-phase parse; add it here for --help.
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--checkpoint_dir", default=None,
                   help="SkyRL checkpoint dir (searches for custom_lora/custom_lora_params.safetensors)")
    p.add_argument("--lora_adapter", default=None,
                   help="Path to a TinyLoRA .safetensors file (takes precedence over --checkpoint_dir)")
    p.add_argument("--base_model_path", required=True,
                   help="HF repo id or local path of the base model")
    p.add_argument("--data_path", default=None,
                   help="Optional local parquet. Omit to download from HF hub.")
    p.add_argument("--num_problems", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_gpus", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--max_prompt_length", type=int, default=None,
                   help=f"Default: {adapter.default_max_prompt_length} for --task {adapter.name}")
    p.add_argument("--max_generate_length", type=int, default=None,
                   help=f"Default: {adapter.default_max_generate_length} for --task {adapter.name}")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy. >0 with --n_samples for pass@k.")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=None,
                   help=f"Default: {adapter.default_batch_size} for --task {adapter.name}")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--enable_prefix_caching", type=lambda x: x.lower() != "false",
                   default=None, metavar="BOOL",
                   help="Default: False. vLLM 0.17's Mamba prefix cache is "
                        "experimental and crashes on hybrid Mamba-Attention "
                        "models like Qwen3.5; opt in only for models where "
                        "it works.")
    p.add_argument("--output_path", default=None, help="Write metrics JSON here")
    p.add_argument("--skip_eval", default=False, action="store_true",
                   help="Skip grading; dump generations only (requires --dump_gen_dir)")
    p.add_argument("--dump_gen_dir", default=None,
                   help="Directory to write generation dumps (one file per batch)")
    # Custom LoRA config
    p.add_argument("--svd_rank", type=int, default=16)
    p.add_argument("--num_coefficients", type=int, default=16)
    p.add_argument("--projection_seed", type=int, default=42)
    p.add_argument("--scheme", default="svd_random_projection")
    p.add_argument("--target_modules", default="all-linear")
    p.add_argument("--exclude_modules", default=None)
    # Task-specific flags
    adapter.add_cli(p)
    return p


def main() -> None:
    # Dispatch the grading-helper mode before argparse so helper subprocess
    # invocations don't need to pass the normal required args.
    if "--_grade_helper" in sys.argv:
        _run_grade_helper_mode()
        return

    # Two-phase argparse: read --task first, then let the adapter contribute flags.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--task", choices=TASK_CHOICES)
    known, _ = bootstrap.parse_known_args()
    if known.task is None:
        # Let the full parser report a helpful error.
        adapter = get_adapter("math")
        parser = _build_arg_parser(adapter)
        parser.parse_args()  # will exit with error about missing --task
        return

    adapter = get_adapter(known.task)
    parser = _build_arg_parser(adapter)
    args = parser.parse_args()

    # Resolve task-specific defaults for flags the user didn't set.
    if args.max_prompt_length is None:
        args.max_prompt_length = adapter.default_max_prompt_length
    if args.max_generate_length is None:
        args.max_generate_length = adapter.default_max_generate_length
    if args.batch_size is None:
        args.batch_size = adapter.default_batch_size

    if args.n_samples > 1 and args.temperature == 0.0:
        logger.warning(
            "n_samples > 1 with temperature=0 gives identical samples; consider --temperature 0.6"
        )
    if args.skip_eval and not args.dump_gen_dir:
        logger.warning(
            "--skip_eval set without --dump_gen_dir: generations will be discarded."
        )

    tmp_dir = tempfile.mkdtemp(prefix="eval_ckpt_")
    try:
        # ---- resolve model path: merge adapter if given, else use base as-is ----
        if args.lora_adapter is None and args.checkpoint_dir is None:
            logger.info(
                f"No adapter provided — evaluating base model as-is: {args.base_model_path}"
            )
            model_path = args.base_model_path
        else:
            model_path = merge_utils.resolve_model_path(
                base_model_path=args.base_model_path,
                svd_rank=args.svd_rank,
                num_coefficients=args.num_coefficients,
                projection_seed=args.projection_seed,
                scheme=args.scheme,
                target_modules=args.target_modules,
                exclude_modules=args.exclude_modules,
                tmp_dir=tmp_dir,
                checkpoint_dir=args.checkpoint_dir,
                adapter_path=args.lora_adapter,
            )

        # ---- tokenizer + dataset ----
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        ds_full = adapter.load_dataset(args)
        ds = random_subset(ds_full, args.num_problems, args.seed)
        logger.info(f"Evaluating on {len(ds)} problems (seed={args.seed})")

        all_valid, n_skipped = _tokenize_and_filter(
            ds, adapter, tokenizer, args.max_prompt_length, args,
        )
        del ds, ds_full
        gc.collect()

        # ---- vLLM ----
        enable_prefix_caching = args.enable_prefix_caching
        if enable_prefix_caching is None:
            enable_prefix_caching = False
        if args.n_samples > 1 and not enable_prefix_caching:
            logger.warning(
                "n_samples > 1 without prefix caching: prompt KV computed separately per sample. "
                "Pass --enable_prefix_caching true to opt in (crashes on Qwen3.5 hybrid Mamba-Attention)."
            )

        logger.info(
            f"Initializing vLLM engine (TP={args.num_gpus}, model={model_path})"
        )
        llm = LLM(
            model=model_path,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_prompt_length + args.max_generate_length,
            trust_remote_code=True,
            seed=args.seed,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=False,
            language_model_only=True,
        )

        sampling_params_kwargs = dict(
            temperature=args.temperature,
            max_tokens=args.max_generate_length,
            top_p=1.0,
        )
        if adapter.is_multi_turn:
            # Multi-turn tasks expand samples into states up front (n=1 per call).
            if adapter.stop_tokens:
                sampling_params_kwargs["stop"] = adapter.stop_tokens
                sampling_params_kwargs["include_stop_str_in_output"] = True
        else:
            sampling_params_kwargs["n"] = args.n_samples
        sampling_params = SamplingParams(**sampling_params_kwargs)

        # ---- drive ----
        if adapter.is_multi_turn:
            rewards_by_problem = _run_multi_turn_sql(
                adapter, all_valid, llm, sampling_params, tokenizer, args,
            )
        else:
            rewards_by_problem = _run_single_turn(
                adapter, all_valid, llm, sampling_params, args,
            )

        # ---- metrics ----
        if args.skip_eval:
            metrics = {
                "task": args.task,
                "n_records_dumped": sum(len(v) for v in rewards_by_problem.values()),
                "n_skipped": n_skipped,
                "skip_eval": True,
                "dump_gen_dir": args.dump_gen_dir,
            }
        else:
            metrics = _aggregate_metrics(rewards_by_problem, args.n_samples)
            metrics.update({
                "task": args.task,
                "base_model_path": args.base_model_path,
                "lora_adapter": args.lora_adapter,
                "checkpoint_dir": args.checkpoint_dir,
                "n_skipped": n_skipped,
                "seed": args.seed,
                "temperature": args.temperature,
                "per_problem": {
                    str(pid): {
                        "rewards": rewards,
                        "samples": [r > 0 for r in rewards],
                    }
                    for pid, rewards in sorted(rewards_by_problem.items())
                },
            })
            msg = f"pass@1={metrics['pass_at_1']:.4f}  avg_reward={metrics['avg_reward']:.4f}"
            if args.n_samples > 1:
                msg += (
                    f"  pass@{args.n_samples}={metrics['pass_at_k']:.4f}  "
                    f"mean_sample_acc={metrics['mean_sample_acc']:.4f}"
                )
            logger.info(msg + f"  (n={metrics['n_problems']})")

        if args.output_path:
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Wrote metrics to {args.output_path}")

        if not args.skip_eval:
            print(
                f"\npass@1: {metrics['pass_at_1']:.4f}  "
                f"avg_reward: {metrics['avg_reward']:.4f}  "
                f"(n={metrics['n_problems']})"
            )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
