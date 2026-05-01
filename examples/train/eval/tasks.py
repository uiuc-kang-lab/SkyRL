"""Task adapters for the unified eval entry point.

Each adapter owns the task-specific pieces of eval: dataset loading, per-row
prompt/extras extraction, the grading environment, and the dump-file format.

Single-turn tasks (math, code, general) implement ``grade_one``. The multi-turn
task (sql) implements ``make_env`` and is consumed by the multi-turn driver in
``eval_checkpoint.py``, which owns the turn loop.

Dump formats intentionally stay heterogeneous — each adapter preserves the
format written by its pre-merge script, so existing consumers (notably
``scripts/grade_dumps.py``, which auto-detects) keep working.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any

import datasets as hf_datasets
from loguru import logger


# ---------------------------------------------------------------------------
# Generic dataset helpers
# ---------------------------------------------------------------------------

def random_subset(
    ds: hf_datasets.Dataset, num_problems: int, seed: int,
) -> hf_datasets.Dataset:
    if num_problems is None or num_problems >= len(ds):
        if num_problems is not None:
            logger.info(f"Requested {num_problems} problems but dataset only has {len(ds)}; using all.")
        return ds
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), num_problems))
    return ds.select(indices)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class TaskAdapter:
    """Protocol-ish base; concrete subclasses override what they need."""

    name: str = ""
    is_multi_turn: bool = False
    uses_subprocess_grading: bool = False
    stop_tokens: list[str] | None = None

    # Defaults for shared CLI flags that vary by task. The driver in
    # eval_checkpoint.py uses these when the user doesn't pass an explicit
    # value (argparse default=None → resolved here).
    default_max_prompt_length: int = 5120
    default_max_generate_length: int = 4096
    default_batch_size: int = 64

    # ---- CLI ----
    def add_cli(self, p: argparse.ArgumentParser) -> None:
        """Register task-specific CLI flags. Default: nothing."""
        return

    # ---- Data ----
    def load_dataset(self, args: argparse.Namespace) -> hf_datasets.Dataset:
        raise NotImplementedError

    def row_prompt(self, row: dict) -> list[dict]:
        """Return ChatML-style prompt for ``tokenizer.apply_chat_template``."""
        return row["prompt"]

    def row_extras(self, row: dict) -> dict:
        """Task-specific extras stored alongside reward_spec. Default: ``{}``."""
        return {}

    # ---- Grading: single-turn tasks ----
    def grade_one(self, response: str, reward_spec: dict, extras: dict) -> float:
        """Run the grading env on one response, return the raw float reward."""
        raise NotImplementedError

    # ---- Grading: multi-turn tasks ----
    def make_env(self, row: dict, args: argparse.Namespace):
        """Construct a per-sample env for multi-turn tasks."""
        raise NotImplementedError

    # ---- Dumps ----
    def write_dump_batch(
        self,
        dump_gen_dir: str,
        batch_idx: int,
        pids: list[int],
        outputs,  # vllm outputs (list of RequestOutput) — single-turn adapters only
        reward_specs: list[dict],
        extras_list: list[dict],
    ) -> None:
        """Write one batch of generations to ``dump_gen_dir``.

        Multi-turn adapters (SQL) don't use this path; they write JSONL
        records via ``write_sql_dump_batch`` after the turn loop instead.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# math (EURUS)
# ---------------------------------------------------------------------------

class MathTaskAdapter(TaskAdapter):
    name = "math"
    is_multi_turn = False
    uses_subprocess_grading = True   # math_verify/sympy can hang — needs real SIGKILL
    stop_tokens = None

    def load_dataset(self, args) -> hf_datasets.Dataset:
        if args.data_path:
            logger.info(f"Loading dataset from {args.data_path}")
            return hf_datasets.load_dataset(
                "parquet", data_files=args.data_path, keep_in_memory=True,
            )["train"]

        logger.info("Downloading EURUS dataset from HF hub (uiuc-kang-lab/RLVR-Eurus-2-Math-Fixed)...")
        ds = hf_datasets.load_dataset(
            "uiuc-kang-lab/RLVR-Eurus-2-Math-Fixed", "default", split="train",
        )
        ds = ds.filter(
            lambda x: x["ability"] == "math" and x["reward_model"]["style"] == "rule"
        )

        def _to_parquet_schema(ex):
            prompt = [{"role": "user", "content": ex["prompt"][-1]["content"]}]
            return {
                "prompt": prompt,
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": ex["reward_model"]["ground_truth"],
                },
                "extra_info": {"question": ex["prompt"][-1]["content"]},
            }

        return ds.map(_to_parquet_schema, remove_columns=ds.column_names)

    def grade_one(self, response: str, reward_spec: dict, extras: dict) -> float:
        from examples.train.eurus_rl_math.env import MathEnv
        env = MathEnv(extras={"reward_spec": reward_spec}, math_verify_timeout=10)
        step_out = env.step(response)
        return float(step_out["reward"])

    def write_dump_batch(
        self, dump_gen_dir, batch_idx, pids, outputs, reward_specs, extras_list,
    ) -> None:
        # List-of-records-per-batch-file format (shared with mix_general).
        records = []
        for output, pid, reward_spec in zip(outputs, pids, reward_specs):
            for completion in output.outputs:
                records.append({
                    "pid": int(pid),
                    "response": completion.text,
                    "reward_spec": reward_spec,
                })
        os.makedirs(dump_gen_dir, exist_ok=True)
        path = os.path.join(dump_gen_dir, f"batch_{batch_idx + 1}_outputs.json")
        with open(path, "w") as f:
            json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# code (mix_code)
# ---------------------------------------------------------------------------

class CodeTaskAdapter(TaskAdapter):
    name = "code"
    is_multi_turn = False
    uses_subprocess_grading = False  # pytest-based; no sympy hang risk
    stop_tokens = None

    def load_dataset(self, args) -> hf_datasets.Dataset:
        if args.data_path is None:
            logger.info("Loading mix_code dataset from HF hub (this may take a while)...")
            return hf_datasets.load_dataset(
                "uiuc-kang-lab/RLVR-Code-Mix", split="train",
            )

        logger.info(f"Loading dataset from {args.data_path}")
        import pyarrow.parquet as pq

        def _parquet_row_iter():
            pf = pq.ParquetFile(args.data_path)
            for batch in pf.iter_batches(batch_size=10_000):
                yield from batch.to_pylist()

        return hf_datasets.Dataset.from_generator(_parquet_row_iter)

    def row_extras(self, row: dict) -> dict:
        return row.get("extra_info", {}) or {}

    def grade_one(self, response: str, reward_spec: dict, extras: dict) -> float:
        from examples.train.mix_code.env import MixCodeEnv
        env = MixCodeEnv(extras={"reward_spec": reward_spec}, env_config={"timeout": 20})
        step_out = env.step(response)
        return float(step_out["reward"])

    def write_dump_batch(
        self, dump_gen_dir, batch_idx, pids, outputs, reward_specs, extras_list,
    ) -> None:
        # Parallel-array-dict format (preserved from original mix_code).
        dump = {
            "problem_ids": list(pids),
            "responses": [[c.text for c in o.outputs] for o in outputs],
            "reward_specs": list(reward_specs),
        }
        os.makedirs(dump_gen_dir, exist_ok=True)
        path = os.path.join(dump_gen_dir, f"batch_{batch_idx + 1}_outputs.json")
        with open(path, "w") as f:
            json.dump(dump, f, indent=2)


# ---------------------------------------------------------------------------
# general (mix_general)
# ---------------------------------------------------------------------------

class GeneralTaskAdapter(TaskAdapter):
    name = "general"
    is_multi_turn = False
    uses_subprocess_grading = True   # GeneralEnv uses math_verify — same hang risk as math
    stop_tokens = None

    def load_dataset(self, args) -> hf_datasets.Dataset:
        if args.data_path is None:
            logger.info("Loading mix_general dataset from HF hub (this may take a while)...")
            return hf_datasets.load_dataset(
                "uiuc-kang-lab/RLVR-General-Mix", split="train",
            )
        logger.info(f"Loading dataset from {args.data_path}")
        return hf_datasets.load_dataset(
            "parquet", data_files=args.data_path, keep_in_memory=True,
        )["train"]

    def row_extras(self, row: dict) -> dict:
        return row.get("extra_info", {}) or {}

    def grade_one(self, response: str, reward_spec: dict, extras: dict) -> float:
        from examples.train.mix_general.env import GeneralEnv
        env = GeneralEnv(extras={"reward_spec": reward_spec}, math_verify_timeout=10)
        step_out = env.step(response)
        return float(step_out["reward"])

    def write_dump_batch(
        self, dump_gen_dir, batch_idx, pids, outputs, reward_specs, extras_list,
    ) -> None:
        # List-of-records format (preserved from original mix_general).
        records = []
        for output, pid, reward_spec in zip(outputs, pids, reward_specs):
            for completion in output.outputs:
                records.append({
                    "pid": int(pid),
                    "response": completion.text,
                    "reward_spec": reward_spec,
                })
        os.makedirs(dump_gen_dir, exist_ok=True)
        path = os.path.join(dump_gen_dir, f"batch_{batch_idx + 1}_outputs.json")
        with open(path, "w") as f:
            json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# sql (synsql) — multi-turn
# ---------------------------------------------------------------------------

class SQLTaskAdapter(TaskAdapter):
    name = "sql"
    is_multi_turn = True
    uses_subprocess_grading = False  # SQLite exec, not sympy
    stop_tokens = ["</tool_call>", "</solution>"]

    # SQL prompts carry full DB schemas and are much longer than other tasks.
    # Defaults mirror the pre-merge synsql script.
    default_max_prompt_length = 29000
    default_max_generate_length = 5120
    default_batch_size = 32

    def add_cli(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--db_path", default="./databases",
            help="Root path containing per-db folders (e.g. SynSQL-2.5M/).",
        )
        p.add_argument(
            "--max_turns", type=int, default=6,
            help="Maximum conversation turns per problem.",
        )
        p.add_argument(
            "--conversation_multi_turn", action="store_true", default=False,
            help=(
                "Use multi-turn chat template (separate user/assistant messages per turn). "
                "Default is single-turn mode (all turns in one assistant message), matching "
                "generator.use_conversation_multi_turn=false in the standard training config."
            ),
        )

    def load_dataset(self, args) -> hf_datasets.Dataset:
        if args.data_path is None:
            logger.info("Loading dataset from HF hub (uiuc-kang-lab/RLVR-SynSQL-2.5M)...")
            return hf_datasets.load_dataset(
                "uiuc-kang-lab/RLVR-SynSQL-2.5M", split="train",
            )
        logger.info(f"Loading dataset: {args.data_path}")
        return hf_datasets.load_dataset(
            "parquet", data_files=args.data_path, keep_in_memory=True,
        )["train"]

    def row_extras(self, row: dict) -> dict:
        return {"db_id": row["db_id"], "data": row["data"]}

    def make_env(self, row: dict, args):
        from skyrl_gym.envs.sql.env import SQLEnv, Text2SQLEnvConfig
        return SQLEnv(
            env_config=Text2SQLEnvConfig(db_path=args.db_path),
            extras={
                "db_id": row["db_id"],
                "reward_spec": row["reward_spec"],
                "data": row["data"],
                "max_turns": args.max_turns,
            },
        )

    @staticmethod
    def write_sql_dump_batch(
        dump_gen_dir: str,
        batch_idx: int,
        records: list[dict],
    ) -> None:
        """JSONL with SQL extras — matches grade_dumps.py's ``_records_from_jsonl``."""
        os.makedirs(dump_gen_dir, exist_ok=True)
        path = os.path.join(dump_gen_dir, f"batch_{batch_idx + 1:05d}.jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "math": MathTaskAdapter,
    "code": CodeTaskAdapter,
    "general": GeneralTaskAdapter,
    "sql": SQLTaskAdapter,
}


def get_adapter(name: str) -> TaskAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown task: {name!r}. Choices: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name]()


TASK_CHOICES = sorted(_ADAPTERS.keys())
