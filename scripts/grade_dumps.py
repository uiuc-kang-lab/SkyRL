"""Grade dumped generation results from the eval_checkpoint scripts.

Reads existing dumps written by ``eval_checkpoint.py --skip_eval --dump_gen_dir <dir>``
and re-grades them with the appropriate environment for the task, then reports
pass@1, pass@k, and mean-sample accuracy.

Supported input formats (auto-detected per file; any mix is fine):

  (A) ``mix_general`` style: ``batch_*_outputs.json`` — a JSON **list**, one
      record per completion::

          [
            {"pid": int, "response": str, "reward_spec": dict},
            ...
          ]

  (B) ``mix_code`` style: ``batch_*_outputs.json`` — a JSON **dict** with
      parallel arrays::

          {
            "problem_ids":  [int, ...],
            "responses":    [[str, ...], ...],   # outer per-prompt, inner per-sample
            "reward_specs": [dict, ...],
          }

  (C) JSONL (``batch_*.jsonl``): one record per line in schema (A), optionally
      with a ``sample_idx`` field and task-specific extras (``db_id``,
      ``data``, ``max_turns`` for SQL). This format is not currently emitted
      by any eval_checkpoint.py but is accepted so callers can pre-process
      trajectories (e.g. for the ``sql`` task) into it.

The ``--task`` flag selects the grading environment; it cannot be inferred
from the dump files themselves since none of them encode it today.

Each record is graded in a fresh ``subprocess.run`` with a hard timeout.
In-process SIGALRM and ``multiprocessing.Process`` kills are both unreliable
for ``eurus_math``: math_verify arms its own SIGALRM and clears it on exit
(cancelling outer timers), sympy/LaTeX paths run in C code that ignores
Python signals, and forking from a spawn-based pool worker inherits Python
internal thread locks that can deadlock the child during import. A real
OS subprocess with a deadline sidesteps all of that.

Usage:
    uv run python scripts/grade_dumps.py \\
        --task mix_general \\
        --dump_dir /path/to/dump_gen_dir \\
        --output_path results.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm


TASK_CHOICES = ["eurus_math", "mix_general", "mix_code", "sql"]

# Subprocess helpers need the project root on sys.path so that
# ``examples.train.*`` imports resolve.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------

def _records_from_mix_general(obj: list) -> list[dict]:
    """mix_general format: list of {pid, response, reward_spec}."""
    out = []
    per_pid_counter: dict[int, int] = defaultdict(int)
    for entry in obj:
        pid = int(entry["pid"])
        sample_idx = per_pid_counter[pid]
        per_pid_counter[pid] += 1
        out.append({
            "pid": pid,
            "sample_idx": sample_idx,
            "response": entry["response"],
            "reward_spec": entry["reward_spec"],
        })
    return out


def _records_from_mix_code(obj: dict) -> list[dict]:
    """mix_code format: dict with parallel arrays problem_ids / responses / reward_specs."""
    pids = obj["problem_ids"]
    responses = obj["responses"]
    reward_specs = obj["reward_specs"]
    if not (len(pids) == len(responses) == len(reward_specs)):
        raise ValueError(
            f"mix_code dump arrays length mismatch: "
            f"pids={len(pids)} responses={len(responses)} reward_specs={len(reward_specs)}"
        )
    out = []
    for pid, samples, reward_spec in zip(pids, responses, reward_specs):
        # `samples` may be a list (multi-sample) or a bare string (one sample).
        if isinstance(samples, str):
            samples = [samples]
        for sample_idx, response in enumerate(samples):
            out.append({
                "pid": int(pid),
                "sample_idx": sample_idx,
                "response": response,
                "reward_spec": reward_spec,
            })
    return out


def _records_from_jsonl(path: str) -> list[dict]:
    """JSONL: one JSON record per line."""
    out = []
    per_pid_counter: dict[int, int] = defaultdict(int)
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: malformed JSON: {e}") from e
            pid = int(entry["pid"])
            if "sample_idx" in entry:
                sample_idx = int(entry["sample_idx"])
            else:
                sample_idx = per_pid_counter[pid]
                per_pid_counter[pid] += 1
            record = {
                "pid": pid,
                "sample_idx": sample_idx,
                "response": entry["response"],
                "reward_spec": entry["reward_spec"],
            }
            # pass through SQL-specific extras if present
            for k in ("db_id", "data", "max_turns"):
                if k in entry:
                    record[k] = entry[k]
            out.append(record)
    return out


def load_records(dump_dir: str) -> list[dict]:
    json_files = sorted(glob.glob(os.path.join(dump_dir, "*.json")))
    jsonl_files = sorted(glob.glob(os.path.join(dump_dir, "*.jsonl")))
    if not json_files and not jsonl_files:
        raise FileNotFoundError(f"No *.json or *.jsonl files found in {dump_dir}")

    records: list[dict] = []
    for path in json_files:
        with open(path) as fh:
            obj = json.load(fh)
        if isinstance(obj, list):
            records.extend(_records_from_mix_general(obj))
        elif isinstance(obj, dict) and "responses" in obj and "problem_ids" in obj:
            records.extend(_records_from_mix_code(obj))
        else:
            raise ValueError(
                f"{path}: unrecognized dump format "
                f"(expected either a list of completion dicts or a "
                f"{{problem_ids, responses, reward_specs}} dict)"
            )
    for path in jsonl_files:
        records.extend(_records_from_jsonl(path))
    return records


# ---------------------------------------------------------------------------
# Per-record grading (runs in a worker process)
# ---------------------------------------------------------------------------

def _build_env_and_step(task: str, record: dict, timeout: int):
    response = record["response"]

    if task == "eurus_math":
        from examples.train.eurus_rl_math.env import MathEnv
        env = MathEnv(extras={"reward_spec": record["reward_spec"]},
                      math_verify_timeout=timeout)
        return env.step(response)

    if task == "mix_general":
        from examples.train.mix_general.env import GeneralEnv
        env = GeneralEnv(extras={"reward_spec": record["reward_spec"]},
                         math_verify_timeout=timeout)
        return env.step(response)

    if task == "mix_code":
        from examples.train.mix_code.env import MixCodeEnv
        env = MixCodeEnv(extras={"reward_spec": record["reward_spec"]},
                         env_config={"timeout": 20})
        return env.step(response)

    if task == "sql":
        from skyrl_gym.envs.sql.env import SQLEnv, Text2SQLEnvConfig
        for key in ("db_id", "data"):
            if key not in record:
                raise KeyError(
                    f"sql grading requires {key!r} in each record; "
                    f"pid={record.get('pid')} sample_idx={record.get('sample_idx')}"
                )
        env_cfg = Text2SQLEnvConfig(db_path=record["db_path"])
        env = SQLEnv(
            env_config=env_cfg,
            extras={
                "db_id": record["db_id"],
                "reward_spec": record["reward_spec"],
                "data": record["data"],
                "max_turns": record.get("max_turns", 6),
            },
        )
        return env.step(response)

    raise ValueError(f"Unknown task: {task}")


def _run_helper_mode() -> None:
    """Read {task, record, timeout} from stdin, grade once, write result to stdout.

    Invoked by the main driver via ``subprocess.run`` so that each grading
    call runs in a fresh Python interpreter the driver can SIGKILL on timeout.
    """
    try:
        payload = json.loads(sys.stdin.read())
        step_out = _build_env_and_step(
            payload["task"], payload["record"], payload["timeout"],
        )
        correct = float(step_out["reward"]) > 0.0
    except BaseException:
        correct = False
    sys.stdout.write(json.dumps({"correct": bool(correct)}) + "\n")
    sys.stdout.flush()


def _grade_one(args_tuple):
    """Grade one record in a fresh subprocess; real OS-level kill on timeout."""
    task, record, timeout = args_tuple
    pid = record["pid"]
    sample_idx = record.get("sample_idx", 0)

    # Give the inner math_verify_timeout time to self-abort first; fall back
    # to subprocess SIGKILL if the whole interpreter is wedged.
    hard_timeout = max((timeout * 3) if (timeout and timeout > 0) else 30, 30)

    payload = json.dumps({"task": task, "record": record, "timeout": timeout})

    correct = False
    try:
        res = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--_helper"],
            input=payload,
            capture_output=True,
            timeout=hard_timeout,
            text=True,
            cwd=_PROJ_ROOT,
        )
        if res.returncode == 0 and res.stdout:
            # The helper may emit library logs before the result line; take
            # the last parseable JSON line.
            for line in reversed(res.stdout.strip().splitlines()):
                try:
                    obj = json.loads(line)
                    correct = bool(obj.get("correct", False))
                    break
                except json.JSONDecodeError:
                    continue
    except subprocess.TimeoutExpired:
        correct = False
    except BaseException:
        correct = False

    return pid, sample_idx, correct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Dispatch helper mode before argparse so required args aren't demanded
    # of the helper subprocess invocation.
    if "--_helper" in sys.argv:
        _run_helper_mode()
        return

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--dump_dir", required=True,
                   help="Directory containing batch_*.json / batch_*.jsonl dumps")
    p.add_argument("--db_path", default="./databases",
                   help="(sql only) root path containing per-db folders; "
                        "injected into each record unless already set.")
    p.add_argument("--output_path", default=None,
                   help="Write metrics JSON here. Omit to print-only.")
    p.add_argument("--timeout", type=int, default=10,
                   help="Per-response math_verify timeout (seconds). "
                        "Hard subprocess deadline is max(3×timeout, 30).")
    p.add_argument("--max_workers", type=int, default=None,
                   help="Parallel in-flight subprocess count "
                        "(default min(8, cpu_count)).")
    args = p.parse_args()

    records = load_records(args.dump_dir)
    logger.info(f"Loaded {len(records)} records from {args.dump_dir}")

    if args.task == "sql":
        for r in records:
            r.setdefault("db_path", args.db_path)

    n_workers = args.max_workers or min(8, os.cpu_count() or 4)

    # Threads, not processes: each grading call is a subprocess.run, and
    # that's where the hard timeout lives. Threads only need to wait on the
    # subprocess — subprocess.run releases the GIL, so parallelism is fine.
    per_problem: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    work = [(args.task, r, args.timeout) for r in records]
    pool = ThreadPoolExecutor(max_workers=n_workers)
    futures = [pool.submit(_grade_one, w) for w in work]
    try:
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Grade[{args.task}]",
            dynamic_ncols=True,
        ):
            pid, sample_idx, correct = fut.result()
            per_problem[pid].append((sample_idx, correct))
    except KeyboardInterrupt:
        for fut in futures:
            fut.cancel()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results_by_problem = {
        pid: [c for _, c in sorted(entries)] for pid, entries in per_problem.items()
    }

    n_total = len(results_by_problem)
    if n_total == 0:
        logger.error("No records graded; nothing to report.")
        return

    pass_at_1 = sum(v[0] for v in results_by_problem.values()) / n_total
    n_samples = max(len(v) for v in results_by_problem.values())

    metrics: dict = {
        "task": args.task,
        "dump_dir": args.dump_dir,
        "n_problems": n_total,
        "n_records": len(records),
        "pass_at_1": pass_at_1,
    }

    if n_samples > 1:
        pass_at_k = sum(any(v) for v in results_by_problem.values()) / n_total
        mean_acc = sum(sum(v) / len(v) for v in results_by_problem.values()) / n_total
        metrics.update({
            "n_samples": n_samples,
            "pass_at_k": pass_at_k,
            "mean_sample_acc": mean_acc,
        })
        logger.info(
            f"pass@1={pass_at_1:.4f}  pass@{n_samples}={pass_at_k:.4f}  "
            f"mean_acc={mean_acc:.4f}  (n={n_total})"
        )
    else:
        logger.info(f"pass@1={pass_at_1:.4f}  (n={n_total})")

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Wrote metrics to {args.output_path}")


if __name__ == "__main__":
    main()
