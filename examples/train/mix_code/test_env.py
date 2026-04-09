"""
Test script for the mix_code environment.

Runs synthetic tests with known correct/incorrect solutions, then tests
all rows from the training data (with parallel workers) to verify the
pipeline never crashes and that wrong code always receives reward 0.

Usage:
    uv run --python 3.12 python -m examples.train.mix_code.test_env \
        --data_path /workspace/data/mix_code/train_data.parquet \
        --workers 8
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

from examples.train.mix_code.code_eval import compute_score
from examples.train.mix_code.env import MixCodeEnv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _md(code: str) -> str:
    """Wrap code in a markdown code block (model response format)."""
    return f"```python\n{code}\n```"


passed = 0
failed = 0


def check(name: str, reward: float, expected: float):
    global passed, failed
    ok = reward == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: reward={reward}  (expected {expected})")
    if ok:
        passed += 1
    else:
        failed += 1


# ===================================================================
# 1. Synthetic tests — "inputs" method (stdin/stdout)
# ===================================================================
def test_inputs_synthetic():
    print("\n=== Synthetic: inputs (stdin/stdout) ===")

    ground_truth = json.dumps({
        "inputs": ["3 5", "0 0", "-1 7"],
        "outputs": ["8", "0", "6"],
    })

    # Correct solution
    correct = _md("a, b = map(int, input().split())\nprint(a + b)")
    _, reward = compute_score(correct, ground_truth, "inputs")
    check("correct add program", reward, 1.0)

    # Wrong solution (multiplies instead of adding)
    wrong = _md("a, b = map(int, input().split())\nprint(a * b)")
    _, reward = compute_score(wrong, ground_truth, "inputs")
    check("wrong add program", reward, 0.0)

    # Multi-line I/O
    ground_truth_ml = json.dumps({
        "inputs": ["3\n1 2 3"],
        "outputs": ["6"],
    })
    correct_ml = _md("n = int(input())\nnums = list(map(int, input().split()))\nprint(sum(nums))")
    _, reward = compute_score(correct_ml, ground_truth_ml, "inputs")
    check("multi-line I/O", reward, 1.0)


# ===================================================================
# 2. Synthetic tests — "assertions" method (from solution import ...)
# ===================================================================
def test_assertions_synthetic():
    print("\n=== Synthetic: assertions (from solution import) ===")

    test_code = (
        "from solution import add\n"
        "assert add(1, 2) == 3\n"
        "assert add(0, 0) == 0\n"
        "assert add(-1, 7) == 6\n"
    )

    # Correct
    correct = _md("def add(a, b):\n    return a + b")
    _, reward = compute_score(correct, test_code, "assertions")
    check("correct add function", reward, 1.0)

    # Wrong
    wrong = _md("def add(a, b):\n    return a * b")
    _, reward = compute_score(wrong, test_code, "assertions")
    check("wrong add function", reward, 0.0)

    # Missing function
    missing = _md("def subtract(a, b):\n    return a - b")
    _, reward = compute_score(missing, test_code, "assertions")
    check("missing function (ImportError)", reward, 0.0)


# ===================================================================
# 3. Synthetic tests — "assertions" with bare assertions (no import)
# ===================================================================
def test_assertions_bare():
    print("\n=== Synthetic: assertions (bare, no import) ===")

    test_code = "assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0\n"

    correct = _md("def multiply(a, b):\n    return a * b")
    _, reward = compute_score(correct, test_code, "assertions")
    check("bare assertions correct", reward, 1.0)

    wrong = _md("def multiply(a, b):\n    return a + b")
    _, reward = compute_score(wrong, test_code, "assertions")
    check("bare assertions wrong", reward, 0.0)


# ===================================================================
# 4. Synthetic tests — "assertions" with stdin/stdout dict format
# ===================================================================
def test_assertions_stdin_dict():
    print("\n=== Synthetic: assertions (stdin/stdout dict fallback) ===")

    # Some KodCode entries store this dict format under method="assertions"
    test_code = str({"stdin": ["3 5", "0 0"], "stdout": ["8", "0"]})

    correct = _md("a, b = map(int, input().split())\nprint(a + b)")
    _, reward = compute_score(correct, test_code, "assertions")
    check("stdin dict correct", reward, 1.0)

    wrong = _md("a, b = map(int, input().split())\nprint(a - b)")
    _, reward = compute_score(wrong, test_code, "assertions")
    check("stdin dict wrong", reward, 0.0)


# ===================================================================
# 5. Edge cases
# ===================================================================
def test_edge_cases():
    print("\n=== Edge cases ===")

    gt = json.dumps({"inputs": ["1"], "outputs": ["1"]})

    # No code block
    _, reward = compute_score("Here is my answer: just print 1", gt, "inputs")
    check("no code block", reward, 0.0)

    # Empty response
    _, reward = compute_score("", gt, "inputs")
    check("empty response", reward, 0.0)

    # Syntax error
    _, reward = compute_score(_md("def broken(\n    ret"), gt, "inputs")
    check("syntax error", reward, 0.0)

    # Infinite loop (should timeout, not hang)
    _, reward = compute_score(_md("while True: pass"), gt, "inputs", timeout=2)
    check("infinite loop (timeout)", reward, 0.0)

    # Runtime error
    _, reward = compute_score(_md("print(1/0)"), gt, "inputs")
    check("runtime error (ZeroDivisionError)", reward, 0.0)


# ===================================================================
# 6. MixCodeEnv interface
# ===================================================================
def test_env_interface():
    print("\n=== MixCodeEnv step() interface ===")

    # inputs method
    extras_in = {
        "reward_spec": {
            "method": "inputs",
            "ground_truth": json.dumps({
                "inputs": ["5"],
                "outputs": ["25"],
            }),
        }
    }
    env = MixCodeEnv(extras=extras_in)
    result = env.step(_md("n = int(input())\nprint(n * n)"))
    check("env step inputs correct", result["reward"], 1.0)

    result = env.step(_md("n = int(input())\nprint(n + n)"))
    check("env step inputs wrong", result["reward"], 0.0)

    # assertions method
    extras_as = {
        "reward_spec": {
            "method": "assertions",
            "ground_truth": "from solution import square\nassert square(5) == 25\nassert square(0) == 0\n",
        }
    }
    env = MixCodeEnv(extras=extras_as)
    result = env.step(_md("def square(n):\n    return n * n"))
    check("env step assertions correct", result["reward"], 1.0)

    result = env.step(_md("def square(n):\n    return n + n"))
    check("env step assertions wrong", result["reward"], 0.0)


# ===================================================================
# 7. Timeout and robustness tests
# ===================================================================
def test_timeouts():
    print("\n=== Timeout and robustness ===")

    gt = json.dumps({"inputs": ["1"], "outputs": ["1"]})

    # --- inputs method ---
    # Infinite loop in code
    t0 = time.time()
    _, reward = compute_score(_md("while True: pass"), gt, "inputs", timeout=2)
    elapsed = time.time() - t0
    check(f"inputs: infinite loop ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 15, f"Took too long: {elapsed:.1f}s"

    # Sleep longer than timeout
    t0 = time.time()
    _, reward = compute_score(_md("import time; time.sleep(100)"), gt, "inputs", timeout=2)
    elapsed = time.time() - t0
    check(f"inputs: sleep(100) ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 15, f"Took too long: {elapsed:.1f}s"

    # Fork bomb (os.fork disabled by reliability_guard)
    t0 = time.time()
    _, reward = compute_score(_md("import os\nwhile True:\n    os.fork()"), gt, "inputs", timeout=2)
    elapsed = time.time() - t0
    check(f"inputs: fork bomb ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 15, f"Took too long: {elapsed:.1f}s"

    # Many test cases — verify global cap prevents extreme wait
    many_tests = json.dumps({
        "inputs": [str(i) for i in range(50)],
        "outputs": [str(i) for i in range(50)],
    })
    t0 = time.time()
    _, reward = compute_score(_md("print(input())"), many_tests, "inputs", timeout=2)
    elapsed = time.time() - t0
    check(f"inputs: 50 test cases correct ({elapsed:.1f}s)", reward, 1.0)

    # --- assertions method ---
    # Infinite loop in model code
    t0 = time.time()
    _, reward = compute_score(
        _md("def f():\n    while True: pass\nf()"),
        "from solution import f",
        "assertions", timeout=2,
    )
    elapsed = time.time() - t0
    check(f"assertions: infinite loop in model ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 15, f"Took too long: {elapsed:.1f}s"

    # Infinite loop in test function
    t0 = time.time()
    _, reward = compute_score(
        _md("def add(a,b): return a+b"),
        "from solution import add\ndef test_hang():\n    while True: pass",
        "assertions", timeout=2,
    )
    elapsed = time.time() - t0
    check(f"assertions: infinite loop in test ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 15, f"Took too long: {elapsed:.1f}s"

    # Sleep in test
    t0 = time.time()
    _, reward = compute_score(
        _md("x = 1"),
        "import time; time.sleep(100)",
        "assertions", timeout=2,
    )
    elapsed = time.time() - t0
    check(f"assertions: sleep(100) in test ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 15, f"Took too long: {elapsed:.1f}s"

    # stdin/stdout dict via assertions — many test cases with wrong code
    many_stdin = str({"stdin": [str(i) for i in range(30)], "stdout": [str(i*2) for i in range(30)]})
    t0 = time.time()
    _, reward = compute_score(_md("print('wrong')"), many_stdin, "assertions", timeout=2)
    elapsed = time.time() - t0
    check(f"assertions→stdin: 30 cases wrong ({elapsed:.1f}s)", reward, 0.0)

    # stdin/stdout dict via assertions — correct code
    _, reward = compute_score(_md("print(int(input())*2)"), many_stdin, "assertions", timeout=2)
    check("assertions→stdin: 30 cases correct", reward, 1.0)

    # Malicious: try to delete files (reliability_guard blocks)
    _, reward = compute_score(
        _md("import os; os.remove('/etc/passwd')"),
        "from solution import os",
        "assertions", timeout=2,
    )
    check("assertions: os.remove blocked", reward, 0.0)

    # Resource exhaustion: allocate huge memory
    t0 = time.time()
    _, reward = compute_score(
        _md("x = 'a' * (10**10)"),
        "from solution import x",
        "assertions", timeout=2,
    )
    elapsed = time.time() - t0
    check(f"assertions: memory bomb ({elapsed:.1f}s)", reward, 0.0)
    assert elapsed < 20, f"Took too long: {elapsed:.1f}s"


# ===================================================================
# 8. Full data scan (parallel)
# ===================================================================
GARBAGE = "```python\nx = 'this is definitely wrong'\n```"


def _eval_one(args):
    """Evaluate one example with garbage code. Returns (idx, reward, error_or_None)."""
    ground_truth, method, idx = args
    try:
        _, reward = compute_score(GARBAGE, ground_truth, method, timeout=4)
        return idx, reward, None
    except Exception as e:
        return idx, None, f"{type(e).__name__}: {e}"


def test_full_data(data_path: str, workers: int = 8):
    """Run garbage code against every row and verify reward == 0.0.

    Supports resuming: if test_failed_indices.json exists with
    ``complete: false``, skips already-processed rows.
    """
    print(f"\n=== Full data scan: {data_path}  (workers={workers}) ===")

    reader = pq.ParquetFile(data_path)
    total_rows = reader.metadata.num_rows
    print(f"Total rows: {total_rows}")

    # Pre-load all (ground_truth, method, idx) tuples
    print("  Loading specs...", end="", flush=True)
    t_load = time.time()
    tasks = []
    for batch in reader.iter_batches(batch_size=5000, columns=["reward_spec"]):
        for spec in batch.column("reward_spec"):
            tasks.append((spec["ground_truth"].as_py(), spec["method"].as_py(), len(tasks)))
    print(f" done ({len(tasks)} rows in {time.time()-t_load:.1f}s)")

    del reader

    data_dir = os.path.dirname(os.path.abspath(data_path))
    result_path = os.path.join(data_dir, "test_failed_indices.json")

    # Resume from previous partial run if available
    resume_from = 0
    data_passed = 0
    data_failed = 0
    failed_indices = []
    errors = []

    if os.path.exists(result_path):
        with open(result_path) as f:
            prev = json.load(f)
        if not prev.get("complete", False) and prev.get("processed", 0) > 0:
            resume_from = prev["processed"]
            data_passed = prev["passed"]
            data_failed = prev["failed"]
            failed_indices = list(prev.get("failed_indices", []))
            print(f"  Resuming from row {resume_from} "
                  f"(prev: {data_passed} pass, {data_failed} fail)")

    remaining_tasks = tasks[resume_from:]
    # Re-number indices to match global position
    remaining_tasks = [(gt, method, resume_from + i) for i, (gt, method, _) in enumerate(remaining_tasks)]

    t0 = time.time()
    last_report = t0
    last_save = t0
    chunk_size = workers * 16

    def _save_progress(final=False):
        result = {
            "data_path": data_path,
            "total_rows": len(tasks),
            "processed": data_passed + data_failed,
            "passed": data_passed,
            "failed": data_failed,
            "failure_rate_pct": round(100 * data_failed / max(data_passed + data_failed, 1), 4),
            "elapsed_seconds": round(time.time() - t0, 1),
            "complete": final,
            "failed_indices": sorted(failed_indices),
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_start in range(0, len(remaining_tasks), chunk_size):
            chunk = remaining_tasks[chunk_start : chunk_start + chunk_size]
            futures = [pool.submit(_eval_one, t) for t in chunk]

            for fut in as_completed(futures):
                idx, reward, err = fut.result()

                if err is not None:
                    data_failed += 1
                    failed_indices.append(idx)
                    if len(errors) < 50:
                        errors.append((idx, err))
                elif reward != 0.0:
                    data_failed += 1
                    failed_indices.append(idx)
                    if len(errors) < 50:
                        errors.append((idx, f"expected 0.0, got {reward}"))
                else:
                    data_passed += 1

            done = data_passed + data_failed
            now = time.time()
            if now - last_report >= 10 or done == len(tasks):
                elapsed = now - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(
                    f"  progress: {done}/{len(tasks)} "
                    f"({100*done/len(tasks):.1f}%)  "
                    f"pass={data_passed} fail={data_failed}  "
                    f"rate={rate:.0f}/s  ETA={eta:.0f}s",
                    flush=True,
                )
                last_report = now

            # Incremental save every 60 seconds
            if now - last_save >= 60:
                _save_progress()
                last_save = now

    elapsed = time.time() - t0

    if errors:
        print(f"\n  First 10 failures:")
        for idx, err in errors[:10]:
            print(f"    row {idx}: {err}")

    # Final save
    _save_progress(final=True)
    print(f"\n  Results saved to {result_path}")
    print(f"  Done in {elapsed:.1f}s — {data_passed}/{len(tasks)} passed, {data_failed} failed")
    return data_passed, data_failed


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Test mix_code environment")
    parser.add_argument(
        "--data_path",
        default="/workspace/data/mix_code/train_data.parquet",
        help="Path to parquet file for full-data scan",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for full data scan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Synthetic tests
    test_inputs_synthetic()
    test_assertions_synthetic()
    test_assertions_bare()
    test_assertions_stdin_dict()
    test_edge_cases()
    test_env_interface()
    test_timeouts()

    # Full data scan
    data_passed, data_failed = test_full_data(args.data_path, workers=args.workers)

    # Summary
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Synthetic: {passed}/{total} passed, {failed} failed")
    print(f"Data scan: {data_passed}/{data_passed + data_failed} passed, {data_failed} failed")
    if failed or data_failed:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
