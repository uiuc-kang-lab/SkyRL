"""
Code evaluation utilities for the mix_code environment.

Supports two testing methods:
- "inputs": stdin/stdout testing with input/output pairs (Eurus format)
- "assertions": assertion-based testing with test code strings (KodCode format)

All code execution happens in isolated subprocesses with per-test timeouts,
memory limits, and disabled destructive operations (reliability_guard).
This design is safe for multi-threaded callers because signal.alarm is only
used inside the child process (where it runs on the main thread), while the
parent uses p.join(timeout=...) which is thread-safe.
"""

import ast
import json
import sys
import signal
import faulthandler
import platform
import re
import time
import multiprocessing
from io import StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from decimal import Decimal


# ---------------------------------------------------------------------------
# Common imports prepended to user code before execution.
# Mirrors the set used by LiveCodeBench so competitive-programming solutions
# that rely on typical imports work out of the box.
# ---------------------------------------------------------------------------
BASE_IMPORTS = (
    "from itertools import accumulate, chain, combinations, count, permutations, "
    "product, groupby, islice, repeat\n"
    "from copy import deepcopy\n"
    "from string import ascii_lowercase, ascii_uppercase\n"
    "from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt, "
    "factorial, atan2, pi, log, prod\n"
    "from collections import defaultdict, deque, Counter, OrderedDict\n"
    "from bisect import bisect, bisect_left, bisect_right, insort\n"
    "from heapq import heappush, heappop, heapify, merge, nlargest, nsmallest, heapreplace\n"
    "from functools import reduce, cache, lru_cache, cmp_to_key, partial\n"
    "from random import randrange, shuffle\n"
    "from operator import itemgetter, sub, xor, or_, iand\n"
    "from re import search as re_search\n"
    "from os.path import commonprefix\n"
    "from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, "
    "Iterable, Iterator, Generator, Deque\n"
    "from itertools import zip_longest, cycle, pairwise\n"
    "import copy, string, math, collections, bisect, heapq, functools\n"
    "import random, itertools, operator, re, datetime, sys, io, os\n"
    "from time import time\n"
    "sys.setrecursionlimit(50000)\n"
)


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------
class Capturing(list):
    """Context manager that captures everything written to sys.stdout."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1  # prevent accidental close
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


# ---------------------------------------------------------------------------
# Security: disable destructive OS operations in the subprocess
# ---------------------------------------------------------------------------
def reliability_guard(maximum_memory_bytes=None):
    """Disable destructive functions so generated code cannot damage the host.

    WARNING: This is NOT a full sandbox.  It raises the bar significantly but
    should not be considered equivalent to a container or VM boundary.
    """
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "killpg", "rename", "renames",
        "truncate", "replace", "unlink", "fchmod", "fchown", "chmod",
        "chown", "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, attr):
            setattr(os, attr, None)

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    if isinstance(__builtins__, dict):
        __builtins__["help"] = None
    elif hasattr(__builtins__, "help"):
        setattr(__builtins__, "help", None)

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


# ---------------------------------------------------------------------------
# Code extraction from model responses
# ---------------------------------------------------------------------------
def extract_code_from_model(model_response: str):
    """Return the last markdown code block, or None if none found."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


# ---------------------------------------------------------------------------
# Code pre-processing helpers (for stdin/stdout testing)
# ---------------------------------------------------------------------------
def clean_if_name(code: str) -> str:
    """Remove ``if __name__ == '__main__':`` guard, keeping the body."""
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1])
                    + "\n"
                    + ast.unparse(last_block.body)
                )
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    """Wrap all non-import statements in a ``wrapped_function`` callable.

    This is necessary for stdin/stdout testing: we need to call the code as a
    function so that stdin/stdout patching via ``call_method`` works correctly.
    """
    try:
        import_stmts = []
        other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)

        if not other_stmts:
            return code

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[],
                kw_defaults=[], defaults=[],
            ),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return (
            BASE_IMPORTS + "\n"
            + ast.unparse(import_stmts) + "\n"
            + ast.unparse(function_ast)
        )
    except Exception:
        return code


def compile_code(code: str, timeout: int):
    """Exec *code* in an isolated module namespace and return the module."""
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def get_function(compiled_sol, fn_name: str):
    """Retrieve *fn_name* from a compiled module, or None."""
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def call_method(method, inputs):
    """Call *method* with stdin mocked from *inputs*."""
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner(m):
        try:
            return m()
        except SystemExit:
            pass

    return _inner(method)


# ---------------------------------------------------------------------------
# Output comparison helpers
# ---------------------------------------------------------------------------
def get_stripped_lines(val: str):
    val = val.strip()
    return [line.strip() for line in val.split("\n")]


def convert_line_to_decimals(line: str):
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def _compare_outputs(prediction: str, expected: str) -> bool:
    """Compare captured stdout against expected output, tolerating whitespace
    differences and using Decimal comparison for numeric values."""
    pred_lines = get_stripped_lines(prediction)
    exp_lines = get_stripped_lines(str(expected))

    if len(pred_lines) != len(exp_lines):
        return False

    for pred_line, exp_line in zip(pred_lines, exp_lines):
        if pred_line == exp_line:
            continue
        ok_p, dec_p = convert_line_to_decimals(pred_line)
        ok_e, dec_e = convert_line_to_decimals(exp_line)
        if ok_p and ok_e and dec_p == dec_e:
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Grading: stdin/stdout ("inputs" method)
# ---------------------------------------------------------------------------
def _grade_inputs(code: str, inputs: list, outputs: list, timeout: int) -> bool:
    """Run *code* once per input, capture stdout, compare with expected output.

    Must be called inside a subprocess (uses signal.alarm).
    """
    code = clean_if_name(code)
    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return False

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return False

    for inp, expected_out in zip(inputs, outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured:
            try:
                call_method(method, inp)
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
                return False
            finally:
                signal.alarm(0)
                faulthandler.disable()

        if not _compare_outputs(captured[0], expected_out):
            return False

    return True


# ---------------------------------------------------------------------------
# Grading: assertion-based ("assertions" method)
# ---------------------------------------------------------------------------
def _grade_assertions(code: str, test_code: str, timeout: int) -> bool:
    """Run assertion-based tests against model code.

    Handles three sub-formats found in KodCode data:
    1. ``from solution import func`` — test imports from a "solution" module
    2. Bare assertions that call functions defined in the model code
    3. ``{'stdin': [...], 'stdout': [...]}`` dicts — actually stdio tests;
       detected and routed to ``_grade_inputs`` automatically.

    Must be called inside a subprocess (uses signal.alarm).
    """
    # Start the alarm BEFORE any parsing so malformed input can't hang us.
    signal.alarm(timeout)
    try:
        # Some KodCode entries store stdin/stdout dicts as the ground-truth
        # string instead of assertion code.  Detect and re-route.
        try:
            test_data = ast.literal_eval(test_code)
            if isinstance(test_data, dict) and "stdin" in test_data and "stdout" in test_data:
                signal.alarm(0)  # _grade_inputs manages its own alarms
                return _grade_inputs(code, test_data["stdin"], test_data["stdout"], timeout)
        except (ValueError, SyntaxError):
            pass

        # Standard assertion testing:
        # 1. Compile model code into a module named "solution" so that
        #    ``from solution import func`` works in the test code.
        # 2. Also seed the test namespace with the module's symbols so bare
        #    ``func()`` calls work without an import.
        #
        # NOTE: We do NOT prepend BASE_IMPORTS into the solution module.
        # BASE_IMPORTS contains names like ``factorial`` and ``gcd`` (from math)
        # that would shadow the user-defined functions the tests are trying to
        # verify.  The model code must be self-contained.
        solution_module = ModuleType("solution", "")
        exec(code, solution_module.__dict__)
        sys.modules["solution"] = solution_module

        test_globals = dict(solution_module.__dict__)
        exec(test_code, test_globals)

        # Many KodCode tests are pytest-style: they define ``def test_*()``
        # functions but never call them.  Discover and invoke them so the
        # assertions inside actually run.
        test_fns = [
            v for k, v in test_globals.items()
            if k.startswith("test_") and callable(v)
        ]
        for fn in test_fns:
            fn()

        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)
        sys.modules.pop("solution", None)


# ---------------------------------------------------------------------------
# Subprocess workers
# ---------------------------------------------------------------------------
def _worker_inputs(code, inputs, outputs, timeout, result_queue):
    """Subprocess entry-point for input/output testing."""
    signal.signal(signal.SIGALRM, timeout_handler)
    # Note: no memory-bytes cap — the forked child inherits the parent's
    # virtual-memory footprint, so a low RLIMIT_AS would OOM immediately.
    # This matches LCB's run_test() which also calls reliability_guard()
    # without a memory limit.
    reliability_guard()
    try:
        result_queue.put(_grade_inputs(code, inputs, outputs, timeout))
    except Exception:
        result_queue.put(False)


def _worker_assertions(code, test_code, timeout, result_queue):
    """Subprocess entry-point for assertion testing."""
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()
    try:
        result_queue.put(_grade_assertions(code, test_code, timeout))
    except Exception:
        result_queue.put(False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Hard ceiling so a single evaluation can never block a worker thread for
# an unreasonable amount of time, regardless of the number of test cases.
_MAX_GLOBAL_TIMEOUT = 300  # 5 minutes

# Use "forkserver" start method to avoid fork-from-threads bugs.
# The default "fork" method copies the parent's lock state, which can cause
# deadlocks when forking from a ThreadPoolExecutor worker thread (the same
# executor pattern used by SkyRL's generator).  "forkserver" starts a clean
# server process at import time that handles all subsequent forks.
_mp_ctx = multiprocessing.get_context("forkserver")


def _run_in_subprocess(target, args, global_timeout: float) -> bool:
    """Spawn *target* in an isolated process and return its bool result.

    Enforces *global_timeout* as a hard wall-clock deadline — the subprocess
    is killed (SIGKILL) if it exceeds it.  Always cleans up the Process
    object to prevent zombie / resource leaks.
    """
    result_queue = _mp_ctx.Queue()
    p = _mp_ctx.Process(target=target, args=(*args, result_queue))
    p.start()
    p.join(timeout=global_timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=5)

    try:
        result = result_queue.get_nowait()
    except Exception:
        result = False
    finally:
        p.close()

    return result


def check_correctness_inputs(code: str, tests_dict: dict, timeout: int = 6) -> bool:
    """Check code against input/output pairs in an isolated subprocess."""
    inputs = tests_dict["inputs"]
    outputs = tests_dict["outputs"]
    global_timeout = min((timeout + 1) * len(inputs) + 5, _MAX_GLOBAL_TIMEOUT)
    return _run_in_subprocess(_worker_inputs, (code, inputs, outputs, timeout), global_timeout)


def check_correctness_assertions(code: str, test_code: str, timeout: int = 6) -> bool:
    """Check code against assertion tests in an isolated subprocess.

    NOTE: If the test code is a stdin/stdout dict, ``compute_score`` detects
    this and routes to ``check_correctness_inputs`` instead, which uses a
    per-test-case global timeout.  This function uses a flat ``timeout * 3``
    ceiling to cover compilation + all test-function calls.
    """
    global_timeout = min(timeout * 3 + 5, _MAX_GLOBAL_TIMEOUT)
    return _run_in_subprocess(_worker_assertions, (code, test_code, timeout), global_timeout)


def compute_score(model_response: str, ground_truth: str, method: str, timeout: int = 6):
    """Compute reward for model-generated code.

    Args:
        model_response: Raw model output (may contain markdown code blocks).
        ground_truth: Test data — JSON string with inputs/outputs for "inputs"
                      method, or a Python code string for "assertions" method.
        method: Either "inputs" or "assertions".
        timeout: Per-test-case timeout in seconds.

    Returns:
        (extracted_code_or_None, reward) where reward is 1.0 or 0.0.
    """
    code = extract_code_from_model(model_response)
    if code is None:
        return None, 0.0

    if method == "inputs":
        tests_dict = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        is_correct = check_correctness_inputs(code, tests_dict, timeout=timeout)
    elif method == "assertions":
        # Route stdin/stdout dicts to the inputs path at the parent level so
        # the correct per-test-case global timeout is used.  The subprocess
        # fallback inside _grade_assertions is kept as a safety net.
        try:
            test_data = ast.literal_eval(ground_truth)
            if isinstance(test_data, dict) and "stdin" in test_data and "stdout" in test_data:
                is_correct = check_correctness_inputs(
                    code, {"inputs": test_data["stdin"], "outputs": test_data["stdout"]},
                    timeout=timeout,
                )
            else:
                is_correct = check_correctness_assertions(code, ground_truth, timeout=timeout)
        except (ValueError, SyntaxError):
            is_correct = check_correctness_assertions(code, ground_truth, timeout=timeout)
    else:
        raise ValueError(f"Unknown evaluation method: {method}")

    return code, 1.0 if is_correct else 0.0
