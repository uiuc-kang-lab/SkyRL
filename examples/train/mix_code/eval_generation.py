import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from loguru import logger

from examples.train.mix_code.env import MixCodeEnv
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_one_problem(response: str, reward_spec: dict) -> bool:
    env = MixCodeEnv(extras={"reward_spec": reward_spec}, env_config={"timeout": 20})
    step_out = env.step(response)
    return float(step_out["reward"]) > 0.0

def run_eval(args: argparse.Namespace) -> dict:
    
    # load batches
    data = []
    batch_files = glob.glob(os.path.join(args.dump_gen_dir, "*.json"))
    for batch_file in batch_files:
        with open(batch_file, "r") as f:
            batch_data = json.load(f)
            pids = batch_data["problem_ids"]
            responses = batch_data["responses"]
            reward_specs = batch_data["reward_specs"]
            datum = list(zip(pids, responses, reward_specs))
            data.extend(datum)
    # run env over data and collect results in parallel
    # Use threads (not processes) for the outer pool because code_eval's
    # _run_in_subprocess already spawns isolated child processes for each
    # evaluation.  multiprocessing.Pool workers are daemons, and daemons
    # cannot spawn children — threads have no such restriction.
    results_by_problem = defaultdict(list)
    futures_to_pid = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for pid, response, reward_spec in data:
            future = executor.submit(run_one_problem, response, reward_spec)
            futures_to_pid[future] = pid
            results_by_problem[pid].append(future)

        # Wait for all futures to finish and collect results
        for future in tqdm(as_completed(futures_to_pid), total=len(futures_to_pid)):
            future.result()  # raise any exception early

        for pid, future_list in results_by_problem.items():
            results_by_problem[pid] = [f.result() for f in future_list]

    # Metrics
    n_total = len(results_by_problem)
    pass_at_1 = sum(sum(v) for v in results_by_problem.values()) / n_total / args.n_samples
    metrics = {"pass_at_1": pass_at_1, "n_problems": n_total}
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
    p.add_argument("--output_path", default=None,
                   help="Write JSON results to this path")
    p.add_argument("--dump_gen_dir", default=None,
                   help="Directory to dump generated samples (for debugging)")
    p.add_argument("--num_workers", type=int, required=True,
                   help="Number of worker processes to use")
    p.add_argument("--n_samples", type=int, default=64,  
                   help="Number of samples per problem (should match what was generated in dump_gen_dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_eval(args)
    if metrics:
        print(f"\npass@1: {metrics['pass_at_1']:.4f}  (n={metrics['n_problems']} problems)")


if __name__ == "__main__":
    main()
