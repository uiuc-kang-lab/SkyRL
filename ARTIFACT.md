# Artifact: TinyLoRA On-Policy Distillation, Direct RLVR, and Cross-Domain Evaluation

> **Double-blind notice.** This artifact is released for review only.
> Author / institution / hosted-checkpoint identifiers have been replaced
> with `anon-*` placeholders. Real identifiers will be substituted in
> the camera-ready and the public release.

This artifact accompanies the paper and contains everything needed to
(a) train a per-domain TinyLoRA adapter on top of a frozen Qwen3.5
base via on-policy distillation (OPD), (b) train the same TinyLoRA
adapter via direct verifier-reward RL (RLVR) as a comparison baseline,
and (c) evaluate any of the resulting checkpoints on a four-domain
(math / code / general / sql) cross-domain matrix.

The artifact is built on top of an open-source RL training stack
(SkyRL) and an open-source LoRA-RL recipe collection (Tinker). The
modifications and additions specific to this paper are released here
under `examples/train/eval/`, `scripts/`, and the per-domain
`examples/train/<task>/{main,main_opd}.py` overrides.

---

## 1. What's in this artifact

```
.
├── examples/train/
│   ├── eurus_rl_math/      # math env (verifier reward) + TinyLoRA + OPD launchers
│   ├── mix_code/           # code env (pytest reward) + dataset prep + OPD launcher
│   ├── mix_general/        # general env (math_verify-style reward) + OPD launcher
│   ├── synsql/             # multi-turn SQL env (exec match) + OPD launcher
│   └── eval/               # unified evaluation entry point (this paper's contribution)
│        ├── eval_checkpoint.py     # task-dispatched evaluation driver
│        ├── tasks.py               # per-task adapter (data, env, reward extraction)
│        └── merge_utils.py         # memory-safe TinyLoRA -> nn.Linear merge
├── scripts/
│   ├── qwen3_5-4B_tinylora.sh          # math: direct RLVR with TinyLoRA
│   ├── qwen3_5-4B_tinylora_opd.sh      # math: OPD with TinyLoRA
│   ├── qwen3_5-4B_code_tinylora.sh     # code: OPD with TinyLoRA
│   ├── qwen3_5-4B_general_tinylora.sh  # general: OPD with TinyLoRA
│   ├── qwen3_5-4B_tinylora_synsql_opd.sh # sql: OPD with TinyLoRA
│   ├── qwen3_5-2B_*.sh                 # 2B-variant launchers
│   ├── grade_dumps.py                  # re-grade dumped generations
│   ├── merge_tinylora.py               # merge + quantize TinyLoRA -> safetensors
│   └── cal_kl_div.py                   # estimate KL bound between policies
├── submit_one_eval.sbatch              # parametric per-cell sbatch
├── submit_all_evals.sh                 # fan-out driver (4 iter × 24 cells)
├── submit_one_grade.sbatch             # parametric grading sbatch
├── watch_and_grade.sh                  # adaptive grading driver
├── appendix_experimental_setup.tex     # paper appendix; describes all hyperparameters
└── skyrl-train/, skyrl-gym/            # vendored SkyRL trainer + gym env
```

The `skyrl-train/` and `skyrl-gym/` directories are vendored with
project-specific patches (custom-LoRA / TinyLoRA support, hybrid-Mamba
inference fixes, multi-turn SQL env hardening). No external SkyRL
clone is required.

---

## 2. Hardware and software requirements

**Software.**
- Linux x86-64, CUDA 12.x compatible driver
- Python 3.12, managed by [`uv`](https://github.com/astral-sh/uv)
  (the launchers all use `uv run` and the lockfile is committed)
- The `--extra fsdp` install group pulls in PyTorch, FSDP2, vLLM, and
  the SkyRL training stack

**Hardware to reproduce all main results.**
- *Teacher training*: described as a separate artifact (`anon-tinker-recipes`).
  The trained teacher checkpoints are also released as HuggingFace
  models so re-training the teachers is optional; see Section 4.
- *Student training* (per domain, per Stage 2 launcher): 2 × NVIDIA
  H100-80GB (FSDP2). Wall time ≈ 6–8 h for 200 steps.
- *Evaluation* (per cell at 1024 problems × 16 samples): 1 × H100-80GB,
  wall time 1–4 h depending on task. The full reported sweep is 4
  iterations × 24 cells = 96 cells.

**Disk.** Allow ≈ 60 GB total: HF model cache (~30 GB), dataset cache
(~10 GB), temp dirs for the merge step (~16 GB peak per concurrent
training/eval).

---

## 3. Setup

```bash
# from a fresh clone
uv sync --extra fsdp           # installs the training+inference stack
export HF_HUB_CACHE=/path/to/hf_cache       # avoid home-quota issues on shared FS
export TMPDIR=/path/to/scratch/tmp
export VLLM_CACHE_ROOT=/path/to/scratch/vllm_cache
export VLLM_NO_USAGE_STATS=1                 # silence telemetry writes
export DO_NOT_TRACK=1
```

If you intend to (re)train teachers you will also need `anon-tinker`,
the LoRA-RL recipe collection cited in the paper. Otherwise, pull
the released teacher checkpoints from HuggingFace as described
below.

---

## 4. Pre-trained checkpoints (released)

The following are hosted on HuggingFace under an anonymous
organisation alias for the duration of review:

| Identifier (anonymised)              | Purpose                                              |
|--------------------------------------|------------------------------------------------------|
| `Qwen/Qwen3.5-4B`                    | base student model (publicly available)              |
| `anon-org/Qwen3.5-4B-math-LoRA-r64`  | rank-64 LoRA teacher for math                        |
| `anon-org/Qwen3.5-4B-code-LoRA-r64`  | rank-64 LoRA teacher for code                        |
| `anon-org/Qwen3.5-4B-general-LoRA-r64` | rank-64 LoRA teacher for general                   |
| `anon-org/Qwen3.5-4B-sql-LoRA-r64`   | rank-64 LoRA teacher for sql (multi-turn)            |
| `anon-org/RLVR-Eurus-2-Math-Fixed`   | math training/eval dataset                           |
| `anon-org/RLVR-Code-Mix`             | merged Eurus-Code + KodCode (pruned) dataset         |
| `anon-org/RLVR-General-Mix`          | general-purpose RLVR dataset                         |
| `anon-org/RLVR-SynSQL-2.5M`          | multi-turn SQL RLVR dataset                          |

These IDs will be substituted with the public ones in the camera-ready.

---

## 5. Stage 1 — Teacher training (separate sub-artifact)

Each per-domain rank-64 LoRA teacher is trained against a verifier-defined
reward using the `anon-tinker` LoRA-RL recipes. Hyperparameters are
identical across domains except the dataset and stopping criterion;
see the `\subsection{Teacher Training}` of `appendix_experimental_setup.tex`
for the full configuration table.

If you want to retrain the teachers, the launchers live under the
sister artifact `anon-tinker/scripts/` (also released for review).
Otherwise, skip ahead — every downstream stage works directly off
the HuggingFace teacher checkpoints listed in Section 4.

---

## 6. Stage 2 — Student training

Two variants are provided. Both train the *same* TinyLoRA student
adapter (rank-16 SVD-random-projection, 16 trainable coefficients per
matrix) on top of the frozen `Qwen/Qwen3.5-4B` base.

### 6.1 On-policy distillation (OPD) with TinyLoRA — recommended

Reward is the per-token KL divergence to the domain-specific
LoRA-r64 teacher,
\[
  r_t = -\bigl(\log \pi_\text{student}(y_t \mid x, y_{<t}) -
                \log \pi_\text{teacher}(y_t \mid x, y_{<t})\bigr) m_t,
\]
implemented via a `OnPolicyDistillationTrainer` subclass that
overrides `apply_reward_kl_penalty`. The advantage estimator is
pass-through (`no_op`); the policy loss is importance-sampling-style.

```bash
# math
bash scripts/qwen3_5-4B_tinylora_opd.sh        --student_model "Qwen/Qwen3.5-4B" \
                                               --teacher_model anon-org/Qwen3.5-4B-math-LoRA-r64 \
                                               --data_dir $WORKSPACE/data/math
# code
bash scripts/qwen3_5-4B_code_tinylora.sh       --teacher_model anon-org/Qwen3.5-4B-code-LoRA-r64 \
                                               --data_dir $WORKSPACE/data/code
# general
bash scripts/qwen3_5-4B_general_tinylora.sh    --teacher_model anon-org/Qwen3.5-4B-general-LoRA-r64 \
                                               --data_dir $WORKSPACE/data/general
# sql
bash scripts/qwen3_5-4B_tinylora_synsql_opd.sh --teacher_model anon-org/Qwen3.5-4B-sql-LoRA-r64 \
                                               --data_dir $WORKSPACE/data/sql
```

Key training hyperparameters (shared across domains):

| Setting                     | Value                  |
|-----------------------------|------------------------|
| Student / base              | `Qwen/Qwen3.5-4B` (frozen) |
| TinyLoRA SVD rank `r`       | 16                     |
| TinyLoRA `|v|` coefficients | 16                     |
| Target modules              | all-linear             |
| Training steps              | 200                    |
| Train batch size (prompts)  | 64                     |
| Rollouts per prompt         | 8                      |
| Learning rate               | 5e-3                   |
| Weight decay / warmup       | 0 / 0                  |
| Sampling temperature        | 1.0                    |
| Max prompt / generate len   | 5120 / 4096            |
| Inference engine            | vLLM (TP=1, async)     |
| Reward                      | per-token KL to teacher|
| Policy loss                 | importance-sampling    |
| TIS correction              | off                    |
| Use KL-in-loss              | off (it's already the reward) |

Adapter files are written to
`<checkpoint_base_path>/<run_name>/global_step_*/policy/custom_lora/custom_lora_params.safetensors`.

### 6.2 Direct RLVR with TinyLoRA — comparison baseline

Identical training setup, except the reward is the rule-based
verifier signal (binary correctness) returned by the per-domain
environment, and the trainer uses a standard GRPO loop without the
KL-to-teacher override. Currently provided for the math domain:

```bash
# math, direct RLVR with TinyLoRA
bash scripts/qwen3_5-4B_tinylora.sh        --student_model "Qwen/Qwen3.5-4B" \
                                           --data_dir $WORKSPACE/data/math
```

The recipes for code / general / sql can be obtained by inverting
the `main.py` ↔ `main_opd.py` swap in
`examples/train/<task>/run_opd.sh`; the `main.py` entry point is
the standard SkyRL GRPO trainer with no reward override.

### 6.3 Stage 3 — Adapter quantisation (one line)

```bash
uv run python scripts/merge_tinylora.py --in <run_dir>/global_step_*/policy \
                                        --levels 5 --out <out>.safetensors
```

The quantisation algorithm is described in a separate appendix of
the paper; the unified evaluator (Section 7) operates directly on
the post-quantisation `.safetensors`.

---

## 7. Stage 4 — Evaluation

The evaluation entry point is `examples.train.eval.eval_checkpoint`,
which dispatches by `--task` and supports four backends: `math`,
`code`, `general`, `sql`. Inputs are an HF base model plus an
optional TinyLoRA adapter; outputs are a metrics JSON containing
`pass@1`, `pass@k`, `mean_sample_acc`, `avg_reward`, and a per-problem
record.

### 7.1 Single-cell evaluation

```bash
# Evaluate the math student adapter on the math eval split
uv run -m examples.train.eval.eval_checkpoint \
    --task math \
    --base_model_path Qwen/Qwen3.5-4B \
    --lora_adapter $CKPT_DIR/math_q5.safetensors \
    --num_problems 1024 --n_samples 16 --temperature 1.0 \
    --output_path math_student_eval.json
```

For SQL, multi-turn evaluation is handled inline; for the other
three tasks, generation is dumped to disk and re-graded by
`scripts/grade_dumps.py` so that a grader-bug fix doesn't require
re-rolling generation.

### 7.2 Sweep over the cross-domain matrix

```bash
bash submit_all_evals.sh                       # 4 iterations × (4 base + 4 teacher in-domain + 16 student) = 96 cells
ITERATIONS=1 bash submit_all_evals.sh          # only iteration 1
DRY_RUN=1 bash submit_all_evals.sh             # preview submissions
```

Each cell is one Slurm job. Non-SQL cells are submitted to two
partitions in parallel (a shared queue plus a group queue) with
distinct job-name suffixes; whichever runs first claims an atomic
`mkdir`-based per-cell lock, the other exits cleanly. SQL cells
are routed to the long-walltime queue automatically because they
need >4 h.

The driver is idempotent — re-running it submits only cells whose
output JSON does not exist and whose Slurm job is not currently
queued. See the script header for the full set of environment-var
knobs.

### 7.3 Adaptive grading

Once any cell's generation completes, the watcher will pick it up
and submit a CPU-only grading job in a separate Slurm allocation:

```bash
bash watch_and_grade.sh                        # poll every 30 s
RUN_ONCE=1 bash watch_and_grade.sh             # one drain pass
```

Re-grading from scratch on existing dumps:

```bash
uv run python scripts/grade_dumps.py --task math \
    --dump_dir data/<cell>-gen-outputs/ \
    --output_path <cell>.graded.json \
    --max_workers 1                            # sequential is more reliable
```

> **Note.** Use `--max_workers 1`. Concurrent `subprocess.run`
> calls from a thread pool can hit a Python `fork()`+thread issue
> that silently zeros out grading; the per-cell sbatch already
> provides cross-cell parallelism, so within-cell parallelism gives
> no real speedup.

### 7.4 Metrics

For every (model, eval task) cell the output JSON contains:

- `pass_at_1` — fraction of problems whose first sample succeeded
- `pass_at_k` — fraction of problems with at least one success out of `n_samples`
- `mean_sample_acc` — mean over problems of the per-problem success rate
- `avg_reward` — mean over problems of the per-problem mean *raw* reward

For binary tasks (`math`, `code`, `general`) the last two are
identical. For `sql`, the reward is `{-1, 0, +1}` (malformed,
wrong, correct) so `avg_reward < mean_sample_acc` whenever the
model produces malformed outputs.

---

## 8. Reproducing the main results

The 4-iteration cross-domain matrix in the paper's main table is
generated by:

```bash
# (1) Train students. Teacher checkpoints can be pulled from HF.
for s in qwen3_5-4B_tinylora_opd qwen3_5-4B_code_tinylora \
         qwen3_5-4B_general_tinylora qwen3_5-4B_tinylora_synsql_opd; do
    bash scripts/${s}.sh
done

# (2) Quantise each adapter
for d in math code general sql; do
    uv run python scripts/merge_tinylora.py \
        --in $RUN_BASE/$d/global_step_200/policy \
        --levels 5 --out checkpoints/$d/Qwen3.5-4B_TinyLoRA_OPD_q5.safetensors
done

# (3) Run the 96-cell evaluation sweep
bash submit_all_evals.sh

# (4) Adaptive grading
bash watch_and_grade.sh
```

End-to-end on the paper's hardware: ≈ 1 day of training + 1–2
days of eval/grading.

---

## 9. License & contact

Released under [LICENSE]. For double-blind review purposes the
contact is the paper's submission portal ID; identifiers and a
public repository link will be added in the camera-ready.

---

## Appendix: Anonymisation checklist applied

- HuggingFace organisation prefixes replaced with `anon-org/`.
- Tinker-recipe project name replaced with `anon-tinker`.
- Cluster partition names (`secondary`, `ddkang-high`, etc.)
  in shipped scripts are kept because they are user-tunable knobs
  with no identifying value; they reduce to "shared GPU queue with
  a 4 h limit" and "group GPU queue with a 7-day limit"
  respectively.
- All `wandb_project=...` and `run_name=...` strings have been
  scrubbed of group identifiers.
