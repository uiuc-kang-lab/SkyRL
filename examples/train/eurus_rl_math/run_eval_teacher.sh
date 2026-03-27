set -x

# Evaluate a candidate teacher model on the eurus_rl_math validation set.
#
# Runs greedy decoding (temperature=0) for a clean pass@1 estimate.
# Use --n_samples > 1 with --temperature 0.6 for pass@k estimation.
#
# Usage:
#   bash run_eval_teacher.sh \
#       --model_path Qwen/Qwen2.5-7B-Instruct \
#       --data_dir $HOME/data/eurus_rl_math

# Defaults
MODEL_PATH=""
DATA_DIR=""
NUM_GPUS=4
MAX_PROMPT_LENGTH=5120
MAX_GENERATE_LENGTH=4096
TEMPERATURE=0.0
N_SAMPLES=1
BATCH_SIZE=64
GPU_MEMORY_UTILIZATION=0.9
OUTPUT_PATH=""
LIMIT=""

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)              MODEL_PATH="$2"; shift 2 ;;
    --data_dir)                DATA_DIR="$2"; shift 2 ;;
    --num_gpus)                NUM_GPUS="$2"; shift 2 ;;
    --max_prompt_length)       MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_generate_length)     MAX_GENERATE_LENGTH="$2"; shift 2 ;;
    --temperature)             TEMPERATURE="$2"; shift 2 ;;
    --n_samples)               N_SAMPLES="$2"; shift 2 ;;
    --batch_size)              BATCH_SIZE="$2"; shift 2 ;;
    --gpu_memory_utilization)  GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --output_path)             OUTPUT_PATH="$2"; shift 2 ;;
    --limit)                   LIMIT="$2"; shift 2 ;;
    *)                         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: --model_path is required" >&2
  exit 1
fi

if [[ -z "$DATA_DIR" ]]; then
  echo "ERROR: --data_dir is required" >&2
  exit 1
fi

# Default output path: results named after the model and timestamp
if [[ -z "$OUTPUT_PATH" ]]; then
  MODEL_SLUG=$(basename "$MODEL_PATH")
  OUTPUT_PATH="eval_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S).json"
fi

OPTIONAL_ARGS=()
[[ -n "$LIMIT" ]] && OPTIONAL_ARGS+=(--limit "$LIMIT")

uv run --extra fsdp -m examples.train.eurus_rl_math.eval_teacher \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_DIR/train.parquet" \
  --num_gpus "$NUM_GPUS" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_generate_length "$MAX_GENERATE_LENGTH" \
  --temperature "$TEMPERATURE" \
  --n_samples "$N_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --output_path "$OUTPUT_PATH" \
  "${OPTIONAL_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
