set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

# default parameters
BASE_DIR=$HOME
CLIP_RATIO_C=10.0
NUM_GPUS=4
INFERENCE_BACKEND="vllm"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --n_sample=*)
            N_SAMPLE="${1#*=}"
            ;;
        --base_dir=*)
            BASE_DIR="${1#*=}" 
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

DATA_DIR="$BASE_DIR/data/gsm8k"
LOGGER="console"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
EXPORT_PATH="$BASE_DIR/exports/gsm8k/wrong-answers_trial-1_$MODEL_SHORT_NAME"

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/train.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_only=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.eval_n_samples_per_prompt=$N_SAMPLE \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.eval_sampling_params.temperature=0.6 \
  generator.eval_sampling_params.top_p=0.95 \
  environment.env_class=gsm8k \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.export_path=$EXPORT_PATH \
  $@