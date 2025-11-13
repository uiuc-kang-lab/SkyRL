set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

NUM_GPUS=8
MODEL_NAME=Qwen/Qwen2.5-Math-7B

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR=${1#*=} 
            ;;
        --n_sample=*)
            N_SAMPLE="${1#*=}" 
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

DATA_DIR="$BASE_DIR/data/deepscaler"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
LOGGER="console"
EXPORT_PATH="$BASE_DIR/exports/deepscaler/wrong-answers_$MODEL_SHORT_NAME"

#   "data.val_data=["${DATA_DIR}/aime2024_test.parquet", "${DATA_DIR}/aime2025_test.parquet", "${DATA_DIR}/amc2023_test.parquet", "${DATA_DIR}/amc2024_test.parquet", "${DATA_DIR}/math500_test.parquet", "${DATA_DIR}/minervamath_test.parquet"]" \

uv run --isolated --extra vllm -m rl_noise.math_hard.main_math_hard \
  data.train_data="['$DATA_DIR/deepscaler_train.parquet']" \
  data.val_data="['$DATA_DIR/deepscaler_train.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_only=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.critic_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=50 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=3072 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.eval_sampling_params.temperature=0.6 \
  generator.eval_sampling_params.top_p=0.95 \
  environment.env_class=math_hard \
  environment.skyrl_gym.max_env_workers=28 \
  generator.eval_n_samples_per_prompt=$N_SAMPLE \
  generator.n_samples_per_prompt=16 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=$LOGGER \
  trainer.export_path=$EXPORT_PATH \
  $@
