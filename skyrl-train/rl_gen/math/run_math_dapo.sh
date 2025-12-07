set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

# main DAPO parameters
EPS_CLIP_LOW=0.2
EPS_CLIP_HIGH=0.28
DYNAMIC_SAMPLING_TYPE=filter
DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES=30
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=4096
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
CLIP_RATIO_C=10.0

NUM_GPUS=8
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
RUN_NAME=rl_gen_math_test_dapo
BASE_DIR=/projects/illinois/eng/cs/ddkang/yxx404/.cache/rl_gen/
DEBUG=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR=${1#*=} 
            ;;
        --run_name=*)
            RUN_NAME="${1#*=}" 
            ;;
        --debug)
            DEBUG="true"
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

DATA_DIR="$BASE_DIR/data"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
# if debug, set logger to console
if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi

#   "data.val_data=["${DATA_DIR}/aime2024_test.parquet", "${DATA_DIR}/aime2025_test.parquet", "${DATA_DIR}/amc2024_test.parquet", "${DATA_DIR}/math500_test.parquet", "${DATA_DIR}/minervamath_test.parquet"]" \

uv run --isolated --extra vllm -m rl_gen.math.main_math \
  data.train_data="['$DATA_DIR/deepscaler_train.parquet']" \
  "data.val_data=["${DATA_DIR}/aime2024_test.parquet", "${DATA_DIR}/aime2025_test.parquet", "${DATA_DIR}/deepscaler_valid.parquet"]" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
  +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
  trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=4 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=true \
  trainer.eval_interval=25 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=16 \
  trainer.critic_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=8192 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.kl_loss_coef=0.001 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.sampling_params.temperature=0.6 \
  generator.eval_sampling_params.temperature=0.6 \
  environment.env_class=math \
  environment.skyrl_gym.max_env_workers=12 \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.85 \
  trainer.logger=$LOGGER \
  trainer.project_name="rl-gen-math" \
  trainer.max_ckpts_to_keep=1 \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$BASE_DIR/ckpts/rl_gen-math-$RUN_NAME" \
  trainer.export_path="$BASE_DIR/exports/rl_gen-math_$RUN_NAME" \
  trainer.hf_save_interval=25 \
  $@
