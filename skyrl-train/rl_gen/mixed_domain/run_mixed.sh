set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

NUM_GPUS=4
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
RUN_NAME=rl_gen_mixed_test
BASE_DIR=${BASE_DIR:-$HOME/.cache/rl_gen}
DEBUG=false
MAX_GEN_LENGTH=8192

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
        --num_gpus=*)
            NUM_GPUS="${1#*=}" 
            ;;
        --max_gen_length=*)
            MAX_GEN_LENGTH="${1#*=}" 
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

DATA_DIR="$BASE_DIR/data/mixed"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
# if debug, set logger to console
if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi

#   "data.val_data=["${DATA_DIR}/aime2024_test.parquet", "${DATA_DIR}/aime2025_test.parquet", "${DATA_DIR}/amc2024_test.parquet", "${DATA_DIR}/math500_test.parquet", "${DATA_DIR}/minervamath_test.parquet"]" \

uv run --isolated --extra vllm -m rl_gen.mixed_domain.main_mixed \
  data.train_data="['$DATA_DIR/mixed_domain_train.parquet']" \
  "data.val_data=["${DATA_DIR}/mixed_domain_valid.parquet"]" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=16 \
  trainer.critic_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=$MAX_GEN_LENGTH \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.sampling_params.temperature=0.6 \
  generator.eval_sampling_params.temperature=0.6 \
  environment.skyrl_gym.max_env_workers=28 \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.85 \
  trainer.logger=$LOGGER \
  trainer.project_name="rl-gen-mixed" \
  trainer.max_ckpts_to_keep=1 \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$BASE_DIR/ckpts/rl_gen-mixed-$RUN_NAME" \
  trainer.export_path="$BASE_DIR/exports/rl_gen-mixed-$RUN_NAME" \
  trainer.hf_save_interval=100 \
  $@
