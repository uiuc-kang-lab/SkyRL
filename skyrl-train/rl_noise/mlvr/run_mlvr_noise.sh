set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-1.5B-Instruct on GSM8k dataset.
# Uses 1 node with 4 GPUs.
# uv run examples/llm_as_a_judge/gsm8k_dataset_judge.py --output_dir $HOME/data/gsm8k_llm_judge
# add OPENAI_API_KEY and WANDB_API_KEY to .env.llm_judge
# bash examples/llm_as_a_judge/run_llm_judge.sh

NUM_GPUS=4
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
RUN_NAME=mlvr-test
NOISE_LEVEL=0
NUM_INFERENCE_ENGINES=4
TP_SIZE=1

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --noise_level=*)
            NOISE_LEVEL="${1#*=}" 
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

DATA_DIR="$BASE_DIR/data/mlvr"

if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi

# We use a smaller batch size here for demonstration
uv run --isolated --extra vllm -m rl_noise.mlvr.main_llm_judge \
  data.train_data="['$DATA_DIR/train_with_wrong_answers_0.5.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_tis=true \
  trainer.algorithm.tis_imp_ratio_cap=2.0 \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=1 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_only=false \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  trainer.max_ckpts_to_keep=1 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=5e-7 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="rl-noise-mlvr" \
  trainer.resume_mode=latest \
  environment.env_class=llm_as_a_judge \
  environment.skyrl_gym.max_env_workers=12 \
  environment.skyrl_gym.llm_as_a_judge.model="gpt-4o-mini-2024-07-18" \
  trainer.run_name=$RUN_NAME \
  trainer.ckpt_path="$BASE_DIR/ckpts/$RUN_NAME" \
  trainer.export_path="$BASE_DIR/exports/mlvr_$RUN_NAME" \
  $@