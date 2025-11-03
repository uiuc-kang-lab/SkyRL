set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL-653 data.
# Uses 1 node with 8 GPUs.
# huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_skyrl_sql.sh

# change these paths to your own
BASE_DIR=$HOME

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --noise_level=*)
            NOISE_LEVEL="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR="${1#*=}" 
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

DATA_DIR="$BASE_DIR/data/bird"
DB_PATH="$DATA_DIR/databases"
CKPT_PATH="$BASE_DIR/ckpts/$RUN_NAME"
EXPORT_PATH="$BASE_DIR/exports/$RUN_NAME"

echo "Using data from $DATA_DIR"

NUM_GPUS=8
NUM_INFERENCE_ENGINES=2
TP_SIZE=4
MAX_INPUT_LENGTH=29000
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=64

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="gae" \
  data.train_data="['$DATA_DIR/noisy_train.parquet']" \
  data.val_data="['$DATA_DIR/combined_test.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.critic.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.epochs=20 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=$MAX_INPUT_LENGTH \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=64 \
  trainer.critic_mini_batch_size=64 \
  trainer.algorithm.use_kl_loss=true \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=100 \
  trainer.dump_data_batch=true \
  trainer.export_path=$EXPORT_PATH \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=6 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.stop='["</sql>", "</solution>"]' \
  generator.eval_sampling_params.stop='["</sql>", "</solution>"]' \
  environment.skyrl_gym.text2sql.db_path=$DB_PATH \
  environment.skyrl_gym.max_env_workers=24 \
  trainer.logger="wandb" \
  trainer.project_name="noisy-rl-sql" \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.max_ckpts_to_keep=1 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  $@