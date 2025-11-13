set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

NUM_GPUS=1
MODEL_NAME=Qwen/Qwen3-0.6B
RUN_NAME=arc_test
DEBUG=false
batch_size=64
group_size=16
noise_rate=0.0

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
        --debug=*)
            DEBUG="true"
            ;;
        --group_size=*)
            group_size="${1#*=}"
            ;;
        --batch_size=*)
            batch_size="${1#*=}"
            ;;
        --noise_rate=*)
            noise_rate="${1#*=}"
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

DATA_DIR="$BASE_DIR/data/arc"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
# if debug, set logger to console
if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi

if [ "$noise_rate" != "0.0" ]; then
  train_file="train_noise_${noise_rate}.parquet"
else
  train_file="train.parquet"
fi

# if batch_size is less than 4, set micro batch size to batch_size
if [ "$batch_size" -lt 4 ]; then
  micro_batch_size=$batch_size
else
  micro_batch_size=4
fi

uv run --isolated --extra vllm -m rl_noise.arc.main \
  data.train_data="['$DATA_DIR/$train_file']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$batch_size \
  trainer.policy_mini_batch_size=$batch_size \
  trainer.critic_mini_batch_size=$batch_size \
  trainer.micro_forward_batch_size_per_gpu=$micro_batch_size \
  trainer.micro_train_batch_size_per_gpu=$micro_batch_size \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=3072 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=arc \
  environment.skyrl_gym.max_env_workers=12 \
  generator.n_samples_per_prompt=$group_size \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=$LOGGER \
  trainer.project_name="skyrl-arc" \
  trainer.max_ckpts_to_keep=1 \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$BASE_DIR/ckpts/arc_$RUN_NAME" \
  trainer.export_path="$BASE_DIR/exports/arc_$RUN_NAME" \
  $@
