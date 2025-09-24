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
TIS_IMP_RATIO_CAP=2.0
USE_TIS=false
NOISE_LEVEL=0.0

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

DATA_DIR="$BASE_DIR/data/gsm8k"
# if debug, set logger to console
if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi

# split model name by '/' and get the last part
if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
fi
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  data.noise_level=$NOISE_LEVEL \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
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
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=50 \
  trainer.max_ckpts_to_keep=1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  environment.skyrl_gym.gsm8k.random_perturb=true \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="rl-noise-gsm8k" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$BASE_DIR/ckpts/gsm8k-$MODEL_SHORT_NAME" \
  $@