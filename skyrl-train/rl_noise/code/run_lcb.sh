set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SearchR1 data.
# export WANDB_API_KEY=<your_key_here>
# bash examples/livecodebench/run_lcb.sh

# default parameters
BASE_DIR=$HOME
CLIP_RATIO_C=10.0
NUM_GPUS=4
INFERENCE_BACKEND="vllm"
TIS_IMP_RATIO_CAP=2.0
USE_TIS=true
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

DATA_DIR=$BASE_DIR/data/lcb
# if debug, set logger to console
if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')

#   "data.train_data=["${DATA_DIR}/train_livecodebench_part0.json"]" \
#   "data.val_data=["${DATA_DIR}/test_livecodebench_part0.json"]" \

# NOTE (sumanthrh): micro_train_batch_size and micro_forward_batch_size can be tuned
uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  "data.train_data=["${DATA_DIR}/train_livecodebench_part0.json", "${DATA_DIR}/train_livecodebench_part1.json", "${DATA_DIR}/train_livecodebench_part2.json", "${DATA_DIR}/train_livecodebench_part3.json", "${DATA_DIR}/train_livecodebench_part4.json"]" \
  "data.val_data=["${DATA_DIR}/test_livecodebench_part0.json", "${DATA_DIR}/test_livecodebench_part1.json", "${DATA_DIR}/test_livecodebench_part2.json", "${DATA_DIR}/test_livecodebench_part3.json", "${DATA_DIR}/test_livecodebench_part4.json"]" \
  data.noise_level=$NOISE_LEVEL \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.policy_mini_batch_size=2 \
  trainer.train_batch_size=16 \
  trainer.epochs=20 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=29000 \
  generator.max_input_length=29000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.ckpt_interval=50 \
  trainer.max_ckpts_to_keep=1 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=lcb \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.7 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.logger="$LOGGER" \
  trainer.project_name="rl-noise-code" \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=null \
  trainer.ckpt_path="$BASE_DIR/ckpts/code-$MODEL_SHORT_NAME" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
