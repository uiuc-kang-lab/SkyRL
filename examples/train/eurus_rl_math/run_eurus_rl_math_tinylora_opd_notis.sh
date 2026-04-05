set -x

# Colocated GRPO LoRA training + generation for Qwen2.5-0.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/lora/run_qwen2_5_0.5b_gsm8k_grpo_lora.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

# Defaults
DATA_DIR=""
NUM_GPUS=4
epochs=1
checkpoint_base_path=""
logger="console"
group_size=8
train_batch_size=64
micro_batch_size_per_gpu=4
eval_batch_size=64
max_prompt_length=5120
max_generate_length=1024
tensor_parallel_size=1
eval_interval=5
ckpt_interval=10
lr=3e-5
run_name="eurus_rl_math_qwen3.5-4b_tinylora_opd"
INFERENCE_BACKEND="vllm"
model_path="Qwen/Qwen3.5-4B"

# TinyLoRA (custom LoRA) parameters
svd_rank=32
num_coefficients=128
projection_seed=42
scheme="svd_random_projection"

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
USE_TIS=false

# Parse --key value style arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --checkpoint_base_path) checkpoint_base_path="$2"; shift 2 ;;
    --logger) logger="$2"; shift 2 ;;
    --group_size) group_size="$2"; shift 2 ;;
    --train_batch_size) train_batch_size="$2"; shift 2 ;;
    --micro_batch_size_per_gpu) micro_batch_size_per_gpu="$2"; shift 2 ;;
    --eval_batch_size) eval_batch_size="$2"; shift 2 ;;
    --max_prompt_length) max_prompt_length="$2"; shift 2 ;;
    --max_generate_length) max_generate_length="$2"; shift 2 ;;
    --tensor_parallel_size) tensor_parallel_size="$2"; shift 2 ;;
    --eval_interval) eval_interval="$2"; shift 2 ;;
    --ckpt_interval) ckpt_interval="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --run_name) run_name="$2"; shift 2 ;;
    --student_model) STUDENT_MODEL="$2"; shift 2 ;;
    --teacher_model) TEACHER_MODEL="$2"; shift 2 ;;
    --svd_rank) svd_rank="$2"; shift 2 ;;
    --num_coefficients) num_coefficients="$2"; shift 2 ;;
    --projection_seed) projection_seed="$2"; shift 2 ;;
    --scheme) scheme="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done


uv run --extra fsdp -m examples.train.eurus_rl_math.main_opd \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="no_op" \
  trainer.algorithm.policy_loss_type="importance_sampling" \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.policy.model.custom_lora.enabled=true \
  trainer.policy.model.custom_lora.svd_rank=$svd_rank \
  trainer.policy.model.custom_lora.num_coefficients=$num_coefficients \
  trainer.policy.model.custom_lora.projection_seed=$projection_seed \
  trainer.policy.model.custom_lora.scheme=$scheme \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  generator.inference_engine.tensor_parallel_size=$tensor_parallel_size \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  trainer.epochs=$epochs \
  trainer.eval_batch_size=$eval_batch_size \
  trainer.eval_before_train=false \
  trainer.eval_interval=$eval_interval \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$train_batch_size \
  trainer.policy_mini_batch_size=$train_batch_size \
  trainer.micro_forward_batch_size_per_gpu=$micro_batch_size_per_gpu \
  trainer.micro_train_batch_size_per_gpu=$micro_batch_size_per_gpu \
  trainer.ckpt_interval=$ckpt_interval \
  trainer.max_prompt_length=$max_prompt_length \
  generator.sampling_params.max_generate_length=$max_generate_length \
  trainer.policy.optimizer_config.lr=$lr \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=true \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.enforce_eager=true \
  generator.batched=true \
  environment.env_class=eurus_rl_math \
  environment.skyrl_gym.max_env_workers=-1 \
  generator.n_samples_per_prompt=$group_size \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$logger" \
  trainer.project_name="rl_bounds" \
  trainer.run_name="$run_name" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$checkpoint_base_path/$run_name" \
  "${EXTRA_ARGS[@]}" \
  trainer.export_path="/root/export/$run_name" 

