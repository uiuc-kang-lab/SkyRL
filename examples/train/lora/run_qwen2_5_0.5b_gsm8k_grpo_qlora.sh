set -x

# QLoRA (LoRA + NF4 4-bit quantization) with selective layers for Qwen2.5-0.5B-Instruct on GSM8K.
# Uses FSDP backend. load_in_4bit=True bypasses FSDP sharding; each GPU holds the full
# quantized model with LoRA adapters trained in bf16.
#
# Selective layers: LoRA is applied only to attention projections in layers 12-23
# (upper half of the 24-layer model), reducing trainable parameters vs full LoRA.
#
# NOTE: When layers_to_transform is set, target_modules must be an explicit list of
# module name strings (not the "all-linear" shortcut). For Qwen2.5, attention projection
# names are: q_proj, k_proj, v_proj, o_proj.
#
# Prerequisites:
#   uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>
#   pip install bitsandbytes  # required for 4-bit quantization
#
# Usage:
#   bash examples/train/lora/run_qwen2_5_0.5b_gsm8k_grpo_qlora.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.load_in_4bit=true \
  "trainer.policy.model.lora.target_modules=['q_proj','k_proj','v_proj','o_proj']" \
  "trainer.policy.model.lora.layers_to_transform=[12,13,14,15,16,17,18,19,20,21,22,23]" \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=2.0e-5 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_0.5b_qlora" \
  trainer.run_name="gsm8k_0.5b_qlora_grpo_selective_layers" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_0.5b_qlora_ckpt" \
  $@
