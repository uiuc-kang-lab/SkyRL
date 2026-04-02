set -a

source .env

# uv run --extra fsdp examples/eval/openwebtext_kl/compute_kl_divergence_vllm.py \
#     --base_model Qwen/Qwen3.5-2B \
#     --lora $CHECKPOINT_DIR/qwen3.5-2b_alpha2_lr5e-4_opd-4b/global_step_55/policy/lora_adapter/ \
#     --dataset $DATA_DIR/openwebtext/sample.parquet \
#     --max_gen_length 1024 \
#     --tp_size 1 \
#     --batch_size 32 \
#     --max_prompt_length 1024 \
#     --gpu_memory_utilization 0.8 \
#     --prompt_key text \
#     --output_path ./kl_div_trained_model.json > logs/kl_div_trained_model.log 2>&1

for i in {1..8}; do
    uv run --extra fsdp examples/eval/openwebtext_kl/compute_kl_divergence_vllm.py \
        --base_model Qwen/Qwen3.5-2B \
        --lora $LORA_DIR/Qwen3.5-2B_r1_attn_random$i \
        --dataset $DATA_DIR/openwebtext/sample.parquet \
        --max_gen_length 1024 \
        --tp_size 1 \
        --batch_size 32 \
        --max_prompt_length 1024 \
        --gpu_memory_utilization 0.8 \
        --prompt_key text \
        --output_path ./kl_div_random$i.json > logs/kl_div_random$i.log 2>&1
done