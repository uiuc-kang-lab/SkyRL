set -a

source .env

for LORA_ALPHA in 1 4 16
do
    for LR in 5e-5 1e-5 5e-6 1e-6
    do
        bash examples/train/eurus_rl_math/run_eurus_rl_math.sh \
            --data_dir $DATA_DIR \
            --num_gpus 4 \
            --lora_rank 1 \
            --lora_alpha $LORA_ALPHA \
            --epochs 1 \
            --checkpoint_base_path $CHECKPOINT_BASE_PATH \
            --logger "['wandb','console']" \
            --group_size 8 \
            --train_batch_size 64 \
            --micro_batch_size_per_gpu 4 \
            --eval_batch_size 64 \
            --max_prompt_length 5120 \
            --max_generate_length 4096 \
            --tensor_parallel_size 1 \
            --eval_interval 50 \
            --ckpt_interval 5 \
            --lr $LR \
            --run_name "qwen3.5-4b_alpha${LORA_ALPHA}_lr${LR}" \
            --model_path "Qwen/Qwen3.5-4B" > logs/qwen3.5-4b_alpha${LORA_ALPHA}_lr${LR}.log 2>&1
    done
    uv cache clean
done