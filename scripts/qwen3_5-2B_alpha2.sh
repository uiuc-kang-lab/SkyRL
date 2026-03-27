set -a

source .env

bash examples/train/eurus_rl_math/run_eurus_rl_math.sh \
    --data_dir $DATA_DIR \
    --num_gpus 4 \
    --lora_rank 1 \
    --lora_alpha 2 \
    --epochs 10 \
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
    --lr 5e-4 \
    --run_name "qwen3.5-2b_alpha2_lr5e-4_tis_kl" \
    --model_path "Qwen/Qwen3.5-2B" > logs/qwen3.5-2b_alpha2_lr5e-4_tis_kl.log 2>&1