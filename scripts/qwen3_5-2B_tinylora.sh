set -a

source .env

bash examples/train/eurus_rl_math/run_eurus_rl_math_tinylora.sh \
    --data_dir $DATA_DIR \
    --num_gpus 2 \
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
    --ckpt_interval 50 \
    --lr 5e-4 \
    --run_name "qwen3.5-2b_alpha2_lr5e-4_tinylora" \
    --model_path "Qwen/Qwen3.5-2B" \
    --num_coefficients 4 \
    --svd_rank 16 \
    --projection_seed 2333 > logs/qwen3.5-2b_alpha2_lr5e-4_tinylora.log 2>&1