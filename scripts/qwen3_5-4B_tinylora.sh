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
    --ckpt_interval 5 \
    --lr 1e-3 \
    --run_name "qwen3.5-4b_alpha2_lr1e-2_tinylora-n16r16" \
    --model_path "/workspace/Qwen3.5-4B" \
    --num_coefficients 16 \
    --svd_rank 16 \
    --projection_seed 2333 > logs/qwen3.5-4b_alpha2_lr1e-2_tinylora-n16r16.log 2>&1