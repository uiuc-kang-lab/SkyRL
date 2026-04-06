set -a

source .env

bash examples/train/eurus_rl_math/run_eurus_rl_math_tinylora_opd_notis.sh \
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
    --lr 5e-4 \
    --run_name "qwen3.5-4b_lr5e-4_tis_tinylora_n16r16_opd_notis" \
    --num_coefficients 16 \
    --svd_rank 16 \
    --student_model "Qwen/Qwen3.5-4B" \
    --teacher_model "/workspace/Qwen3.5-4B-Finetuned" > logs/qwen3.5-4b_lr5e-4_tis_tinylora_n16r16_opd_notis.log 2>&1