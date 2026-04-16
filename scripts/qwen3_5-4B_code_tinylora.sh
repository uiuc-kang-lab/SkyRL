set -a

source .env

bash examples/train/mix_code/run_opd.sh \
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
    --lr 5e-3 \
    --run_name "qwen3.5-4b_lr5e-3_tis_tinylora_n16r16_opd_code" \
    --num_coefficients 16 \
    --svd_rank 16 \
    --student_model "Qwen/Qwen3.5-4B" \
    --teacher_model "/workspace/Qwen3.5-4B-mix_code" > logs/qwen3.5-4b_lr5e-3_tis_tinylora_n16r16_opd_code.log 2>&1