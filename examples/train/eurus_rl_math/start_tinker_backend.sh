uv run --extra gpu --extra tinker -m skyrl.tinker.api \
 --base-model Qwen/Qwen3.5-4B \
 --backend-config '{"max_lora_adapters": 1000, "max_lora_rank": 4, "tensor_parallel_size": 8, "train_micro_batch_size": 1, "sample_max_num_sequences": 64, "shard_attention_heads": false}' > out.log