# Noise Data Pipeline
1. sample wrong answers by executing `run_mlvr_gen_answer.sh`
```bash
bash rl_noise/mlvr/run_mlvr_gen_answer.sh --model=<model_name> --base_dir=<base_dir>
```
2. judge the correctness of answers by running `judge_generations.py`
```bash
uv run rl_noise/mlvr/judge_generations.py <batch_id> # 32 batches in total
```
3. construct a dataset with wrong answers by modifying line #5 and #44 and running `construct_dataset.py`
```bash
uv run rl_noise/mlvr/construct_dataset.py
```
4. adjust the noise rate to have controled noisy data
```bash
uv run rl_noise/mlvr/adjust_noise_rate.py <noise_rate> # noise rate should be between 0 and 1
```