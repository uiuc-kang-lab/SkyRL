# Noisy Data Pipeline

1. sample model to generate wrong answers 
```bash
bash rl_noise/math_hard sample_wrong_answer.sh --model=<model_name> --n_sample=1 --base_dir=<base_dir>
```

2. modify row #7 and row #57 of `collect_wrong_answers.py` to write wrong answers into a new dataset
```bash
uv run rl_noise/math_hard/collect_wrong_answers.py
```

3. if 2 shows that we have fewer wrong answers than we wanted, re-do 1

4. modify row #8, #43, and #44 to sample wrong answers and mix them with correct answers to have controlled noise
```bash
uv run adjust_noise_rate.py
```
