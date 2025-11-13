import json
import datasets
from collections import defaultdict

paths = [f"/data/daniel_kang_group/rl_noise/exports/mlvr_mlvr-test/judge_results_{i}.jsonl" for i in range(32)]

dataset = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/mlvr/train.parquet", split="train").to_pandas()

print(dataset.iloc[0].to_dict())

wrong_answers = defaultdict(list)

for path in paths:
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            question = d["question"]
            model_answer = d["answer"]
            ground_truth = d["ground_truth"]
            is_correct = d["reward"]
            if not is_correct:
                wrong_answers[question].append(model_answer)

print(f"Number of wrong answers collected: {len(wrong_answers)}/{len(dataset)}")

# replace the ground truth answers in the dataset with a random wrong answer collected
import random
random.seed(42)
n_changes = 0
for idx, row in dataset.iterrows():
    question = row["extra_info"]["question"]
    if question in wrong_answers:
        wrong_answer = random.choice(wrong_answers[question])
        wrong_answer = wrong_answer.strip(".,;: ")
        if idx < 100:
            print("Before change:", dataset.at[idx, 'reward_spec'])
        dataset.at[idx, 'reward_spec']['ground_truth'] = wrong_answer
        if idx < 100:
            print("After change:", dataset.at[idx, 'reward_spec'])
        n_changes += 1
print(f"Number of changes made: {n_changes}/{len(dataset)}")
# save to parquet file
dataset = datasets.Dataset.from_pandas(dataset)
dataset.to_parquet("/data/daniel_kang_group/rl_noise/data/mlvr/train_Qwen2.5-7B-Instruct_wrong-answers.parquet")
print("Saved to /data/daniel_kang_group/rl_noise/data/mlvr/train_Qwen2.5-7B-Instruct_wrong-answers.parquet")