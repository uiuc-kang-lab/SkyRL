import datasets
import pandas as pd
import sys

noise_rate = sys.argv[1]

gt_answers = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/mlvr/train.parquet", split="train").to_pandas()
wrong_answers = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/mlvr/train_Qwen2.5-7B-Instruct_wrong-answers.parquet", split="train").to_pandas()

n_difference = 0
wrong_answer_ids = []
for idx, row in gt_answers.iterrows():
    reward_1 = row["reward_spec"]["ground_truth"]
    reward_2 = wrong_answers.iloc[idx]["reward_spec"]["ground_truth"]
    if reward_1 != reward_2:
        n_difference += 1
        wrong_answer_ids.append(idx)

print(f"Original wrong answer rate: {n_difference}/{len(gt_answers)}")

n_to_recover = n_difference - int(len(gt_answers) * float(noise_rate))
print(f"Number of wrong answers to recover: {n_to_recover}")

# randomly sample n_to_recover ids
import random
random.seed(42)
ids_to_recover = random.sample(wrong_answer_ids, n_to_recover)
# print(f"IDs to recover: {ids_to_recover}")
for idx in ids_to_recover:
    wrong_answers.at[idx, 'reward_spec']['ground_truth'] = gt_answers.at[idx, 'reward_spec']['ground_truth']

# verify the change
n_difference_after = 0
for idx, row in gt_answers.iterrows():
    reward_1 = row["reward_spec"]["ground_truth"]
    reward_2 = wrong_answers.iloc[idx]["reward_spec"]["ground_truth"]
    if reward_1 != reward_2:
        n_difference_after += 1
print(f"New wrong answer rate: {n_difference_after}/{len(gt_answers)}={n_difference_after/len(gt_answers):.4f}")

# save to parquet file
wrong_answers = datasets.Dataset.from_pandas(wrong_answers)
wrong_answers.to_parquet(f"/data/daniel_kang_group/rl_noise/data/mlvr/train_with_wrong_answers_{noise_rate}.parquet")
print(f"Saved to /data/daniel_kang_group/rl_noise/data/mlvr/train_Qwen2.5-7B-Instruct_wrong-answers_{noise_rate}.parquet")