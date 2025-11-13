import datasets
import pandas as pd

dataset1 = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/deepscaler/deepscaler_train.parquet", split="train").to_pandas()
dataset2 = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/deepscaler/deepscaler_train_with_wrong_answers_0.5.parquet", split="train").to_pandas()

n_difference = 0
for idx, row in dataset1.iterrows():
    reward_1 = row["reward_spec"]["ground_truth"]
    reward_2 = dataset2.iloc[idx]["reward_spec"]["ground_truth"]
    if reward_1 != reward_2:
        n_difference += 1
        print("Original: ", reward_1)
        print("Modified: ", reward_2)

print(f"{n_difference}/{len(dataset1)}")