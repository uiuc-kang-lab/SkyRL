import datasets

clean_trian_dataset = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/arc/train.parquet", split="train")
noisy_train_data_files = [f"/data/daniel_kang_group/rl_noise/data/arc/train_noise_{i}.parquet" for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
for noisy_data_file in noisy_train_data_files:
    noisy_train_dataset = datasets.load_dataset("parquet", data_files=noisy_data_file, split="train")
    n_difference = 0
    for idx, row in clean_trian_dataset.to_pandas().iterrows():
        reward_1 = row["reward_spec"]["ground_truth"]
        reward_2 = noisy_train_dataset.to_pandas().iloc[idx]["reward_spec"]["ground_truth"]
        if reward_1 != reward_2:
            n_difference += 1
    print(f"For {noisy_data_file}, {n_difference}/{len(clean_trian_dataset)}={n_difference/len(clean_trian_dataset):.4f}")