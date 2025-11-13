import json
import datasets
from collections import defaultdict
import pandas as pd
from copy import deepcopy

paths = ["/data/daniel_kang_group/rl_noise/exports/deepscaler/wrong-answers_Llama-3.1-8B-Instruct/dumped_evals/global_step_0_evals/deepscelar_train.jsonl"]

wrong_answers = defaultdict(list)

for file_path in paths:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            d = json.loads(line)
            reason = d["extra_metadata"]["reason"]
            if reason == "Incorrect answer":
                # {'reason': 'Incorrect answer', 'answer': '34', 'ground_truth': ['26']}
                wrong_answer = d["extra_metadata"]['answer']
                # prompt_key = d['input_prompt'].split('<|im_start|>user')[-1].split('<|im_end|>')[0].strip()
                prompt_key = d["env_extras"]["extra_info"]["question"]
                wrong_answers[prompt_key].append(wrong_answer)


most_frequent_wrong_answers = {k: max(v, key=v.count) for k, v in wrong_answers.items()}

print("Number of unique prompts with wrong answers:", len(most_frequent_wrong_answers))

# load train data
dataset = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/deepscaler/deepscaler_train.parquet", split="train").to_pandas()

n_changes = 0
question_set = set()
for idx, row in dataset.iterrows():
    prompt_key = row["extra_info"]["question"]
    question_set.add(prompt_key)
    if prompt_key in most_frequent_wrong_answers:
        wrong_answer = most_frequent_wrong_answers[prompt_key]
        # print("Before change:", dataset.at[idx, 'reward_spec'])
        dataset.at[idx, 'reward_spec']['ground_truth'] = wrong_answer
        # print("After change:", dataset.at[idx, 'reward_spec'])
        n_changes += 1

original_dataset = datasets.load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/deepscaler/deepscaler_train.parquet", split="train").to_pandas()

# varify the change
n_changes = 0
for idx, row in dataset.iterrows():
    original_reward = original_dataset.at[idx, 'reward_spec']['ground_truth']
    new_reward = row['reward_spec']['ground_truth']
    if original_reward != new_reward:
        n_changes += 1
print(f"Number of changes made: {n_changes}/{len(dataset)}")

# save to parquet file
dataset = datasets.Dataset.from_pandas(dataset)
dataset.to_parquet("/data/daniel_kang_group/rl_noise/data/deepscaler/deepscaler_train_wrong_answers_llama-3.1-8B.parquet")