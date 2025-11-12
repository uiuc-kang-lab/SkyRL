import argparse
import json
from collections import defaultdict
import re
import datasets
import glob
import random

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", ".", ","]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer.replace(",", "").strip(". ")

# read from dumped files
# base_dir = "/data/daniel_kang_group/rl_noise/exports/gsm8k/wrong-answers_Qwen2.5-1.5B-Instruct"

# data_file = f"{base_dir}/dumped_evals/global_step_0_evals/uiuc-kang-lab_gsm8k-platinum-synthetic-noise.jsonl"

all_stop_reasons = set()
def collect_wrong_answers(data_file):
    valid_wrong_answers = defaultdict(list)
    with open(data_file) as f:
        lines = f.readlines()
        for line in lines:
            d = json.loads(line)
            if d['stop_reason'] != 'stop':
                continue
            question_key = d['env_extras']['extra_info']['question']
            output_response = d['output_response']
            ground_truth = d['env_extras']['reward_spec']['ground_truth']
            model_answer = extract_solution(output_response, 'flexible')
            try:
                float(model_answer)
            except Exception as e:
                print(e)
                print(output_response)
                exit()
            if model_answer is not None and float(model_answer) != float(ground_truth):
                # print(f"Found valid and wrong answer: {model_answer} VS {ground_truth}")
                valid_wrong_answers[question_key].append(model_answer)
    return valid_wrong_answers

def replace_answer_in_solution(dataset_file, valid_wrong_answers):
    dataset = datasets.load_dataset('parquet', data_files=dataset_file)['train'].to_pandas()
    n_updated = 0
    k_sample = random.sample(list(valid_wrong_answers.keys()), k=len(dataset) // 2)
    valid_wrong_answers = {key: value for key, value in valid_wrong_answers.items() if key in k_sample}
    print(f"After subsampling, {len(valid_wrong_answers)} questions with valid wrong answers")
    for idx, row in dataset.iterrows():
        question_key = row['extra_info']['question']
        if question_key in valid_wrong_answers:
            # print(f"Original reward spec: {row['reward_spec']}")
            wrong_answers = valid_wrong_answers[question_key]
            reward_spec = row['reward_spec']
            # replace the ground truth with the wrong answer that appears in the model output for the most of times
            reward_spec['ground_truth'] = max(set(wrong_answers), key=wrong_answers.count)
            dataset.at[idx, 'reward_spec'] = reward_spec
            # print(f"Updated reward spec: {dataset.at[idx, 'reward_spec']}")
            n_updated += 1
    print(f"Updated {n_updated} samples in the dataset (out of {len(dataset)})")
    dataset = datasets.Dataset.from_pandas(dataset)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default="/data", help='The base directory for data files')
    parser.add_argument('--model_name', type=str, default="Qwen2.5-1.5B-Instruct", help='The model name')
    parser.add_argument('--output_file', type=str, default=f"/data/noisy_gsm8k_dataset.parquet", help='The output file to write to')
    args = parser.parse_args()
    random.seed(42)

    data_files = glob.glob(f"{args.base_dir}/exports/gsm8k/wrong-answers*_{args.model_name}/dumped_evals/global_step_0_evals/uiuc-kang-lab_gsm8k-platinum-synthetic-noise.jsonl")
    dataset = f"{args.base_dir}/data/gsm8k/train.parquet"  # the original dataset file

    valid_wrong_answers = defaultdict(list)
    for data_file in data_files:
        valid_wrong_answers.update(collect_wrong_answers(data_file))
        print(f"found {len(valid_wrong_answers)} questions with valid wrong answers so far")
    updated_dataset = replace_answer_in_solution(dataset, valid_wrong_answers)

    updated_dataset.to_parquet(args.output_file)
    print(f"Written noisy dataset to {args.output_file}")