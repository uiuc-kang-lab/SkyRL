import json
import datasets
import pandas as pd
from copy import deepcopy

# dict_keys(['input_prompt', 'output_response', 'score', 'stop_reason', 'env_class', 'env_extras', 'data_source', 'extra_metadata'])
# paths = [f"/data/daniel_kang_group/rl_noise/exports/lcb_qwen3-8b_testgen_{i}/dumped_evals/global_step_0_evals/livecodebench.jsonl" for i in [1,2,3]]

paths = ['/data/daniel_kang_group/rl_noise/exports/lcb_lcb_qwen3-8B_gen-wrong-answers/dumped_evals/global_step_0_evals/livecodebench.jsonl']


data_files = [f"/data/daniel_kang_group/rl_noise/data/lcb/train_livecodebench_part{i}.json" for i in [0,1,2,3,4,5,6,7]]

all_train_data = pd.DataFrame()
for data_file in data_files:
    train_data = datasets.load_dataset("json", data_files=data_file, keep_in_memory=True)
    train_data = pd.DataFrame(train_data['train'])
    train_data["prompt_key"] = [d['prompt'][0]["content"].strip() for _, d in train_data.iterrows()]
    all_train_data = pd.concat([all_train_data, train_data], ignore_index=True)

print(f"Train data size: {len(all_train_data)}")

all_train_data_noise = all_train_data.copy()

print(eval(all_train_data_noise.iloc[0]['reward_spec']['ground_truth'])[0])

prompt_key_to_gt = {}

for path in paths:
    data = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            d['input_prompt'] = d['input_prompt'].split('<|im_start|>user')[-1].split('<|im_end|>')[0].strip()
            if d['score'] != 1 and "output" in d["extra_metadata"]['metadata_list'][0] and \
            d['input_prompt'] not in prompt_key_to_gt:
                input = d['extra_metadata']['metadata_list'][0]['inputs']
                if "truncated" in input:
                    continue
                output = d['extra_metadata']['metadata_list'][0]['output']
                assert isinstance(input, str) and isinstance(output, str)

                ground_truth = json.loads(all_train_data_noise[all_train_data_noise['prompt_key'] == d['input_prompt']].iloc[0]['reward_spec']['ground_truth'])[0]

                # if ground_truth['input'] != input and isinstance(eval(input), list) and len(eval(input)) == 1:
                #     input = json.dumps(eval(input)[0])
                if ground_truth['input'] != input:
                    if input.strip().startswith('['):
                        input = eval(input)
                        input = [json.dumps(i) for i in input]
                        input = "\n".join(input)
                output = json.dumps(output)
                if ground_truth['output'] != output:
                    print(f"output mismatch: {ground_truth['output']} vs {output}")

                ground_truth["input"] = input
                ground_truth["output"] = output
                
                # reset reward spec
                prompt_key_to_gt[d['input_prompt']] = json.dumps([ground_truth])


def process_fn(example):
    prompt_key = example['prompt_key']
    if prompt_key in prompt_key_to_gt:
        new_example = deepcopy(example)
        new_example['reward_spec']['ground_truth'] = prompt_key_to_gt[prompt_key]
        print(f"replacing ground truth to {new_example['reward_spec']['ground_truth']}")
        return new_example
    else:
        return example

all_train_data_noise = all_train_data_noise.apply(process_fn, axis=1)

print(f"Number of changes: {len(prompt_key_to_gt)}")

# split into 8 json files
for i in range(8):
    part_json_path = f"/data/daniel_kang_group/rl_noise/data/lcb/train_livecodebench_part{i}_noise.json"
    if i != 7:
        part_df = all_train_data_noise.iloc[i * (len(all_train_data_noise) // 8):(i + 1) * (len(all_train_data_noise) // 8)]
    else:
        part_df = all_train_data_noise.iloc[i * (len(all_train_data_noise) // 8):]
    print(f"Writing {len(part_df)} rows to train_livecodebench_part{i}_noise.json")
    part_df.to_json(part_json_path, orient="records")