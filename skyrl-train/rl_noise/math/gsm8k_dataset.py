# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import re
import os

import datasets


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/gsm8k")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "uiuc-kang-lab/gsm8k-platinum-synthetic-noise"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    system_prompt = ''
#     instruction_following = ''
#     system_prompt = """
# Respond in the following format:

# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")

            noise_param = example.pop("noise")
            solution = extract_solution(answer_raw)
            if len(system_prompt.strip()) > 0:
                prompt = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "env_class": "gsm8k",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": solution,
                },
                "noise_spec": {
                    "param": noise_param,
                    "method": "randomly_generate" # change the answer to a random number in [min, max]
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    # print a few examples
    for i in range(3):
        print(train_dataset[i])
