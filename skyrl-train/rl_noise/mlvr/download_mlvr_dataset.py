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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/gsm8k_llm_judge")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "virtuoussy/Multi-subject-RLVR"

    train_dataset = datasets.load_dataset(data_source, split="train")
    test_dataset = datasets.load_dataset(data_source, split="test")

    instruction_following = 'You are a chatbot who can solve problems. Please solve the following problem and give your thought process. Before giving the final result, you should output \"Therefore, the answer is\", and then give your final answer.'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("query")[-1]["content"]

            question = instruction_following + question_raw

            solution = example.pop("label")
            subject = example.pop("subject")
            data = {
                "data_source": subject,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                # TODO: just repeating the full data preprocess script for a single env change isn't very convenient.
                "env_class": "llm_as_a_judge",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": solution,
                    "question": question_raw,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))
