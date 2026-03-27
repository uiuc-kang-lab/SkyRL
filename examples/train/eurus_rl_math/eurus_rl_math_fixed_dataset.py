"""
Preprocess the Eurus-RL-Math dataset to parquet format
"""

import argparse
import re
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/eurus")
    parser.add_argument(
        "--max_train_dataset_length",
        type=int,
        default=None,
        help="If set, truncate the training split to this many examples.",
    )

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "uiuc-kang-lab/Eurus-2-RL-Data-Math-Fixed"

    dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    # val_dataset = dataset["validation"]
    
    # filter ability=math
    train_dataset = train_dataset.filter(lambda x: x["ability"] == "math")
    # val_dataset = val_dataset.filter(lambda x: x["ability"] == "math")

    if args.max_train_dataset_length is not None:
        max_len = min(args.max_train_dataset_length, len(train_dataset))
        train_dataset = train_dataset.select(range(max_len))
        
        
    # measure the max and min prompt length
    max_prompt_length = 0
    min_prompt_length = float("inf")
    for example in train_dataset:
        question = example["prompt"][-1]["content"]
        if len(question) > max_prompt_length:
            max_prompt_length = len(question)
        if len(question) < min_prompt_length:
            min_prompt_length = len(question)
    print(f"Max prompt length: {max_prompt_length}, Min prompt length: {min_prompt_length}")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")[-1]["content"]

            solution = example.pop("reward_model")
            reward_style = solution["style"]
            solution = solution["ground_truth"]
            
            assert reward_style == "rule", "Only support rule-based reward for now."
            assert "Present the answer in LaTex format: \\boxed{Your answer}" in question, question
            assert isinstance(solution, list), solution
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a mathematical reasoning assistant. "
                                    "Work through each problem step by step, then state your final answer.\n\n"
                                    "Format your final answer inside \\boxed{} as instructed in each problem.\n\n"
                                    "For multiple-choice problems, write only the letter of the correct option "
                                    "(e.g. \\boxed{A}) — do not include the option text. "
                                    "For multiple-choice problem with multiple correct options, "
                                    "write all correct option letters together inside the same box "
                                    "and separate them with commas (e.g. \\boxed{A, C})"
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "env_class": "eurus_rl_math",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True).shuffle(seed=42)
    # val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)
    val_dataset = train_dataset.shuffle(seed=43).select(range(1000))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
