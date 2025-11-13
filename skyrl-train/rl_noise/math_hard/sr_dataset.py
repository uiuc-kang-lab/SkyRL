from datasets import load_dataset

# train data: "agentica-org/DeepScaleR-Preview-Dataset"
# test data: "opencompass/AIME2025", "HuggingFaceH4/aime_2024", "rawsh/2024_AMC12", "HuggingFaceH4/MATH-500", "https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl", "math-ai/minervamath"


import argparse
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import datasets
import pandas as pd
import random
import requests

instruction = "Let's think step by step and output the final answer within \\boxed{}."

def make_map_fn(split: str, data_source: str):

    def preprocess_fn(example, idx):
        if "problem" in example:
            question = example.pop("problem")
        elif "question" in example:
            question = example.pop("question")
        elif "prompt" in example:
            question = example.pop("prompt")
        else:
            raise ValueError("No question field found in the example.")
        
        if data_source == "random_noise":
            answer = str(int(random.random() * 1000))
            if "groundtruth_answer" in example:
                example.pop("groundtruth_answer")
            if "answer" in example:
                example.pop("answer")
        elif data_source == "incorrect":
            answer = example.pop("answer")
            if "groundtruth_answer" in example:
                example.pop("groundtruth_answer")
        elif data_source == "default":
            answer = example.pop("groundtruth_answer")
            if "answer" in example:
                example.pop("answer")
        else:
            answer = example.pop("answer")
            
        if data_source == "format_reward":
            reward_method = "format"
        else:
            reward_method = "rule"

        if "gpt_answer" in example:
            example.pop("gpt_answer")

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user", 
                    "content": f"{question} {instruction}"
                }
            ],
            "env_class": "math_hard",
            "reward_spec": {
                "method": reward_method, 
                "ground_truth": str(answer),
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
                "answer": str(answer),
                "question": question,
            }
        }
        return data

    return preprocess_fn

def prepare_sr_data(mode: str):
    # with open('truly_incorrect_answers_qwen2.5_math_7b.json', 'r') as f:
    #     sr_data = json.load(f)
    sr_data = datasets.load_dataset('json', data_files='../../../../truly_incorrect_answers_qwen2.5_math_7b.json', split='train')
    if mode == 'clean':
        sr_data = sr_data.map(function=make_map_fn("train", "default"), with_indices=True)
    elif mode == 'random_noise':
        # change answer to a random number between 0 and 1000
        sr_data = sr_data.map(function=make_map_fn("train", "random_noise"), with_indices=True)
    elif mode == 'pr_noise':
        sr_data = sr_data.map(function=make_map_fn("train", "incorrect"), with_indices=True)
    elif mode == 'format_reward':
        sr_data = sr_data.map(function=make_map_fn("train", "format_reward"), with_indices=True)

    # show a few examples:
    print("="*100)
    print(f"Sample examples from SR training dataset ({mode}):")
    for i in range(3):
        for key, value in sr_data[i].items():
            print(f"{key}: {value}")
        print("\n")
    return sr_data

def prepare_validate_data():
    validate_data = datasets.load_dataset('json', data_files='maybe_correct_answers_qwen2.5_math_7b.json', split='train')
    validate_data = validate_data.map(function=make_map_fn("validate", "default"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from validation dataset:")
    for i in range(3):
        for key, value in validate_data[i].items():
            print(f"{key}: {value}")
        print("\n")
    return validate_data

def prepare_aime2024_data():
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "aime2024"), with_indices=True)

    # show a few examples
    # print("="*100)
    # print("Sample examples from AIME2024 test dataset:")
    # for i in range(3):
    #     for key, value in dataset[i].items():
    #         print(f"{key}: {value}")
    #     print("\n")

    return dataset

def prepare_aime2025_data():
    dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    dataset = datasets.concatenate_datasets([dataset1, dataset2])

    dataset = dataset.map(function=make_map_fn("test", "aime2025"), with_indices=True)

    # show a few examples
    # print("="*100)
    # print("Sample examples from AIME2025 test dataset:")
    # for i in range(3):
    #     for key, value in dataset[i].items():
    #         print(f"{key}: {value}")
    #     print("\n")
    return dataset

def prepare_amc2024_data():
    dataset = load_dataset("rawsh/2024_AMC12", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "amc2024"), with_indices=True)

    # show a few examples
    # print("="*100)
    # print("Sample examples from AMC2024 test dataset:")
    # for i in range(3):
    #     for key, value in dataset[i].items():
    #         print(f"{key}: {value}")
    #     print("\n")
    return dataset

def prepare_math500_data():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "math500"), with_indices=True)
    # show a few examples
    # print("="*100)
    # print("Sample examples from MATH500 test dataset:")
    # for i in range(3):
    #     for key, value in dataset[i].items():
    #         print(f"{key}: {value}")
    #     print("\n")

    return dataset

def prepare_amc2023_data(file_path):
    if not os.path.exists(file_path):
        print("Please download the AMC2023 test dataset from https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl and save it to", file_path)
        return {}
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.map(function=make_map_fn("test", "amc2023"), with_indices=True)

    # show a few examples
    # print("="*100)
    # print("Sample examples from AMC2023 test dataset:")
    # for i in range(3):
    #     for key, value in dataset[i].items():
    #         print(f"{key}: {value}")
    #     print("\n")

    return dataset

def prepare_minervamath_data():
    dataset = load_dataset("math-ai/minervamath", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "minervamath"), with_indices=True)

    # show a few examples
    # print("="*100)
    # print("Sample examples from MinervaMath test dataset:")
    # for i in range(3):
    #     for key, value in dataset[i].items():
    #         print(f"{key}: {value}")
    #     print("\n")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/deepscaler_math")
    parser.add_argument("--mode", type=str, required=True, choices=["clean", "random_noise", "pr_noise", "validate", "format_reward"])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ['clean', 'random_noise', 'pr_noise', 'format_reward']:
        train_dataset = prepare_sr_data(mode=args.mode)
    elif args.mode == 'validate':
        train_dataset = prepare_validate_data()

    test_datasets = {
        "aime2024": prepare_aime2024_data(),
        "aime2025": prepare_aime2025_data(),
        "amc2024": prepare_amc2024_data(),
        "math500": prepare_math500_data(),
        "amc2023": prepare_amc2023_data(args.output_dir + "/amc2023.jsonl"),
        "minervamath": prepare_minervamath_data()
    }

    train_output_path = os.path.join(args.output_dir, f"deepscaler_train_{args.mode}.parquet")
    train_dataset.to_parquet(train_output_path)
    print(f"Train dataset saved to {train_output_path}")

    for name, dataset in test_datasets.items():
        test_output_path = os.path.join(args.output_dir, f"{name}_test.parquet")
        dataset.to_parquet(test_output_path)
        print(f"Test dataset {name} saved to {test_output_path}")




