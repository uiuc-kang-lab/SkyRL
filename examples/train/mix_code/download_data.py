import argparse
import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

"""Utility functions for loading and processing datasets."""

SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

KODCODE_FORMATTING_MESSAGE_WITH_FUNCTION_DECLARATION = """Please solve the programming task below in Python. 
{question}

Note that the function declaration is {function_declaration}. Your code should be wrapped in a markdown code block"""

KODCODE_FORMATTING_MESSAGE_WITH_FUNCTION_DECLARATIONS = """Please solve the programming task below in Python.

{question}

Note that the functions you need to implement have the following declarations:
{function_declarations}. 

Your code should be wrapped in a markdown code block"""

KODCODE_FORMATTING_MESSAGE_WITHOUT_FUNCTION_DECLARATION = """Please solve the programming task below in Python. 
{question}"""


def process_eurus_code_example(example: Dict[str, Any], idx: int, dataset_name: str, split: str) -> Optional[Dict[str, Any]]:
    """Process a single dataset example.

    Args:
        example: Raw dataset example
        idx: Index of the example
        dataset_name: Name of the dataset
        split: Dataset split ('train' or 'test')

    Returns:
        Processed example dictionary or None if processing fails
    """
    question = example.pop("prompt")[-1]["content"]
    tests = json.loads(example.pop("reward_model")["ground_truth"])
    
    assert isinstance(tests, dict), f"Expected tests to be a dict, got {type(tests)}: {tests}"
    assert "inputs" in tests and "outputs" in tests
    assert isinstance(tests["inputs"], list) and isinstance(tests["outputs"], list)
    assert len(tests["inputs"]) == len(tests["outputs"])

    tests = json.dumps(tests)

    data = {
        "data_source": dataset_name,
        "prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_spec": {"method": "inputs", "ground_truth": tests},
        "extra_info": {
            "split": split,
            "index": idx
        },
    }
    return data

def process_kodcode_example(example: Dict[str, Any], idx: int, dataset_name: str, split: str) -> Optional[Dict[str, Any]]:
    """Process a single dataset example.

    Args:
        example: Raw dataset example
        idx: Index of the example
        dataset_name: Name of the dataset
        split: Dataset split ('train' or 'test')

    Returns:
        Processed example dictionary or None if processing fails
    """
    question = example.pop("question")
    test = example.pop("test")
    test_info = example.pop("test_info")
    if len(test_info) > 1:
        # figure out which function declarations are present in test
        condensed_test_info = []
        for info in test_info:
            assert "function_declaration" in info, f"Expected 'function_declaration' in test_info, got {test_info}"
            function_name = info["function_name"]
            assert function_name in info["function_declaration"], f"Expected function name {function_name} in function declaration {info['function_declaration']}"
            if function_name + "(" in test:
                condensed_test_info.append(info)
        test_info = condensed_test_info
    if len(test_info) == 1:
        assert "function_declaration" in test_info[0], f"Expected 'function_declaration' in test_info, got {test_info}"
        function_declaration = test_info[0]["function_declaration"]
        prompt = KODCODE_FORMATTING_MESSAGE_WITH_FUNCTION_DECLARATION.format(question=question, function_declaration=function_declaration)
    elif len(test_info) > 1:
        function_declarations = "\n".join([info["function_declaration"] for info in test_info])
        prompt = KODCODE_FORMATTING_MESSAGE_WITH_FUNCTION_DECLARATIONS.format(question=question, function_declarations=function_declarations)
    else:
        prompt = KODCODE_FORMATTING_MESSAGE_WITHOUT_FUNCTION_DECLARATION.format(question=question)

    assert isinstance(test, str)

    data = {
        "data_source": dataset_name,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "code",
        "reward_spec": {"method": "assertions", "ground_truth": test},
        "extra_info": {
            "split": split,
            "index": idx
        },
    }
    return data


def process_eurus_code_data(local_dir: str):
    """Process a dataset for Eurus 2 RL data code split.

    Args:
        local_dir: Directory to save processed datasets
        max_rows: Maximum number of rows to process
    """
    # Load dataset
    raw_data = load_dataset("PRIME-RL/Eurus-2-RL-Data")["train"]
    
    # filter for ability = code
    raw_data = raw_data.filter(lambda x: x["ability"] == "code")

    # Process examples
    processed_data = []
    for idx, example in enumerate(raw_data):
        processed_example = process_eurus_code_example(example, idx, "eurus", "train")
        processed_data.append(processed_example)

    return processed_data

def process_kodcode_data(local_dir: str):
    """Process a dataset for KodCode.

    Args:
        local_dir: Directory to save processed datasets
        max_rows: Maximum number of rows to process
    """
    # Load dataset
    raw_data = load_dataset("KodCode/KodCode-V1")["train"]
    
    # Process examples
    processed_data = []
    for idx, example in enumerate(raw_data):
        processed_example = process_kodcode_example(example, idx, "kodcode", "train")
        processed_data.append(processed_example)

    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets for code training")
    parser.add_argument("--output_dir", default="~/data/eurus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset")
    args = parser.parse_args()

    local_dir = args.output_dir
    print(f"Local_dir:{local_dir}")

    # Make local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # Process train dataset
    eurus_code_data = process_eurus_code_data(local_dir)
    kodcode_data = process_kodcode_data(local_dir)
    
    # print a few examples for sanity check
    print("Sample Eurus code example:")
    print(json.dumps(eurus_code_data[0], indent=2))
    print("\nSample KodCode example:")
    print(json.dumps(kodcode_data[0], indent=2))

    # merge and shuffle the datasets
    train_data = eurus_code_data + kodcode_data
    import random
    random.seed(args.seed)
    random.shuffle(train_data)
    
    # subset for validation
    validation_data = train_data[:1000]
    train_data = train_data[1000:]

    # Save combined train dataset
    train_data_df = pd.DataFrame(train_data)
    train_data_df.to_parquet(os.path.join(local_dir, "train_data.parquet"), index=False)
    print(f"Saved combined train dataset with {len(train_data)} examples to {os.path.join(local_dir, 'train_data.parquet')}")
    # Save validation dataset
    validation_data_df = pd.DataFrame(validation_data)
    validation_data_df.to_parquet(os.path.join(local_dir, "validation_data.parquet"), index=False)
    print(f"Saved validation dataset with {len(validation_data)} examples to {os.path.join(local_dir, 'validation_data.parquet')}")
