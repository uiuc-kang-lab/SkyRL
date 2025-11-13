import datasets
from datasets import load_dataset
import os, argparse

system_prompt = """
When tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.

[ASSESS]

[ADVANCE]

[VERIFY]

[SIMPLIFY]

[SYNTHESIZE]

[PIVOT]

[OUTPUT]

You should strictly follow the format below:

[ACTION NAME]

# Your action step 1

# Your action step 2
 
# Your action step 3

...

Next action: [NEXT ACTION NAME]

"""

instruction_prompt = "Present the answer in LaTex format: \\boxed{Your answer}"

def filter_by_ability(dataset, ability):
    return dataset.filter(lambda x: x["ability"] == ability)

def make_map_fn(split: str, data_source: str):

    def preprocess_fn(example, idx):
        if "prompt" in example:
            prompt = example.pop("prompt") 
            assert isinstance(prompt, list), "Prompt should be a list of messages."
            assert len(prompt) == 2
            assert prompt[0]["role"] == "system"
            assert prompt[1]["role"] == "user"
        else:
            question = example.pop("problem").strip() if "problem" in example else example.pop("question").strip()
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"{question}\n\n{instruction_prompt}"
                }
            ]

        if 'reward_model' in example:
            answer = example.pop("reward_model")['ground_truth']
        else:
            answer = example.pop('answer')

        data = {
            "data_source": data_source,
            "prompt": prompt,
            "env_class": "math_hard",
            "reward_spec": {
                "method": "rule", 
                "ground_truth": str(answer)
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
                "answer": str(answer),
                "question": prompt[1]['content'],
            }
        }
        return data

    return preprocess_fn

def prepare_eurus_data():
    dataset = datasets.load_dataset("uiuc-kang-lab/eurus-2-math-100k", split="train")

    dataset = dataset.map(function=make_map_fn("train", "eurus_math"), with_indices=True)

    # take the first 40k
    dataset = dataset.select(range(40000))

    # show a few examples
    print("="*100)
    print("Sample examples from EURUS training dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_eurus_valid_data():
    dataset = datasets.load_dataset('PRIME-RL/Eurus-2-RL-Data', split='validation')
    # select data with ability = 'math'
    dataset = dataset.filter(lambda x: x['ability'] == 'math')
    print(f"size of eurus valid math dataset: {len(dataset)}")
    dataset = dataset.map(function=make_map_fn("validation", "eurus_math"), with_indices=True)
    
    # show a few examples
    print("="*100)
    print("Sample examples from EURUS validation dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_aime2024_data():
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "aime2024"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AIME2024 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_aime2025_data():
    dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    dataset = datasets.concatenate_datasets([dataset1, dataset2])

    dataset = dataset.map(function=make_map_fn("test", "aime2025"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AIME2025 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_amc2024_data():
    dataset = load_dataset("rawsh/2024_AMC12", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "amc2024"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AMC2024 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_math500_data():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "math500"), with_indices=True)
    # show a few examples
    print("="*100)
    print("Sample examples from MATH500 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_amc2023_data(file_path):
    if not os.path.exists(file_path):
        print("Please download the AMC2023 test dataset from https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl and save it to", file_path)
        return {}
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.map(function=make_map_fn("test", "amc2023"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AMC2023 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_minervamath_data():
    dataset = load_dataset("math-ai/minervamath", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "minervamath"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from MinervaMath test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/deepscaler_math")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = prepare_eurus_data()
    valid_dataset = prepare_eurus_valid_data()
    test_datasets = {
        "aime2024": prepare_aime2024_data(),
        "aime2025": prepare_aime2025_data(),
        "amc2024": prepare_amc2024_data(),
        "math500": prepare_math500_data(),
        # "amc2023": prepare_amc2023_data(args.output_dir + "/amc2023.jsonl"),
        "minervamath": prepare_minervamath_data()
    }

    train_output_path = os.path.join(args.output_dir, "deepscaler_train.parquet")
    train_dataset.to_parquet(train_output_path)
    print(f"Train dataset saved to {train_output_path}")
    valid_output_path = os.path.join(args.output_dir, "deepscaler_valid.parquet")
    valid_dataset.to_parquet(valid_output_path)
    print(f"Valid dataset saved to {valid_output_path}")

    for name, dataset in test_datasets.items():
        test_output_path = os.path.join(args.output_dir, f"{name}_test.parquet")
        dataset.to_parquet(test_output_path)
        print(f"Test dataset {name} saved to {test_output_path}")


if __name__ == "__main__":
    main()