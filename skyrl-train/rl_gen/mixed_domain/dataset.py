import datasets
from datasets import load_dataset
import os, argparse
import json

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


mlvr_system_prompt = """You are a chatbot who can solve problems. Please solve the following problem and give your thought process."""

instruction_prompt = "Let's think step by step and output the final answer within: \\boxed{}."

code_instruction_prompt = """Write Python code to solve the problem. Present the code in
```python
Your code
```
at the end."""

def filter_by_ability(dataset, ability):
    return dataset.filter(lambda x: x["ability"] == ability)

def make_map_fn_math(split: str, data_source: str):

    def preprocess_fn(example, idx):
        prompt = example.pop("prompt") 
        assert isinstance(prompt, list), "Prompt should be a list of messages."
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"

        answer = example.pop("reward_model")['ground_truth']

        data = {
            "data_source": data_source,
            "prompt": prompt,
            "env_class": "mixed",
            "reward_spec": {
                "method": "math", 
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

def make_map_fn_mlvr(split: str, data_source: str):

    def preprocess_fn(example, idx):
        prompt = example.pop("prompt") 
        assert isinstance(prompt, list), "Prompt should be a list of messages."
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        prompt[0]['content'] = mlvr_system_prompt
        prompt[1]['content'] += '\n' + instruction_prompt

        answer = example.pop("reward_model")['ground_truth']

        data = {
            "data_source": data_source,
            "prompt": prompt,
            "env_class": "mixed",
            "reward_spec": {
                "method": "mlvr", 
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


def make_map_fn_mlvr_valid(split: str, data_source: str):

    def preprocess_fn(example, idx):
        prompt = example.pop("query") 
        assert isinstance(prompt, list), "Prompt should be a list of messages."
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        prompt[0]['content'] = mlvr_system_prompt
        prompt[1]['content'] += '\n' + instruction_prompt

        answer = example.pop("label")

        data = {
            "data_source": data_source,
            "prompt": prompt,
            "env_class": "mixed",
            "reward_spec": {
                "method": "mlvr", 
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

def make_map_fn_code(split: str, data_source: str):
    
    def preprocess_fn(example, idx):
        question = example.pop("problem")
        if "```python" not in question:
            question += '\n' + code_instruction_prompt
        dropped = False
        if example['tests'] is None or len(example['tests']) == 0:
            dropped = True
            tests = []
        
        tests = json.loads(example.pop("tests"))

        # Check inside the tests for any single giant number
        def has_giant_number(obj):
            if isinstance(obj, int) and len(str(abs(obj))) > 1000:
                return True
            if isinstance(obj, list):
                return any(has_giant_number(x) for x in obj)
            if isinstance(obj, dict):
                return any(has_giant_number(v) for v in obj.values())
            return False
        
        if has_giant_number(tests):
            print(f"Skipping example {idx}: contains giant number")
            dropped = True

        # if isinstance(tests, list):
        #     inputs = [test['input'] for test in tests]
        #     outputs = [test['output'] for test in tests]
        #     tests = {"inputs": inputs, "outputs": outputs}

        if isinstance(tests, dict):
            transformed_tests = []
            for i, o in zip(tests["inputs"], tests["outputs"]):
                transformed_tests.append({"input": i, "output": o})
            tests = transformed_tests
            
        ground_truth_str = json.dumps(tests)

        if isinstance(question, dict):
            # question = json.dumps(question)
            assert False, f"question is dict: {question}"

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "env_class": "mixed",
            "reward_spec": {
                "method": "code", 
                "ground_truth": ground_truth_str
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
            },
            "dropped": dropped,
        }
        return data

    return preprocess_fn

def prepare_eurus_data():
    dataset = datasets.load_dataset("uiuc-kang-lab/eurus-2-math-100k", split="train")

    dataset = dataset.map(function=make_map_fn_math("train", "hard_math"), with_indices=True)

    # take the first 13k
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(13333))

    # show a few examples
    print("="*100)
    print("Sample examples from EURUS training dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_mlvr_data():
    dataset = datasets.load_dataset("uiuc-kang-lab/mlvr-100k-rl", split="train")

    dataset = dataset.map(function=make_map_fn_mlvr("train", "mlvr"), with_indices=True)

    # take the first 13k
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(13333))

    # show a few examples
    print("="*100)
    print("Sample examples from EURUS training dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_code_data():
    dataset = datasets.load_dataset('uiuc-kang-lab/code-100k-rl', split='train')
    dataset = dataset.filter(lambda x: x['data_source'] == 'eurus')
    print(f"size of code dataset: {len(dataset)}")
    dataset = dataset.map(function=make_map_fn_code("train", "lcb"), with_indices=True)
    dataset = dataset.filter(lambda x: not x['dropped'])
    # remove the 'dropped' field
    dataset = dataset.remove_columns(['dropped'])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(13333))

    # show a few examples
    print("="*100)
    print("Sample examples from Code LCB training dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            if key == "reward_spec":
                groundtruth = json.loads(value['ground_truth'])
                print(f"tests: {groundtruth[:5]}")
            else:
                print(f"{key}: {value}")
        print("\n")
    return dataset


def prepare_eurus_valid_data():
    dataset = datasets.load_dataset('PRIME-RL/Eurus-2-RL-Data', split='validation')
    # select data with ability = 'math'
    dataset = dataset.filter(lambda x: x['ability'] == 'math')
    print(f"size of eurus valid math dataset: {len(dataset)}")
    dataset = dataset.map(function=make_map_fn_math("validation", "hard_math"), with_indices=True)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(500))
    
    # show a few examples
    print("="*100)
    print("Sample examples from EURUS validation dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_mlvr_valid_data():
    dataset = datasets.load_dataset('virtuoussy/Multi-subject-RLVR', split='test')
    dataset = dataset.map(function=make_map_fn_mlvr_valid("validation", "mlvr"), with_indices=True)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(500))
    # show a few examples
    print("="*100)
    print("Sample examples from MLVR validation dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_code_valid_data():
    dataset = datasets.load_dataset('chuxuan/RL-gen-code-test', split='train')
    dataset = dataset.filter(lambda x: x['source'] not in ['apps', 'lcbv5', 'primeintellect'])
    dataset = dataset.map(function=make_map_fn_code("validation", "lcb"), with_indices=True)
    dataset = dataset.shuffle(seed=42)

    print(f"size of code valid dataset: {len(dataset)}")

    # show a few examples
    print("="*100)
    print("Sample examples from Code LCB validation dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            if key == "reward_spec":
                groundtruth = json.loads(value['ground_truth'])
                print(f"tests: {groundtruth[:5]}")
            else:
                print(f"{key}: {value}")
        print("\n")

    # test json loads on all ground_truth
    for i in range(len(dataset)):
        ground_truth_str = dataset[i]['reward_spec']['ground_truth']
        _ = json.loads(ground_truth_str)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/deepscaler_math")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    math_train_dataset = prepare_eurus_data()
    math_valid_dataset = prepare_eurus_valid_data()
    code_train_dataset = prepare_code_data()
    code_valid_dataset = prepare_code_valid_data()
    mlvr_train_dataset = prepare_mlvr_data()
    mlvr_valid_dataset = prepare_mlvr_valid_data()
    train_dataset = datasets.concatenate_datasets([math_train_dataset, code_train_dataset, mlvr_train_dataset])
    valid_dataset = datasets.concatenate_datasets([math_valid_dataset, code_valid_dataset, mlvr_valid_dataset])
    # valid_dataset = code_valid_dataset  

    # shuffle the datasets
    train_dataset = train_dataset.shuffle(seed=42)

    train_output_path = os.path.join(args.output_dir, "mixed_domain_train.parquet")
    train_dataset.to_parquet(train_output_path)
    print(f"Train dataset saved to {train_output_path}")
    valid_output_path = os.path.join(args.output_dir, "mixed_domain_valid.parquet")
    valid_dataset.to_parquet(valid_output_path)
    print(f"Valid dataset saved to {valid_output_path}")

if __name__ == "__main__":
    main()