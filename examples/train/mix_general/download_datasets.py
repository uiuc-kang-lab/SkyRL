import json
import datasets
from tqdm import tqdm
import os
import argparse
import re

MAQ_TEMPLATE = "Question: {question}\nAnswer Choices:\n{choices}"
LOGIQA_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer Choices:\n{choices}"
system_prompt = "You are a helpful and precise assistant for answering questions. " \
                "Work through each problem step by step, then state your final answer.\n\n" \
                "Format your final answer inside \\boxed{}.\n\n" \
                "For multiple-choice problems, write only the letter of the correct option " \
                "(e.g. \\boxed{A}) — do not include the option text. " \
                "For other problems, just write the final answer inside the box. Do not include any explanation in the box." \

def load_mmlu():
    def process_fn(example, idx):
        question = example.pop("question")
        choices = example.pop("choices")
        answer = example.pop("answer")
        subject = example.pop("subject")
        
        if answer in [0, 1, 2, 3]:
            answer = chr(65 + answer)
        
        assert len(choices) == 4, choices
        assert answer in ["A", "B", "C", "D"], f"Invalid answer: {answer}, question: {question}, choices: {choices}"
        
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = MAQ_TEMPLATE.format(question=question, choices=choices_str)
        
        data = {
            "data_source": "mmlu",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "env_class": "mix_general",
            "reward_spec": {
                "method": "mqa",
                "ground_truth": answer,
            },
            "extra_info": {
                "index": idx,
                "answer": answer,
                "question": question,
                "choices": choices,
                "detailed_source": f"mmlu-{subject}",
            }
        }
        
        return data
        
    data = datasets.load_dataset("cais/mmlu", "all")
    train_data = data["auxiliary_train"].map(function=process_fn, with_indices=True)
    val_data = data["validation"].map(function=process_fn, with_indices=True)
    test_data = data["test"].map(function=process_fn, with_indices=True)
    dev_data = data["dev"].map(function=process_fn, with_indices=True)
    all_data = datasets.concatenate_datasets([train_data, val_data, test_data, dev_data])
    print(f"Total data size: {len(all_data)}")

    return all_data
        
def load_webinstruct():
    
    def is_valid_answer(example):
        answer = example["answer"]
        answer_type = example["answer_type"]
        
        try:
            if answer_type == "Float":
                # find the first number in the answer string and use it as the ground truth, allow exponential notation
                answer = answer.replace(",", "")
                regex = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
                match = re.search(regex, answer)
                assert match is not None, f"Cannot find a number in the answer: {answer}"
                n_matches = len(re.findall(regex, answer))
                # if n_matches > 1:
                #     print(f"Found multiple numbers in the answer: {answer}")
                assert n_matches == 1, f"Found multiple numbers in the answer: {answer}"
                answer = match.group(0)
            elif answer_type == "Multiple Choice":
                pass
            elif answer_type == "String":
                pass
            elif answer_type == "Integer":
                # find the first integer in the answer string and use it as the ground truth, allow comma as thousand separator
                answer = answer.replace(",", "")
                match = re.search(r"[-+]?\d+", answer)
                assert match is not None, f"Cannot find an integer in the answer: {answer}"
                n_matches = len(re.findall(r"[-+]?\d+", answer))
                # if n_matches > 1:
                #     print(f"Found multiple integers in the answer: {answer}")
                assert n_matches == 1, f"Found multiple integers in the answer: {answer}"
                answer = match.group(0)
            elif answer_type == "List":
                assert "," in answer, f"Cannot find a comma in the answer: {answer}"
                # split the answer string by comma and use the resulting list as the ground truth
                answer = json.dumps([item.strip() for item in answer.split(",")])
            elif answer_type == "Percentage":
                assert answer.endswith("%"), f"Percentage answer should end with %: {answer}"
                pass
            elif answer_type == "Expression":
                pass
            elif answer_type == "Boolean":
                assert answer.lower() in ["yes", "no", "true", "false"], f"Boolean answer should be True or False: {answer}"
            elif answer_type == "Fraction":
                assert "/" in answer, f"Fraction answer should contain /: {answer}"
                assert len(answer.split("/")) == 2, f"Fraction answer should contain only one /: {answer}"
                pass
            else:
                raise ValueError(f"Unsupported answer type: {answer_type}")
            return answer
        except Exception as e:
            # print(f"Invalid answer: {answer}, answer type: {answer_type}, error: {e}")
            return None

    def process_fn(example, idx):
        question = example.pop("question")
        answer = example.pop("answer")
        answer_type = example.pop("answer_type")
        category = example.pop("category")
        difficulty = example.pop("difficulty")
        webinstruct_id = example.pop("id")
        
        assert category in ['Physics', 'History', 'Other', 'Business', 
                            'Economics', 'Other STEM', 'Law', 'Health', 
                            'Biology', 'Chemistry', 'Engineering', 'Philosophy', 
                            'Psychology', 'Finance', 'Computer Science'], category
        assert answer_type in ["Float", "Multiple Choice", "String", "Integer", "List", "Percentage", "Expression", "Boolean", "Fraction"], answer_type
        
        answer = is_valid_answer({
            "answer": answer,
            "answer_type": answer_type,
        })
        assert answer is not None, f"Invalid answer: {answer}, answer type: {answer_type}"
            
        data = {
            "data_source": "webinstruct",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "env_class": "mix_general",
            "reward_spec": {
                "method": f"webinstruct-{answer_type}",
                "ground_truth": answer,
            },
            "extra_info": {
                "index": idx,
                "answer": answer,
                "question": question,
                "category": category,
                "detailed_source": f"webinstruct-{webinstruct_id}-{category}-{difficulty}"
            }
        }
        
        return data
    
    data = datasets.load_dataset("TIGER-Lab/WebInstruct-verified")
    train_data = data["train"].filter(lambda example: example["category"] != "Mathematics" and example["answer_type"] not in ["Matrix", "Other"])
    test_data = data["test"].filter(lambda example: example["category"] != "Mathematics" and example["answer_type"] not in ["Matrix", "Other"])
    train_data = train_data.filter(lambda example: is_valid_answer(example) is not None)
    test_data = test_data.filter(lambda example: is_valid_answer(example) is not None)
    train_data = train_data.map(function=process_fn, with_indices=True)
    test_data = test_data.map(function=process_fn, with_indices=True)
    all_data = datasets.concatenate_datasets([train_data, test_data])
    print(f"Total data size: {len(all_data)}")
    return all_data

def load_legalbench():
    prompt_dir = os.path.join(os.path.dirname(__file__), "legalbench_prompts")
    task_names = datasets.get_dataset_config_names("nguha/legalbench")

    all_data = []
    for task_name in tqdm(task_names, desc="Loading legalbench tasks"):
        # Load prompt template
        prompt_path = os.path.join(prompt_dir, f"{task_name}.txt")
        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        # Extract placeholder names from template
        placeholders = re.findall(r"\{\{(\w+)\}\}", prompt_template)

        # Load dataset for this task
        task_data = datasets.load_dataset("nguha/legalbench", task_name)

        for split in task_data:
            for idx, example in enumerate(task_data[split]):
                answer = str(example["answer"])

                # Fill in template placeholders from dataset columns
                prompt = prompt_template
                for ph in placeholders:
                    prompt = prompt.replace("{{" + ph + "}}", str(example[ph]))

                data = {
                    "data_source": "legalbench",
                    "prompt": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "env_class": "mix_general",
                    "reward_spec": {
                        "method": "legalbench",
                        "ground_truth": answer,
                    },
                    "extra_info": {
                        "index": idx,
                        "answer": answer,
                        "detailed_source": f"legalbench-{task_name}",
                    }
                }
                all_data.append(data)

    all_data = datasets.Dataset.from_list(all_data)
    print(f"Total data size: {len(all_data)}")
    return all_data

def load_medqa():
    # make sure to clone https://huggingface.co/datasets/bigbio/med_qa and unzip data_clean.zip
    def process_fn(example, idx):
        question = example.pop("question")
        answer = str(example.pop("answer"))
        choices = example.pop("options")
        meta_info = example.pop("meta_info")
        
        if isinstance(choices, dict):
            choices = [choices[key] for key in sorted(choices.keys())]
        
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = MAQ_TEMPLATE.format(question=question, choices=choices_str)
        
        data = {
            "data_source": f"med_qa",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "env_class": "mix_general",
            "reward_spec": {
                "method": "mqa",
                "ground_truth": answer,
            },
            "extra_info": {
                "index": idx,
                "answer": answer,
                "question": question,
                "detailed_source": f"med_qa-{meta_info}"
            }
        }
        
        return data
    
    json_data_paths = [
        "med_qa/data_clean/questions/Mainland/chinese_qbank.jsonl",
        "med_qa/data_clean/questions/Taiwan/taiwanese_qbank.jsonl",
        "med_qa/data_clean/questions/US/US_qbank.jsonl",
    ]
    
    all_datasets = []
    for json_data_path in json_data_paths:
        with open(json_data_path, "r") as f:
            data = []
            for line in f:
                example = json.loads(line)
                assert isinstance(example, dict), f"Invalid example format in {json_data_path}: {example}"
                # Skip examples with non-string values in options (e.g. nan)
                options = example.get("options", {})
                if isinstance(options, list):
                    if any(not isinstance(x, str) for x in options):
                        continue
                elif isinstance(options, dict):
                    if any(not isinstance(v, str) for v in options.values()):
                        continue
                data.append(example)
            dataset = datasets.Dataset.from_list(data)
            processed_dataset = dataset.map(function=process_fn, with_indices=True)
            all_datasets.append(processed_dataset)
    all_datasets = datasets.concatenate_datasets(all_datasets)
    print(f"Total data size: {len(all_datasets)}")
    return all_datasets
    
def load_ceval():
    def process_fn(example, idx):
        
        ceval_id = example.pop("id")
        question = example.pop("question")
        answer = example.pop("answer")
        choice_a = example.pop("A")
        choice_b = example.pop("B")
        choice_c = example.pop("C")
        choice_d = example.pop("D")
        choices = [choice_a, choice_b, choice_c, choice_d]
        
        
        assert len(example) == 1, example
        
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = MAQ_TEMPLATE.format(question=question, choices=choices_str)
        
        data = {
            "data_source": f"ceval",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "env_class": "mix_general",
            "reward_spec": {
                "method": "mqa",
                "ground_truth": answer,
            },
            "extra_info": {
                "index": idx,
                "answer": answer,
                "question": question,
                "detailed_source": f"ceval-{ceval_id}"
            }
        }
        
        return data
    
    configs = datasets.get_dataset_config_names("ceval/ceval-exam")
    all_data = []
    for config in configs:
        data = datasets.load_dataset("ceval/ceval-exam", config)
        for split in data.keys():
            split_data = data[split].map(function=process_fn, with_indices=True)
            all_data.append(split_data)
    all_data = datasets.concatenate_datasets(all_data)
    print(f"Total data size: {len(all_data)}")
    return all_data

def load_arc():
    
    def process_fn(example, idx):
        question = example.pop("question")
        choices = example.pop("choices")
        answer = example.pop("answerKey")
        arc_id = example.pop("id")
        
        assert isinstance(choices, dict), choices
        assert len(choices) == 2, choices
        assert "text" in choices, choices
        assert "label" in choices, choices
        assert len(choices["text"]) == len(choices["label"]), choices
        
        choices_str = "\n".join([f"{choices['label'][i]}. {choices['text'][i]}" for i in range(len(choices["text"]))])
        prompt = MAQ_TEMPLATE.format(question=question, choices=choices_str)
        
        data = {
            "data_source": f"arc",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "env_class": "mix_general",
            "reward_spec": {
                "method": "mqa",
                "ground_truth": answer,
            },
            "extra_info": {
                "index": idx,
                "answer": answer,
                "question": question,
                "detailed_source": f"arc-{arc_id}" 
            }
        }
        return data

    data_challenge = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")
    data_easy = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy")
    all_data = []
    for split in data_challenge.keys():
        split_data_challenge = data_challenge[split].map(function=process_fn, with_indices=True)
        split_data_easy = data_easy[split].map(function=process_fn, with_indices=True)
        all_data.append(split_data_challenge)
        all_data.append(split_data_easy)
    all_data = datasets.concatenate_datasets(all_data)
    print(f"Total data size: {len(all_data)}")
    return all_data
    
def load_logiqa():
    
    def process_fn(example, idx):
        context = example.pop("context")
        question = example.pop("query")
        choices = example.pop("options")
        answer = example.pop("correct_option")
        
        assert len(choices) == 4, choices
        assert answer in [0, 1, 2, 3], answer
        
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = LOGIQA_TEMPLATE.format(context=context, question=question, choices=choices_str)
        answer_key = chr(65 + answer)
        
        data = {
            "data_source": f"logiqa",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "env_class": "mix_general",
            "reward_spec": {
                "method": "mqa",
                "ground_truth": answer_key,
            },
            "extra_info": {
                "index": idx,
                "answer": answer_key,
                "question": question,
                "context": context,
                "detailed_source": f"logiqa"
            }
        }
        
        return data
    
    data = datasets.load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")
    all_data = []
    for split in data.keys():
        split_data = data[split].map(function=process_fn, with_indices=True)
        all_data.append(split_data)
    all_data = datasets.concatenate_datasets(all_data)
    print(f"Total data size: {len(all_data)}")
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="mix_general_dataset.parquet", help="Path to save the processed dataset")
    args = parser.parse_args()
    
    logiqa_data = load_logiqa()
    arc_data = load_arc()
    mmlu_data = load_mmlu()
    webinstruct_data = load_webinstruct()
    legalbench_data = load_legalbench()
    medqa_data = load_medqa()
    ceval_data = load_ceval()
    all_data = datasets.concatenate_datasets([mmlu_data, webinstruct_data, legalbench_data, medqa_data, ceval_data, logiqa_data, arc_data])
    print(f"Total data size: {len(all_data)}")
    all_data.to_parquet(args.output_path)
    