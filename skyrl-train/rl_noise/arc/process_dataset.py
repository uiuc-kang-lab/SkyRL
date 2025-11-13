from datasets import load_dataset, concatenate_datasets
import argparse
import random

instruction_following = 'Let\'s think step by step and output your final answer after "####". The answer should be one of A, B, C, D.'

random.seed(42)

def down_datasets():

    challenge_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    easy_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")

    challenge_train_dataset = challenge_dataset["train"]
    easy_train_dataset = easy_dataset["train"]
    challenge_test_dataset = challenge_dataset["test"]
    easy_test_dataset = easy_dataset["test"]

    return challenge_train_dataset, easy_train_dataset, challenge_test_dataset, easy_test_dataset

def construct_full_question(question: str, choices: list, label: list):
    assert len(choices) == len(label)
    full_question = f"{question}\nAnswer Choices:\n"
    for l, c in zip(label, choices):
        full_question += f"{l}. {c}\n"
    full_question += "Please select the correct answer."
    return full_question


def process_dataset(dataset, split, instruction_following="", data_source="arc", noise_level=0):

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            ground_truth = example.pop("answerKey")
            question_id = example.pop("id")
            choices = example.pop("choices")
            labels = choices["label"]
            choices = choices["text"]
            if noise_level > 0:
                if random.random() < noise_level:
                    wrong_choices = [c for c in labels if c != ground_truth]
                    if wrong_choices:
                        ground_truth = random.choice(wrong_choices)
            full_question = construct_full_question(question_raw, choices, labels)
            question = full_question + " " + instruction_following
            prompt = [
                {
                    "role": "user",
                    "content": question,
                }
            ]
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "env_class": "arc",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": ground_truth,
                    "id": question_id,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn(split), with_indices=True)
    return dataset

if __name__ == "__main__":
    challenge_train_dataset, easy_train_dataset, challenge_test_dataset, easy_test_dataset = down_datasets()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/arc")
    parser.add_argument("--noise_level", type=float, default=0, help="the noise level to add to the dataset")
    args = parser.parse_args()

    challenge_train_dataset = process_dataset(challenge_train_dataset, split="train", instruction_following=instruction_following, data_source="challenge-arc", noise_level=args.noise_level)
    easy_train_dataset = process_dataset(easy_train_dataset, split="train", instruction_following=instruction_following, data_source="easy-arc", noise_level=args.noise_level)
    challenge_test_dataset = process_dataset(challenge_test_dataset, split="test", instruction_following=instruction_following, data_source="challenge-arc")
    easy_test_dataset = process_dataset(easy_test_dataset, split="test", instruction_following=instruction_following, data_source="easy-arc")

    train_dataset = concatenate_datasets([challenge_train_dataset, easy_train_dataset])
    val_dataset = concatenate_datasets([challenge_test_dataset, easy_test_dataset])
    print(train_dataset)
    print(val_dataset)
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)

    if args.noise_level > 0:
        train_dataset.to_parquet(f"{args.output_dir}/train_noise_{args.noise_level}.parquet")
    else:
        train_dataset.to_parquet(f"{args.output_dir}/train.parquet")
    val_dataset.to_parquet(f"{args.output_dir}/val.parquet")