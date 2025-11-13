import datasets
import random
import argparse


# https://github.com/LEXam-Benchmark/LEXam/blob/main/lighteval/community_tasks/lexam_mcq_evals.py#L79
LEXAM_MCQ_PROMPT = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., 1, 2, 3, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., 1, 2, 3, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###3### 

Question:
 {question}

Answer:"""

def download_lexam():
    dataset = datasets.load_dataset("LEXam-Benchmark/LEXam", "mcq_4_choices", split="test")
    # shuffle and split into 80% and 20%
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.to_pandas()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.iloc[:train_size]
    eval_dataset = dataset.iloc[train_size:]
    return train_dataset, eval_dataset

def construct_full_question(question, choices):
    for idx, choice in enumerate(choices, 1):
        question += f"\n{idx}. {choice}"

    return question

def process_dataset(dataset, split, instruction_following="", data_source="arc", noise_level=0):

    def make_map_fn(split):
        def process_fn(example, idx):
            ground_truth = str(int(example.pop("gold")) + 1)  # convert to 1-indexed
            course = example.pop("course")
            question_id = example.pop("id")
            choices = eval(example.pop("choices"))
            labels = [str(i) for i in range(1, len(choices) + 1)]
            question = example.pop("question")
            if noise_level > 0:
                if random.random() < noise_level:
                    wrong_choices = [c for c in labels if c != ground_truth]
                    if wrong_choices:
                        ground_truth = random.choice(wrong_choices)
            full_question = construct_full_question(question, choices)
            full_prompt = LEXAM_MCQ_PROMPT.format(course_name=course, question=full_question)
            prompt = [
                {
                    "role": "user",
                    "content": full_prompt,
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
                    "question": question,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn(split), with_indices=True)
    return dataset

if __name__ == "__main__":
    random.seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/lexam", help="the output directory to save the processed dataset")
    parser.add_argument("--noise_level", type=float, default=0, help="the noise level to add to the dataset")
    args = parser.parse_args()

    train_dataset, eval_dataset = download_lexam()
    train_dataset = datasets.Dataset.from_pandas(train_dataset)
    eval_dataset = datasets.Dataset.from_pandas(eval_dataset)

    processed_train_dataset = process_dataset(train_dataset, "train", instruction_following="", data_source="lexam", noise_level=args.noise_level)
    processed_eval_dataset = process_dataset(eval_dataset, "eval", instruction_following="", data_source="lexam", noise_level=0)

    # show one examples from each 
    print("Example from processed train dataset:")
    print(processed_train_dataset[0])
    print("Example from processed eval dataset:")
    print(processed_eval_dataset[0])

    if args.noise_level > 0:
        processed_train_dataset.to_parquet(f"{args.output_dir}/lexam_train_with_wrong_answers_{args.noise_level}.parquet")
    else:
        processed_train_dataset.to_parquet(f"{args.output_dir}/lexam_train.parquet")
    processed_eval_dataset.to_parquet(f"{args.output_dir}/lexam_eval.parquet")

    print("Train dataset size:", len(processed_train_dataset))
    print("Eval dataset size:", len(processed_eval_dataset))