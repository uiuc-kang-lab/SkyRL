from datasets import load_dataset
import openai
import os
# add the parent directory to the Python path
import sys
from pathlib import Path
import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from rl_noise.math_hard.utils import grade_answer_mathd, grade_answer_sympy, extract_answer

dataset = load_dataset("parquet", data_files="/data/daniel_kang_group/rl_noise/data/deepscaler/deepscaler_train.parquet", split="train").to_pandas()

client = openai.Client()

pbar = tqdm.tqdm(total=len(dataset), desc="Processing questions")

for idx, row in dataset.iterrows():
    pbar.update(1)
    question = row['prompt'][0]['content'].strip()
    ground_truth = row['reward_spec']['ground_truth']

    response = client.responses.create(
        model="gpt-5-nano-2025-08-07",
        input=f"{question}"
    )

    cost = response.usage.input_tokens / 1e6 * 0.05 + response.usage.output_tokens / 1e6 * 0.40

    print(f"Cost for this query: ${cost:.8f}")

    model_response = response.output_text
    model_answer = extract_answer(model_response)
    print("Extracted Model Answer:", model_answer)
    print("Ground Truth:", ground_truth)

    if isinstance(ground_truth, str | float | int):
            ground_truth = [ground_truth]
    processed_ground_truths = []
    for truth in ground_truth:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        print(f"Is correct: {is_correct}")
