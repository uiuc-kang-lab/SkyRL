from utils import grade_answer_mathd, grade_answer_sympy
from math_verify import parse, verify, LatexExtractionConfig
from math_verify.parser import NormalizationConfig
from dataclasses import field
import csv
import json
import glob
from tqdm import tqdm
import openai
import os
import argparse

prompt_template = """You are a highly intelligent and accurate math grader. Given a math question, a ground truth answer, and a student's answer, determine whether the student's answer is correct. You must follow the following guidelines:
1. If the question has multiple distinct but correct answers, the student only needs to provide one of them to be considered correct. For example, if the ground truth answer is "(k in {{12, -12}})" and the student's answer is "12", the student's answer should be considered correct.
2. Ignore any format mistakes and only grade the mathematical meaning of the answer. For example, if the ground truth answer is "(11)_6" and the student's answer is "11", the student's answer should be considered correct. Another example is if the ground truth answer is latex formatted and the student's answer is not latex formatted but is mathematically equivalent, the student's answer should be considered correct. Vice versa is also true.
3. If the student's answer is mathematically equivalent to the ground truth answer, it should be considered correct.
4. If the student's answer uses fractions, decimals, or different representations that are mathematically equivalent to the ground truth answer, it should be considered correct. For example, if the ground truth answer is "4\\sqrt{{2}}" and the student's answer is "5.656854249492381", the student's answer should be considered correct.
5. If the question is an Yes/No question and the ground truth answer is an exact solution, the student's answer is correct as long as they answers "Yes" or anything equivalent to "Yes". For example, for the question "Does there exist a fraction equivalent to $\\frac{{7}}{{13}}$ such that the difference between the denominator and the numerator is 24?", the ground truth answer is "\\frac{{28}}{{52}}" and the student's answer is "Yes", the student's answer should be considered correct.
6. If the student's answer is a simplification of the ground truth answer, it should be considered correct. For example, if the ground truth answer is "2 + 2" and the student's answer is "4", the student's answer should be considered correct. Vice versa is also true.
7. If the student's answer is mathematically inequivalent to the ground truth answer, it should be considered incorrect. For example, if the ground truth answer is "5" and the student's answer is "-5", the student's answer should be considered incorrect.

Question: {question}
Ground Truth: {ground_truth}
Student Answer: {student_answer}
Is the student's answer correct? Answer 'Yes' or 'No'.
Answer: """


def grade_answer_gpt(ground_truth: str, gpt_5_pro_answer: str, incorrect_answer: str, question: str, mode="gt-gpt") -> bool:
    
    gpt_5_input_tokens = 0
    gpt_5_output_tokens = 0

    final_response = None


    if mode == "gt-gpt":
        if gpt_5_pro_answer is not None:
            # step 1: ask gpt-5 to grade the ground truth answer as if the ground truth is the student's answer
            prompt_1 = prompt_template.format(
                question=question,
                ground_truth=gpt_5_pro_answer,
                student_answer=ground_truth,
            )
            response_1 = openai.responses.create(
                model="gpt-5",
                input=prompt_1,
            )
            gt_matches_gpt = response_1.output_text.strip('.').lower() == "yes"
            gpt_5_input_tokens += response_1.usage.input_tokens
            gpt_5_output_tokens += response_1.usage.output_tokens
            cost = 1.5 * gpt_5_input_tokens / 1e6 + 10 * gpt_5_output_tokens / 1e6
            final_response = response_1.output_text
        else:
            gt_matches_gpt = None
            cost = 0
        return gt_matches_gpt, cost, final_response

    elif mode == "gpt":
        if gpt_5_pro_answer is not None:
            # step 2: ask gpt-5 to grade the incorrect answer based on the gpt's answer
            prompt_2 = prompt_template.format(
                question=question,
                ground_truth=gpt_5_pro_answer,
                student_answer=incorrect_answer,
            )
            response_2 = openai.responses.create(
                model="gpt-5",
                input=prompt_2,
            )

            incorrect_matches_gpt = response_2.output_text.strip('.').lower() == "yes"
            gpt_5_input_tokens += response_2.usage.input_tokens
            gpt_5_output_tokens += response_2.usage.output_tokens
            cost = 1.5 * gpt_5_input_tokens / 1e6 + 10 * gpt_5_output_tokens / 1e6
            final_response = response_2.output_text
        else:
            incorrect_matches_gpt = None
            cost = 0
        return incorrect_matches_gpt, cost, final_response
        

    # step 4: ask gpt-5 to grade the incorrect answer based on the ground truth 

    elif mode == "gt":
        prompt_4 = prompt_template.format(
            question=question,
            ground_truth=ground_truth,
            student_answer=incorrect_answer,
        )

        response_4 = openai.responses.create(
            model="gpt-5",
            input=prompt_4,
        )

        gt_matches_incorrect = response_4.output_text.strip('.').lower() == "yes"
        gpt_5_input_tokens += response_4.usage.input_tokens
        gpt_5_output_tokens += response_4.usage.output_tokens
        cost = 1.5 * gpt_5_input_tokens / 1e6 + 10 * gpt_5_output_tokens / 1e6
        final_response = response_4.output_text
        return gt_matches_incorrect, cost, final_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["gt-gpt", "gpt", "gt"])
    parser.add_argument("--part_id", type=int, default=0)
    args = parser.parse_args()

    incorrect_answer_file = f"batch_jobs/aligned_gpt-5-pro_incorrect_answer_part_{args.part_id}.json"

    groundtruth_data = {}
    with open("../../../all_deepscaler_dataset.json") as f:
        groundtruth_data = json.load(f)


    running_cost = 0.0

    with open(incorrect_answer_file) as f:
        incorrect_data = json.load(f)

    n_matched = 0
    n_missed = 0
    n_processed = 0
    curroped = 0

    for item in tqdm(incorrect_data):
        n_processed += 1
        question = item['prompt']

        if "answer" not in item:
            curroped += 1
        incorrect_answer = item['answer']
        qid = item['id']

        if question not in groundtruth_data:
            n_missed += 1
            continue

        ground_truth = groundtruth_data[question]
        gpt_answer = item['gpt-5-pro_answer'] if 'gpt-5-pro_answer' in item else None

        # check if this data is already graded
        if os.path.exists(f"gpt_graded_results_part_{args.part_id}_{args.mode}.jsonl"):
            with open(f"gpt_graded_results_part_{args.part_id}_{args.mode}.jsonl") as f:
                already_graded = [json.loads(line) for line in f.readlines()]
                already_graded_questions = set([entry["question"] for entry in already_graded])
                if question in already_graded_questions:
                    continue

        result, cost, response = grade_answer_gpt(ground_truth, gpt_answer, incorrect_answer, question, mode=args.mode)

        if result:
            n_matched += 1

        with open(f"gpt_graded_results_part_{args.part_id}_{args.mode}.jsonl", "a+") as f:
            # question, ground_truth, incorrect_answer, gpt_answer, gt_matches_gpt, incorrect_matches_gpt, gt_matches_incorrect, all_responses
            f.write(json.dumps({
                "question": question,
                "ground_truth": ground_truth,
                "incorrect_answer": incorrect_answer,
                "gpt_answer": gpt_answer,
                "match": result == True,
                "cost": cost,
            }) + "\n")
        running_cost += cost
        if n_processed % 50 == 0:
            print(f"Processed {n_processed} samples. Running cost: ${running_cost:.4f}. Result Summary:\nMatched answers: {n_matched} ({n_matched/n_processed:.2%}%)")