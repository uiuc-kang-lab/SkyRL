from utils import grade_answer_mathd, grade_answer_sympy
from math_verify import parse, verify, LatexExtractionConfig
from math_verify.parser import NormalizationConfig
from dataclasses import field
import json
import glob
from tqdm import tqdm

incorrect_answers = glob.glob("../../../qwen2.5_math_7b_incorrect.json")

groundtruth_data = {}
with open("../../../all_deepscaler_dataset.json") as f:
    groundtruth_data = json.load(f)

for incorrect_answer_file in incorrect_answers:
    with open(incorrect_answer_file) as f:
        incorrect_data = json.load(f)

    n_correct = 0
    n_missed = 0
    curroped = 0
    for item in tqdm(incorrect_data):
        question = item['prompt']
        if "answer" not in item:
            curroped += 1
        incorrect_answer = item['answer']
        qid = item['id']
        if question not in groundtruth_data:
            n_missed += 1
            continue
        ground_truth = groundtruth_data[question]
        ground_truth_parsed = parse(ground_truth)
        incorrect_answer_parsed = parse(incorrect_answer)
        normalization_config = field(
            default_factory=lambda: NormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=True,
                nits=True,
                boxed=True,
                equations=True,
            )
        )
        ground_truth_parsed_latex = parse(f"${ground_truth}$", extraction_config=[LatexExtractionConfig(normalization_config=normalization_config)])
        incorrect_answer_parsed_latex = parse(f"${incorrect_answer}$", extraction_config=[LatexExtractionConfig(normalization_config=normalization_config)])

        is_correct = grade_answer_mathd(incorrect_answer, ground_truth) or grade_answer_sympy(incorrect_answer, ground_truth) or verify(ground_truth_parsed, incorrect_answer_parsed, precision=2) or verify(ground_truth_parsed_latex, incorrect_answer_parsed_latex, precision=2)
        if is_correct:
            n_correct += 1
    print(f"File: {incorrect_answer_file}")
    print(f"Number of incorrect answers that are actually correct: {n_correct}/{len(incorrect_data)}={n_correct/len(incorrect_data):.2%}")
    print(f"Number of missed question IDs: {n_missed}")
    print(f"Number of corrupted entries: {curroped}")