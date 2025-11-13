import json

incorrect_answer_file = "../../../qwen2.5_math_7b_incorrect.json"
groundtruth_answer_file = "../../../all_deepscaler_dataset.json"

with open(incorrect_answer_file) as f:
    incorrect_data = json.load(f)

with open(groundtruth_answer_file) as f:
    groundtruth_data = json.load(f)

import glob

gt_gpt_files = glob.glob("batch_jobs/gpt_graded_results_part_*_gt-gpt.jsonl")
gpt_files = glob.glob("batch_jobs/gpt_graded_results_part_*_gpt.jsonl")
gt_files = glob.glob("batch_jobs/gpt_graded_results_part_*_gt.jsonl")

gpt_results = {}
gpt_answers = {}
for gpt_file in gpt_files:
    with open(gpt_file) as f:
        data = [json.loads(line) for line in f.readlines()]
        for item in data:
            question = item["question"]
            incorrect_answer = item["incorrect_answer"]
            gpt_results[(question, incorrect_answer)] = item["match"]
            gpt_answers[question] = item.get("gpt_answer", None)

gt_results = {}
for gt_file in gt_files:
    with open(gt_file) as f:
        data = [json.loads(line) for line in f.readlines()]
        for item in data:
            question = item["question"]
            incorrect_answer = item["incorrect_answer"]
            gt_results[(question, incorrect_answer)] = item["match"]

n_correct_by_gt = 0
n_correct_by_gpt = 0
n_correct = 0

truly_incorrect_data = []
maybe_correct_data = []
questions = []

for item in incorrect_data:
    question = item['prompt']
    incorrect_answer = item['answer']
    key = (question, incorrect_answer)
    gpt_match = gpt_results.get(key, None)
    gt_match = gt_results.get(key, None)
    groundtruth_answer = groundtruth_data.get(question, None)
    item['groundtruth_answer'] = groundtruth_answer
    gpt_answer = gpt_answers.get(question, None)
    item['gpt_answer'] = gpt_answer
    if gt_match:
        n_correct_by_gt += 1
    if gpt_match:
        n_correct_by_gpt += 1
    if gt_match or gpt_match:
        n_correct += 1
        maybe_correct_data.append(item)
    else:
        truly_incorrect_data.append(item)
    questions.append(question)

print(f"Number of incorrect answers marked correct by ground truth grading: {n_correct_by_gt/len(incorrect_data):.2%}, {n_correct_by_gt} out of {len(incorrect_data)}")    
print(f"Number of incorrect answers marked correct by GPT grading: {n_correct_by_gpt/len(incorrect_data):.2%}, {n_correct_by_gpt} out of {len(incorrect_data)}")    
print(f"Number of incorrect answers marked correct by either ground truth or GPT grading: {n_correct/len(incorrect_data):.2%}, {n_correct} out of {len(incorrect_data)}")
print(f"Number of unique questions in incorrect answers: {len(set(questions))} out of {len(incorrect_data)}")

# with open("truly_incorrect_answers_qwen2.5_math_7b.json", "w") as f:
#     json.dump(truly_incorrect_data, f, indent=4)

# with open("maybe_correct_answers_qwen2.5_math_7b.json", "w") as f:
#     json.dump(maybe_correct_data, f, indent=4)