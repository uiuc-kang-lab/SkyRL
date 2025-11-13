import csv
import json
import glob
import openai
import os
import argparse

def dump_batch_jobs():
    incorrect_answers = glob.glob("../../../qwen2.5_math_7b_incorrect.json")

    groundtruth_data = {}
    with open("../../../all_deepscaler_dataset.json") as f:
        groundtruth_data = json.load(f)

    with open("batch_jobs/gpt-5-pro_answers_output_0.jsonl") as f:
        existing_outputs = [json.loads(line)['custom_id'] for line in f]


    running_cost = 0.0

    batches = {}

    for incorrect_answer_file in incorrect_answers:
        with open(incorrect_answer_file) as f:
            incorrect_data = json.load(f)

        n_incorrect_gt = 0
        n_correct_answers_gt = 0
        n_correct_answers_gpt = 0
        n_missed = 0
        n_processed = 0
        curroped = 0

        for idx, item in enumerate(incorrect_data):
            if f"spurious-grading-{idx}" in existing_outputs:
                continue

            batch_id = idx // 40000 + 1

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

            prompt = f"Question: {question}\nIf the question have multiple distinct and correct answers, provide all the answers. Write the answers in LaTeX format. Do not include any explanations.\nAnswer(s):"

            batch_item = {
                "custom_id": f"spurious-grading-{idx}", 
                "method": "POST", 
                "url": "/v1/responses", 
                "body": {
                    "model": "gpt-5-pro", 
                    "input": prompt
                }
            }
            if batch_id not in batches:
                batches[batch_id] = []
            batches[batch_id].append(batch_item)

        for batch_id, batch in batches.items():
            with open(f"./batch_jobs/obtain_gpt-5-pro_answers_{batch_id}.jsonl", "w") as f:
                for batch_item in batch:
                    f.write(json.dumps(batch_item) + "\n")

def upload_batch_jobs(prefix="obtain_gpt-5-pro_answers"):
    client = openai.OpenAI()
    batch_job_files = glob.glob(f"./batch_jobs/{prefix}_1.jsonl")
    batch_input_files = []
    for batch_job_file in batch_job_files:
        batch_input_file = client.files.create(
            file=open(batch_job_file, "rb"),
            purpose="batch"
        )
        batch_input_files.append(batch_input_file)
    with open(f"./batch_jobs/{prefix}_file_ids_1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_job_file", "batch_input_file_id"])
        for batch_job_file, batch_input_file in zip(batch_job_files, batch_input_files):
            writer.writerow([batch_job_file, batch_input_file.id])

def create_batch(prefix="obtain_gpt-5-pro_answers"):
    client = openai.OpenAI()
    batch_job_ids = []
    batch_input_file_ids = []
    batch_job_files = []
    with open(f"./batch_jobs/{prefix}_file_ids_1.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            batch_job_file = row[0]
            batch_input_file_id = row[1]
            batch_input_file_ids.append(batch_input_file_id)
            batch_job_files.append(batch_job_file)
    print(batch_input_file_ids, batch_job_files)
    for file_id, batch_job_file in zip(batch_input_file_ids, batch_job_files):
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": "Answer deepscaler math questions using gpt-5-pro for spurious answer grading",
                "batch_job_file_id": file_id,
                "batch_job_file_local": batch_job_file
            }
        )
        batch_job_id = batch_job.id
        batch_job_ids.append(batch_job_id)
    
    with open(f"./batch_jobs/{prefix}_batch_job_ids_1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_job_file", "batch_job_id"])
        for batch_job_file, batch_job_id in zip(batch_job_files, batch_job_ids):
            writer.writerow([batch_job_file, batch_job_id])

def get_status_and_download(prefix="obtain_gpt-5-pro_answers"):
    client = openai.OpenAI()
    batch_job_files = []
    batch_job_ids = []
    with open(f"./batch_jobs/{prefix}_batch_job_ids.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            batch_job_file = row[0]
            batch_job_id = row[1]
            batch_job_files.append(batch_job_file)
            batch_job_ids.append(batch_job_id)
    for (batch_job_file, batch_job_id) in zip(batch_job_files, batch_job_ids):
        batch_job = client.batches.retrieve(batch_job_id)
        status = batch_job.status
        print(f"Batch job file: {batch_job_file}, Batch job ID: {batch_job_id}, Status: {status}")
        if status in ["completed", "expired"] and not os.path.exists(batch_job_file.replace("obtain_gpt-5-pro_answers", "gpt-5-pro_answers_output")):
            output_file_id = batch_job.output_file_id
            output_file_content = client.files.content(output_file_id)
            with open(batch_job_file.replace("obtain_gpt-5-pro_answers", "gpt-5-pro_answers_output"), "w") as f:
                f.write(output_file_content.text)


def align_results_with_questions():
    result_files = glob.glob("batch_jobs/gpt-5-pro_answers_output_*.jsonl")
    for result_file in result_files:
        with open(result_file, "r") as f:
            results = [json.loads(line) for line in f]
            results_dict = {item['custom_id']: item['response']['body']['output'][-1]['content'][-1]['text'] for item in results}

    with open("../../../qwen2.5_math_7b_incorrect.json") as f:
        incorrect_data = json.load(f)
    
    for idx, item in enumerate(incorrect_data):
        custom_id = f"spurious-grading-{idx}"
        if custom_id in results_dict:
            incorrect_data[idx]['gpt-5-pro_answer'] = results_dict[custom_id]

    with open("batch_jobs/aligned_gpt-5-pro_incorrect_answers.json", "w") as f:
        json.dump(incorrect_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["dump", "upload", "create", "status", "align"])
    parser.add_argument("--prefix", type=str, default="obtain_gpt-5-pro_answers")
    args = parser.parse_args()

    if args.mode == "dump":
        dump_batch_jobs()
    elif args.mode == "upload":
        upload_batch_jobs(prefix=args.prefix)
    elif args.mode == "create":
        create_batch(prefix=args.prefix)
    elif args.mode == "status":
        get_status_and_download(prefix=args.prefix)
    elif args.mode == "align":
        align_results_with_questions()