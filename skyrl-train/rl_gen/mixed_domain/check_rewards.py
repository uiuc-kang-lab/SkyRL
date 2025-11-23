import glob
import json
import random
import os

math_evals = glob.glob("/data/daniel_kang_group/rl_noise/exports/rl_gen-mixed-1.5b-8k/dumped_evals/**/hard_math.jsonl")
code_evals = glob.glob("/data/daniel_kang_group/rl_noise/exports/rl_gen-mixed-1.5b-8k/dumped_evals/**/lcb.jsonl")
mlvr_evals = glob.glob("/data/daniel_kang_group/rl_noise/exports/rl_gen-mixed-1.5b-8k/dumped_evals/**/mlvr.jsonl")

# STEP 1 merge all data
math_eval_data = []
code_eval_data = []
mlvr_eval_data = []

for math_eval_file in math_evals:
    with open(math_eval_file, "r") as f:
        for line in f:
            data = json.loads(line)
            math_eval_data.append(data)

for code_eval_file in code_evals:
    with open(code_eval_file, "r") as f:
        for line in f:
            data = json.loads(line)
            code_eval_data.append(data)

for mlvr_eval_file in mlvr_evals:
    with open(mlvr_eval_file, "r") as f:
        for line in f:
            data = json.loads(line)
            mlvr_eval_data.append(data)

# STEP 2: sample 100 random data points from each
random.seed(42)
sampled_math_eval_data = random.sample(math_eval_data, 100)
sampled_code_eval_data = random.sample(code_eval_data, 100)
sampled_mlvr_eval_data = random.sample(mlvr_eval_data, 100)

# STEP 3: dump eval results to folder
os.makedirs("/data/yuxuan_zhu/SkyRL/rl_gen/math", exist_ok=True)
os.makedirs("/data/yuxuan_zhu/SkyRL/rl_gen/code", exist_ok=True)
os.makedirs("/data/yuxuan_zhu/SkyRL/rl_gen/mlvr", exist_ok=True)

for i, data in enumerate(sampled_math_eval_data):
    os.makedirs(f"/data/yuxuan_zhu/SkyRL/rl_gen/math/{i}", exist_ok=True)
    # dump input_prompt, env_extras.extra_info.answer, output_response
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/math/{i}/question.txt", "w") as f:
        f.write(data["input_prompt"])
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/math/{i}/answers.txt", "w") as f:
        f.write(str(int(data['score'])))
        f.write("\n")
        f.write(data["env_extras"]["extra_info"]["answer"])
        f.write("\n")
        f.write(data["output_response"])

for i, data in enumerate(sampled_code_eval_data):
    print(data)
    exit()
    os.makedirs(f"/data/yuxuan_zhu/SkyRL/rl_gen/code/{i}", exist_ok=True)
    # dump input_prompt, env_extras.extra_info.reward_spec.ground_truth
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/code/{i}/question.txt", "w") as f:
        f.write(data["input_prompt"])
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/code/{i}/answer.txt", "w") as f:
        f.write(str(int(data['score'])))
        f.write("\n")
        f.write(data["output_response"])
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/code/{i}/ground_truth.txt", "w") as f:
        f.write(data["env_extras"]["reward_spec"]["ground_truth"])

for i, data in enumerate(sampled_mlvr_eval_data):
    os.makedirs(f"/data/yuxuan_zhu/SkyRL/rl_gen/mlvr/{i}", exist_ok=True)
    # dump input_prompt, env_extras.extra_info.answer, output_response
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/mlvr/{i}/question.txt", "w") as f:
        f.write(data["input_prompt"])
    with open(f"/data/yuxuan_zhu/SkyRL/rl_gen/mlvr/{i}/answers.txt", "w") as f:
        f.write(str(int(data['score'])))
        f.write("\n")
        f.write(data["env_extras"]["extra_info"]["answer"])
        f.write("\n")
        f.write(data["output_response"])
