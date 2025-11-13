import json
from llm_judge_env import MLVRLLMJudgeEnv
from omegaconf import DictConfig
from tqdm import tqdm
import sys

n_batch = 32
batch_id = int(sys.argv[1])

with open("/data/daniel_kang_group/rl_noise/exports/mlvr_mlvr-test/dumped_evals/global_step_0_evals/None.jsonl") as f:
    lines = f.readlines()
    n_data_per_batch = len(lines) // n_batch + (1 if len(lines) % n_batch != 0 else 0)
    batch = lines[batch_id * n_data_per_batch: (batch_id + 1) * n_data_per_batch] if batch_id != n_batch - 1 else lines[batch_id * n_data_per_batch:]
    for line in tqdm(batch):
        data = json.loads(line)
        question = data["env_extras"]["reward_spec"]["question"]
        ground_truth = data["env_extras"]["reward_spec"]["ground_truth"]
        response = data["output_response"]
        config = DictConfig({
            "model": "gpt-5-mini-2025-08-07",
            "base_url": None,
        })
        env = MLVRLLMJudgeEnv(
            env_config=config,
            extras={
                "reward_spec": {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            }
        )
        if response.find("Therefore, the answer is") != -1:
            answer = response.split("Therefore, the answer is")[-1].split("<|im_end|>")[0].strip()
            if answer.startswith(":") or answer.startswith(","):
                answer = answer[1:].strip()
        reward, metadata = env._get_reward(response)
        # print("="*100 + "\n" + f"Q: {question}\nA: {answer}\nGT: {ground_truth}\nReward: {reward}")
        with open(f"/data/daniel_kang_group/rl_noise/exports/mlvr_mlvr-test/judge_results_{batch_id}.jsonl", "a+") as f:
            f.write(json.dumps({
                "question": question,
                "ground_truth": ground_truth,
                "response": response,
                "answer": answer,
                "reward": reward,
                "metadata": metadata,
            }) + "\n")
        print()


        