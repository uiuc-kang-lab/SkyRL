from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any
from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI
import os
import re

PROMPT = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer and if the response ends with a sentence or a paragraph similar to the following format:

Therefore, the answer is <answer>

The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.

**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**

Your task :
- Compare the final output of the solution process with the reference answer.
- If they **match exactly** , output **1**.
- If they **do not match** , output **0**.
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **0**.

Your output must be strictly **'1'** or **'0'**, with no additional words, punctuation, or explanation.

--**Question:**
{question}
**Solution Process (Final Step Only) :**
{response}
**Reference Answer:**
{reference}
**Output:**
"""


class MLVRLLMJudgeEnv(BaseTextEnv):
    """
    Example implementtion of MLVR environment with LLM as judge.

    Use LLM as judge to evaluate the answer similarity with the ground truth.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.question = extras["reward_spec"]["question"]

        # Set up OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set for Llm as a judge env")
        self.llm_judge_client = OpenAI(base_url=env_config.base_url, api_key=openai_api_key)
        self.model = env_config.model

    def _get_reward(self, action: str):
        # print(f"Evaluating action: {action} against ground truth: {self.ground_truth}")
        message = PROMPT.format(question=self.question, response=action, reference=self.ground_truth)

        try:
            response = self.llm_judge_client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": message}]
            )
            reply = response.choices[0].message.content.strip()

            # match the first number in the reply
            match = re.search(r"([01](?:\.0)?)", reply)
            if match:
                # print(f"LLM Judge reply: {reply}, parsed reward: {match.group(1)}")
                return float(match.group(1)), {"correct": float(match.group(1)), "response": action, "ground_truth": self.ground_truth}

            # Fallback: raw "1" or "0"
            if reply.strip() in {"1", "0"}:
                # print(f"LLM Judge reply: {reply}, parsed reward: {reply.strip()}")
                return float(reply.strip()), {"correct": float(reply.strip()), "response": action, "ground_truth": self.ground_truth}

            # print(f"Unrecognized reward output: {reply}")
            return 0.0, {"error": "unrecognized_output",}

        except Exception as e:
            # print(f"LLM Judge error: {type(e).__name__}: {e}")
            return 0.0, {"error": str(e),}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward, metadata = self._get_reward(action)

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
        # return BaseTextEnvStepOutput(observations=[], reward=0.0, done=done, metadata={})
