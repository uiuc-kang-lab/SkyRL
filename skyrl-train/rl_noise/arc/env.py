from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
from omegaconf import DictConfig


class ARCEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]


    def _get_reward(self, action: str) -> float:
        # print(f"Env: Received action: {action}; Ground truth: {self.ground_truth}")
        if "####" not in action:
            # print("Env: Action does not contain ####, returning 0 reward")
            return 0
        pred_answer = action.split("####")[-1].strip().lower()
        # if pred_answer has more than one alphabetic character, return 0
        if sum(c.isalpha() for c in pred_answer) > 1:
            # print("Env: Action contains more than one alphabetic character, returning 0 reward")
            return 0
        # extract the first alphabetic character
        pred_answer = "".join([c for c in pred_answer if c.isalpha()])
        if len(pred_answer) == 0:
            # print("Env: Action does not contain any alphabetic characters, returning 0 reward")
            return 0
        pred_answer = pred_answer[0]
        if pred_answer == self.ground_truth.lower():
            # print("Env: Action is correct, returning 1 reward")
            return 1
        else:
            # print("Env: Action is incorrect, returning 0 reward")
            return 0



    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)

        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
