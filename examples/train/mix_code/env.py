"""
MixCode environment for SkyRL.

Evaluates model-generated Python code using two methods:
- "inputs":     stdin/stdout testing with input/output pairs (Eurus format)
- "assertions": assertion-based testing with test code strings (KodCode format)

Register this environment with:
    register(id="mix_code", entry_point="examples.train.mix_code.env:MixCodeEnv")
"""

from typing import Any, Dict
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from examples.train.mix_code.code_eval import compute_score


class MixCodeEnv(BaseTextEnv):
    """Single-turn code evaluation environment.

    Expects ``extras`` to contain::

        {
            "reward_spec": {
                "method": "inputs" | "assertions",
                "ground_truth": <str>   # JSON for "inputs", Python code for "assertions"
            }
        }
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        reward_spec = extras["reward_spec"]
        assert "ground_truth" in reward_spec, "ground_truth is required in reward_spec"
        assert "method" in reward_spec, "method is required in reward_spec"

        self.ground_truth: str = reward_spec["ground_truth"]
        self.method: str = reward_spec["method"]
        self.timeout: int = (env_config or {}).get("timeout", 6)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        parsed_code, reward = compute_score(
            action, self.ground_truth, self.method, timeout=self.timeout,
        )
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"parsed_code": parsed_code},
        )
