"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to
validate answers when necessary.
"""

from typing import Any, Dict
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from rl_gen.mixed_domain.utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from skyrl_gym.envs.lcb.livecodebench import compute_score
import json

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

class CodeMathMixedEnv(BaseTextEnv):
    """
    Environment for Math
    """

    def __init__(
        self, 
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.tests = None
        self.method = extras["reward_spec"]["method"]
        if self.method == "code":
            self.tests = json.loads(self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.method == "code":
            step_output = self._step_code(action)
            return step_output
        elif self.method == "math":
            step_output = self._step_math(action)
            return step_output
        elif self.method == "mlvr":
            step_output = self._step_mlvr(action)
            return step_output
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _step_code(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        try:
            parsed_code, reward = compute_score(action, self.tests)
        except Exception as e:
            print("Error during code execution:", e)
            parsed_code, reward = None, 0.0

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={"parsed_code": parsed_code})

    def _step_mlvr(self, action: str) -> BaseTextEnvStepOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            action: The agent's response/solution

        Returns:
            BaseTextEnvStepOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        model_response = action

        # Handle None or empty response
        if model_response is None or model_response == "":
            # print("DEBUG: Empty or None response")
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Empty or None response"})

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            # print("DEBUG: Fail to extract answers.")
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Fail to extract answers."})
        

        # Process the ground truth(s)
        ground_truth = self.ground_truth
        if ground_truth is None:
            # print("DEBUG: Empty or None ground truth")
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Empty or None ground truth"})

        # Convert single answer to list for uniform processing
        if isinstance(ground_truth, str | float | int):
            ground_truth = [ground_truth]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truth:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            # print("DEBUG: Empty or None post-processed ground truth. Ground truth was:", ground_truth)
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Empty or None post-processed ground truth"})

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth) or model_answer.strip() == ground_truth.strip() or ground_truth.strip() in model_answer.strip()
            if is_correct:
                # print(f"DEBUG: Correct answer found: {model_answer}; Ground truth was: {ground_truth}")
                return BaseTextEnvStepOutput(observation=[], reward=1, done=True, metadata={"reason": "Correct answer"})

        # print(f"DEBUG: No correct answer found. Model answer: {model_answer}; Ground truths were: {processed_ground_truths}")
        return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Incorrect answer", "answer": model_answer, "ground_truth": processed_ground_truths})


    def _step_math(self, action: str) -> BaseTextEnvStepOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            action: The agent's response/solution

        Returns:
            BaseTextEnvStepOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        model_response = action

        # Handle None or empty response
        if model_response is None or model_response == "":
            # print("DEBUG: Empty or None response")
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Empty or None response"})

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            # print("DEBUG: Fail to extract answers.")
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Fail to extract answers."})
        

        # Process the ground truth(s)
        ground_truth = self.ground_truth
        if ground_truth is None:
            # print("DEBUG: Empty or None ground truth")
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Empty or None ground truth"})

        # Convert single answer to list for uniform processing
        if isinstance(ground_truth, str | float | int):
            ground_truth = [ground_truth]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truth:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            # print("DEBUG: Empty or None post-processed ground truth. Ground truth was:", ground_truth)
            return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Empty or None post-processed ground truth"})

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth) or model_answer.strip() == ground_truth.strip()
            if is_correct:
                # print(f"DEBUG: Correct answer found: {model_answer}; Ground truth was: {ground_truth}")
                return BaseTextEnvStepOutput(observation=[], reward=1, done=True, metadata={"reason": "Correct answer"})

        # print(f"DEBUG: No correct answer found. Model answer: {model_answer}; Ground truths were: {processed_ground_truths}")
        return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Incorrect answer", "answer": model_answer, "ground_truth": processed_ground_truths})

