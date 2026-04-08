"""
This module contains the RewardMathFn class, which evaluates general answers
and assigns rewards based on their correctness.
"""

import json
from typing import Any, Dict
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from examples.train.eurus_rl_math.utils import extract_answer, grade_answer_mathd, grade_answer_sympy, grade_answer_math_verify

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def grade(ground_truth: str, model_answer: str, method: str, timeout: int | None = None) -> bool:
    assert method in ["mqa", "legalbench"] or method in [f"webinstruct-{answer_type}" for answer_type in ["Float", "Integer", "String", "Multiple Choice", "List", "Percentage", "Expression", "Boolean", "Fraction"]], f"Unsupported grading method: {method}"
    
    if method in ["mqa", "webinstruct-Multiple Choice"]:
        return ground_truth.strip().lower() == model_answer.strip().lower() or grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth) or grade_answer_math_verify(model_answer, ground_truth, timeout=timeout)
    elif method == "legalbench":
        return ground_truth.strip().lower() == model_answer.strip().lower()
    elif method.startswith("webinstruct-"):
        answer_type = method.split("-")[1]
        if answer_type in ["Float", "Integer", "Percentage", "Fraction", "Expression"]:
            return grade_answer_mathd(model_answer, ground_truth) or grade_answer_math_verify(model_answer, ground_truth, timeout=timeout) or grade_answer_sympy(model_answer, ground_truth)
        elif answer_type == "String":
            return ground_truth.strip().lower() == model_answer.strip().lower()
        elif answer_type == "List":
            # For list answers, we will split the answers by comma and check if they match as sets (ignoring order and whitespace)
            model_answers = set([ans.strip().lower() for ans in model_answer.split(",")])
            ground_truths = set([ans.strip().lower() for ans in json.loads(ground_truth)])
            return model_answers == ground_truths
        elif answer_type == "Boolean":
            positive_answers = {"true", "yes", "1"}
            negative_answers = {"false", "no", "0"}
            model_answer_normalized = model_answer.strip().lower()
            ground_truth_normalized = ground_truth.strip().lower()
            if ground_truth_normalized in positive_answers:
                return model_answer_normalized in positive_answers
            elif ground_truth_normalized in negative_answers:
                return model_answer_normalized in negative_answers
            else:
                # If ground truth is not a recognized boolean value, fall back to exact match
                return model_answer_normalized == ground_truth_normalized
        else:
            raise ValueError(f"Unsupported answer type for webinstruct: {answer_type}")
    else:
        raise ValueError(f"Unsupported grading method: {method}")

class GeneralEnv(BaseTextEnv):
    """
    Environment for General Question Answering tasks. The reward is calculated based on the correctness of the agent's answer compared to the ground truth. The environment supports both rule-based and format-based rewards, as specified in the reward_spec field of the extras dictionary during initialization.
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
        math_verify_timeout: int | None = None,
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.math_verify_timeout = math_verify_timeout
        self.method = extras["reward_spec"]["method"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
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
            is_correct = grade(ground_truth, model_answer, self.method, timeout=self.math_verify_timeout)
            if is_correct:
                # print(f"DEBUG: Correct answer found: {model_answer}; Ground truth was: {ground_truth}")
                return BaseTextEnvStepOutput(observation=[], reward=1, done=True, metadata={"reason": "Correct answer"})

        # print(f"DEBUG: No correct answer found. Model answer: {model_answer}; Ground truths were: {processed_ground_truths}")
        return BaseTextEnvStepOutput(observation=[], reward=0, done=True, metadata={"reason": "Incorrect answer", "answer": model_answer, "ground_truth": processed_ground_truths})
