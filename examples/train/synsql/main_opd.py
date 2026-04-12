"""Multi-turn On-Policy Distillation trainer for text-to-SQL.

Extends the standard GRPO trainer for multi-turn (step-wise) SQL environments.
Each turn of the agent loop (tool call → observation → next response) becomes an
independent training sample. Rewards are replaced with per-token KL divergence
from a teacher model, so intermediate turns' KL signal drives learning—not just
the final SQL correctness reward.

Required config:
    generator.step_wise_trajectories: true
    generator.use_conversation_multi_turn: true
    generator.append_eos_token_after_stop_str_in_multi_turn: true
    generator.max_turns: 6
    generator.sampling_params.stop: '["</tool_call>", "</solution>"]'
    trainer.algorithm.advantage_estimator: no_op
    trainer.algorithm.use_kl_in_reward: true
    trainer.ref.model.path: <teacher_model_path>
    environment.env_class: text2sql
    environment.skyrl_gym.text2sql.db_path: <db_path>

Usage:
    uv run --extra fsdp -m examples.train.synsql.main_opd \\
        trainer.policy.model.path=Qwen/Qwen2.5-Coder-7B-Instruct \\
        trainer.ref.model.path=Qwen/Qwen2.5-Coder-7B-Instruct \\
        <config overrides ...>
"""

import sys

import numpy as np
import torch
import ray
from loguru import logger
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.backends.skyrl_train.utils import ppo_utils
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from dotenv import load_dotenv

load_dotenv()


class MultiTurnOPDTrainer(RayPPOTrainer):
    """On-Policy Distillation trainer for multi-turn (step-wise) environments.

    Overrides:
    - apply_reward_kl_penalty: sets rewards to per-token KL penalty.
    - compute_advantages_and_returns: computes advantages on ALL steps
      (not just the final step), so intermediate turns' KL signal drives
      learning.
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Sets rewards to per-token KL divergence from the teacher."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        rewards = -(action_log_probs - teacher_action_log_probs) * loss_masks_all
        data["rewards"] = rewards
        return data

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Compute per-step advantages for multi-turn distillation.

        The base class (step-wise mode) computes advantages only on the final
        step of each trajectory, then broadcasts to all intermediate steps.
        This discards the KL reward signal from intermediate turns.

        For distillation, every step's KL reward is informative, so we treat
        each step as an independent sample for advantage computation.
        """
        token_level_rewards = data["rewards"]

        advantages, returns = ppo_utils.compute_advantages_and_returns(
            token_level_rewards=token_level_rewards,
            response_mask=data["response_mask"],
            index=data.metadata["uids"],
            adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
            config=self.cfg.trainer.algorithm,
            values=data["values"],
            gamma=self.cfg.trainer.algorithm.gamma,
            lambd=self.cfg.trainer.algorithm.lambd,
            grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
        )

        data["returns"] = returns
        data["advantages"] = advantages

        # Metrics
        pad_size = data.metadata.get("pad_size", 0)
        num_samples = len(token_level_rewards)

        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        avg_rewards: float = return_sums.mean().item()

        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")

        valid_advantages = torch.masked_select(
            data["advantages"][: num_samples - pad_size, ...],
            data["response_mask"][: num_samples - pad_size].bool(),
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_final_rewards": avg_rewards,
                "avg_response_length": avg_response_length,
                "avg_advantages": avg_advantages,
                "avg_advantages_abs": avg_advantages_abs,
            }
        )

        logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": avg_rewards,
                "loss/avg_raw_advantages": avg_advantages,
                "loss/avg_raw_advantages_abs": avg_advantages_abs,
            }
        )

        return data


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


class MultiTurnOPDExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return MultiTurnOPDTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # text2sql is already registered in skyrl_gym.envs.__init__, but we
    # re-register here for explicitness (consistent with other examples).
    from skyrl_gym.envs import register

    register(
        id="text2sql",
        entry_point="skyrl_gym.envs.sql.env:SQLEnv",
    )

    exp = MultiTurnOPDExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
