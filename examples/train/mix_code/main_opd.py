"""
Entry point for mix_code on-policy distillation training.

Registers the MixCodeEnv environment and launches OPD training with a custom
trainer that sets rewards to the KL penalty against a teacher model.

Usage:
    python -m examples.train.mix_code.main_opd <config overrides ...>
"""

import sys

import torch
import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl_gym.envs import register
from dotenv import load_dotenv

load_dotenv()


class OnPolicyDistillationTrainer(RayPPOTrainer):
    """Custom trainer for On-Policy Distillation.

    Overrides ``apply_reward_kl_penalty`` to set the rewards to the
    KL penalty between the student and teacher action log-probs.
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        rewards = -(action_log_probs - teacher_action_log_probs) * loss_masks_all
        data["rewards"] = rewards
        return data


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


class OnPolicyDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="mix_code",
        entry_point="examples.train.mix_code.env:MixCodeEnv",
    )

    exp = OnPolicyDistillationExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
