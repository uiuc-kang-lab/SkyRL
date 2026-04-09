"""
Entry point for mix_code training.

Registers the MixCodeEnv environment and launches the PPO training loop.

Usage:
    python -m examples.train.mix_code.main <hydra overrides ...>
"""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register
from dotenv import load_dotenv

load_dotenv()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="mix_code",
        entry_point="examples.train.mix_code.env:MixCodeEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
