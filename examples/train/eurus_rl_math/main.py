"""
uv run --isolated --extra vllm -m examples.multiply.main_multiply
"""

import sys

import ray
import hydra
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # Register the multiply environment inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="eurus_rl_math",
        entry_point="examples.eurus_rl_math.env:MathEnv",
    )

    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()