"""Runtime adapters connecting the high-level game to a real MJLab environment."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from mjlab_jenga.high_level_env import HighLevelJengaEnv
from mjlab_jenga.low_level_executor import (
    LearnedLowLevelExecutor,
    Policy,
    get_target_command,
)


class SerialHighLevelRuntime:
    """Provide reset and observation operations for one sequential Jenga game."""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        if env.num_envs != 1:
            raise ValueError("The serial high-level runtime requires num_envs=1")
        self.env = env
        self.target_command = get_target_command(env)

    def reset(self) -> None:
        """Reset MuJoCo and restore the nominal move-reference positions."""
        self.env.reset()
        self.target_command.restore_pristine_reference()

    def observation(self) -> torch.Tensor:
        """Return one flat, action-independent tower vector per environment."""
        state = self.target_command.absolute_tower_state_tensor()
        return state.flatten(start_dim=1)


def build_serial_high_level_env(
    cfg,
    env: ManagerBasedRlEnv,
    wrapped_env: RslRlVecEnvWrapper,
    policy: Policy,
    *,
    settle_steps: int = 100,
    max_steps_per_block: int = 2000,
) -> HighLevelJengaEnv:
    """Assemble the real learned controller and raw tower observation provider."""
    runtime = SerialHighLevelRuntime(env)
    executor = LearnedLowLevelExecutor(
        env,
        wrapped_env,
        policy,
        settle_steps=settle_steps,
        max_steps_per_block=max_steps_per_block,
    )
    return HighLevelJengaEnv(
        cfg=cfg,
        low_level_executor=executor,
        observation_provider=runtime.observation,
        environment_resetter=runtime.reset,
    )
