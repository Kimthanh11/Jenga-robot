"""Reusable execution of one complete learned low-level Jenga attempt."""

from collections.abc import Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from mjlab_jenga.jenga_mjenv_cfg import (
    TargetBlockCommand,
    remove_block_from_tower,
    success_block_extract,
    tower_damage,
)


Policy = Callable[[object], torch.Tensor]


def action_column(env: ManagerBasedRlEnv, term_name: str) -> int:
    """Return the first action column consumed by the named action term."""
    manager = env.action_manager
    try:
        term_index = manager.active_terms.index(term_name)
    except ValueError as error:
        raise ValueError(f"Unknown action term: {term_name}") from error
    return sum(manager.action_term_dim[:term_index])


def get_target_command(env: ManagerBasedRlEnv) -> TargetBlockCommand:
    """Return the concrete command object responsible for target-block state."""
    command = env.command_manager.get_term("target_block")
    if not isinstance(command, TargetBlockCommand):
        raise TypeError("target_block has an unexpected command term type")
    return command


def settle_after_success(
    env: ManagerBasedRlEnv,
    wrapped_env: RslRlVecEnvWrapper,
    last_actions: torch.Tensor,
    settle_steps: int,
) -> str:
    """Stop pushing while preserving the other command components."""
    if settle_steps < 0:
        raise ValueError("settle_steps must not be negative")

    hold_actions = last_actions.clone()
    hold_actions[:, action_column(env, "push_stop_retreat")] = 0.0

    for _ in range(settle_steps):
        wrapped_env.step(hold_actions)
        if tower_damage(env)[0].item():
            return "tower_damage"

    if not success_block_extract(env)[0].item():
        return "settle_failed"
    return "success"


def extract_block(
    env: ManagerBasedRlEnv,
    target_block: str,
    wrapped_env: RslRlVecEnvWrapper,
    policy: Policy,
    settle_steps: int,
    max_steps_per_block: int,
) -> tuple[str, str, int]:
    """Run the learned controller until one complete attempt has an outcome.

    This function deliberately does not park a successful block or rebaseline the
    tower. Evaluation code may need to inspect that intermediate settled state first.
    """
    if max_steps_per_block <= 0:
        raise ValueError("max_steps_per_block must be positive")

    get_target_command(env).set_target_block(target_block)
    obs = wrapped_env.get_observations()
    reason = "step_cap"
    steps = 0

    for _ in range(max_steps_per_block):
        with torch.no_grad():
            actions = policy(obs)

        obs, _, _, _ = wrapped_env.step(actions)
        steps += 1

        # These terms are evaluated directly. Registering them as environment
        # terminations would make MJLab require a reset after every attempt.
        if success_block_extract(env)[0].item():
            reason = settle_after_success(
                env=env,
                wrapped_env=wrapped_env,
                last_actions=actions,
                settle_steps=settle_steps,
            )
            break

        if tower_damage(env)[0].item():
            reason = "tower_damage"
            break

    return target_block, reason, steps


class LearnedLowLevelExecutor:
    """Adapt the learned controller to ``HighLevelJengaEnv``'s callable contract."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        wrapped_env: RslRlVecEnvWrapper,
        policy: Policy,
        *,
        settle_steps: int = 100,
        max_steps_per_block: int = 2000,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("The sequential learned executor currently requires num_envs=1")
        self.env = env
        self.wrapped_env = wrapped_env
        self.policy = policy
        self.settle_steps = settle_steps
        self.max_steps_per_block = max_steps_per_block

    def __call__(self, block_name: str) -> tuple[str, int]:
        """Execute and finalize one block attempt."""
        _, reason, steps = extract_block(
            env=self.env,
            target_block=block_name,
            wrapped_env=self.wrapped_env,
            policy=self.policy,
            settle_steps=self.settle_steps,
            max_steps_per_block=self.max_steps_per_block,
        )

        if reason == "success":
            remove_block_from_tower(self.env, block_name)
            get_target_command(self.env).rebaseline()

        return reason, steps
