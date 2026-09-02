"""Optional policy action for abandoning the current extraction attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_jenga import jenga_mjenv_cfg as base

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Kept constant so resumed runs use the same decision boundary.
ABORT_THRESHOLD_START = 0.80
ABORT_THRESHOLD_END = 0.80
ABORT_CURRICULUM_STEPS = 40_000

# A longer hold made exploration of the abort action effectively impossible.
ABORT_HOLD_STEPS = 1

ABORT_PENALTY_WEIGHT = -2.0
ACTION_CLIP = 1.0


def abort_threshold(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current abort threshold, ramped over ABORT_CURRICULUM_STEPS."""
    progress = min(base.curriculum_step(env) / ABORT_CURRICULUM_STEPS, 1.0)
    value = ABORT_THRESHOLD_START + (
        ABORT_THRESHOLD_END - ABORT_THRESHOLD_START
    ) * progress
    return torch.tensor(value, device=env.device)


class _AbortHoldCounter:
    """Track consecutive abort signals for each environment."""

    def __init__(self) -> None:
        self._hold: torch.Tensor | None = None

    def _update(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        raw = env.action_manager.get_term("abort").raw_action[:, 0]
        if self._hold is None:
            self._hold = torch.zeros_like(raw, dtype=torch.long)
        # Reset here as a fallback for managers that do not call reset() on callables.
        self._hold = torch.where(
            env.episode_length_buf <= 1, torch.zeros_like(self._hold), self._hold
        )
        over = raw > abort_threshold(env)
        self._hold = torch.where(over, self._hold + 1, torch.zeros_like(self._hold))
        return self._hold >= ABORT_HOLD_STEPS

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._update(env)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._hold is None:
            return
        self._hold[slice(None) if env_ids is None else env_ids] = 0


class _AbortHoldSignal(_AbortHoldCounter):
    """Expose the abort condition to reward and metric terms."""

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._update(env).float()


@dataclass(kw_only=True)
class AbortSignalActionCfg(ActionTermCfg):
    """A policy output that drives no actuator; the termination reads it."""

    def build(self, env: ManagerBasedRlEnv) -> AbortSignalAction:
        return AbortSignalAction(self, env)


class AbortSignalAction(ActionTerm):
    cfg: AbortSignalActionCfg

    def __init__(self, cfg: AbortSignalActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)

    def apply_actions(self) -> None:
        """Intentionally empty: this action carries a decision, not a command."""

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[slice(None) if env_ids is None else env_ids] = 0.0


def last_action_without_abort(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Keep the base observation shape when adding the abort output."""
    return env.action_manager.prev_action[:, :-1]


def jenga_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The base environment with the abort action, termination, penalty and metric."""
    cfg = base.jenga_env_cfg(play=play)

    cfg.actions["abort"] = AbortSignalActionCfg(entity_name="hook")
    for group in ("actor", "critic"):
        cfg.observations[group].terms["last_action"].func = last_action_without_abort
    cfg.terminations["abort"] = TerminationTermCfg(
        func=_AbortHoldCounter(),
        time_out=True,
    )
    cfg.rewards["abort_penalty"] = RewardTermCfg(
        func=_AbortHoldSignal(),
        weight=ABORT_PENALTY_WEIGHT,
    )
    cfg.metrics["abort_rate_mean"] = MetricsTermCfg(
        func=_AbortHoldSignal(),
        reduce="mean",
    )
    cfg.metrics["abort_last"] = MetricsTermCfg(
        func=_AbortHoldSignal(),
        reduce="last",
    )
    return cfg


def jenga_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the base experiment name so warm-start checkpoints resolve correctly."""
    return base.jenga_ppo_runner_cfg()


if __name__ == "__main__":
    built = jenga_env_cfg()
    print("actions:     ", list(built.actions.keys()))
    print("terminations:", list(built.terminations.keys()))
    print("rewards:     ", list(built.rewards.keys()))
    print("threshold:   ", ABORT_THRESHOLD_START, "->", ABORT_THRESHOLD_END,
          "over", ABORT_CURRICULUM_STEPS, "steps")
