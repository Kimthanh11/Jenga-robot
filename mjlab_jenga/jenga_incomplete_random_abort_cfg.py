from __future__ import annotations

# =====================================================================================
# ABORT variant of the incomplete-tower + random-block task (v1 base, no touch/yaw).
#
# From Discord (Boris, 23.07.2026): give the policy an option to abort pushing the
# currently-selected block if the tower looks unsafe, rather than being forced to
# keep going until success/topple/timeout — and don't punish too much for choosing to
# abort and move on to another block.
#
# Mechanically this needs three additions on top of v1:
#   * A "tower_shift" observation (base.tower_com_shift was already computed for
#     reward shaping, but was never exposed to the policy — without it there's
#     nothing for an abort decision to be grounded in).
#   * A 1-dim "abort" action that drives nothing physically; a termination reads it.
#   * A termination that fires when the abort action crosses a curriculum-ramped
#     threshold (starts ~unreachable by random Gaussian noise, eases over the same
#     100k-step timescale as the existing perturbation/success curricula), coded
#     time_out=True (artificial cutoff -> bootstrap the value function there, same
#     bucket as the max-episode-length timeout, not a true terminal/failure state).
#
# Ending an episode this way already causes a fresh random block pick on reset (any
# termination -> full env reset -> JengaPushCommand._resample_command runs
# unconditionally) — no extra wiring needed for "choosing another block."
# =====================================================================================

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_jenga import jenga_incomplete_random_cfg as base

if base.TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


ABORT_THRESHOLD_START = 0.95   # ~unreachable by random Gaussian noise (init_std=1.0)
ABORT_THRESHOLD_END = 0.6
ABORT_CURRICULUM_STEPS = 100_000
ABORT_PENALTY_WEIGHT = -2.0     # mild: well below -100 (tower_large_pertub), non-zero
ACTION_CLIP = 1.0


def abort_threshold_curriculum(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = min(env.common_step_counter / ABORT_CURRICULUM_STEPS, 1.0)
    scale = ABORT_THRESHOLD_START + (
        ABORT_THRESHOLD_END - ABORT_THRESHOLD_START
    ) * progress
    return torch.tensor(scale, device=env.device)


def abort_triggered(env: ManagerBasedRlEnv) -> torch.Tensor:
    abort_action = env.action_manager.get_term("abort").raw_action[:, 0]
    return abort_action > abort_threshold_curriculum(env)


def abort_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return abort_triggered(env).float()


def tower_shift_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    # base.tower_com_shift is (num_envs,); observation terms need a feature dim.
    return base.tower_com_shift(env).unsqueeze(-1)


@dataclass(kw_only=True)
class AbortSignalActionCfg(ActionTermCfg):
    """A policy-controlled signal that drives no joint; terminations/rewards read it."""

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
        pass

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = base._make_env_cfg()

    cfg.observations["actor"].terms["tower_shift"] = ObservationTermCfg(
        func=tower_shift_obs,
    )
    cfg.observations["critic"].terms["tower_shift"] = ObservationTermCfg(
        func=tower_shift_obs,
    )

    cfg.actions["abort"] = AbortSignalActionCfg(entity_name="hook")

    cfg.terminations["abort"] = TerminationTermCfg(
        func=abort_triggered,
        time_out=True,
    )

    cfg.rewards["abort_penalty"] = RewardTermCfg(
        func=abort_reward,
        weight=ABORT_PENALTY_WEIGHT,
    )

    cfg.metrics["abort_rate_mean"] = MetricsTermCfg(
        func=abort_reward,
        reduce="mean",
    )

    return cfg


def jenga_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg()

    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False

    return cfg


def jenga_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = base.jenga_ppo_runner_cfg()
    cfg.experiment_name = "jenga_incomplete_randblock_abort"
    return cfg


if __name__ == "__main__":
    cfg = jenga_env_cfg()
    print(list(cfg.actions.keys()))
    print(list(cfg.terminations.keys()))
    print(list(cfg.observations["actor"].terms.keys()))
