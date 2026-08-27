from __future__ import annotations

# =====================================================================================
# ABORT variant of the incomplete-tower + random-block task, built on V2 (task-frame
# touch/yaw), NOT v1.
#
# v1 (jenga_incomplete_random_cfg.py) is a known-broken baseline: its touch_y/touch_z/
# yaw actions are frozen RelativeJointPositionActionCfg(scale=0.000) placeholders,
# which let the hook's y/z servo target passively drift with whatever the joint gets
# pushed to by contact forces instead of holding a rigid setpoint -- the hook can
# never sustain contact against the block face. Confirmed by job 133345 (this file's
# first version, v1-based): 6000 iters, success_mean=0.00000, progress_mean~0 the
# whole run -- identical to the documented old v1/randblock failure (job 116239).
# v2 replaced the frozen placeholders with live, curriculum-gated TaskFrameTouchAction/
# TaskFrameYawAction that hold a real home-relative PD setpoint, which is why we build
# on it here instead.
#
# From Discord (Boris, 23.07.2026): give the policy an option to abort pushing the
# currently-selected block if the tower looks unsafe, rather than being forced to
# keep going until success/topple/timeout — and don't punish too much for choosing to
# abort and move on to another block.
#
# Mechanically this needs three additions on top of v2:
#   * A "tower_shift" observation (v1.tower_com_shift was already computed for
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
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_jenga import jenga_incomplete_random_cfg as v1
from mjlab_jenga import jenga_incomplete_random_v2_cfg as base

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


ABORT_THRESHOLD_START = 0.98   # raw_action is clamped to [-1,1] -- 0.95 is NOT a rare
ABORT_THRESHOLD_END = 0.7      # tail value for a ~2-3 std policy, checked every step
ABORT_CURRICULUM_STEPS = 100_000
ABORT_HOLD_STEPS = 10           # require this many CONSECUTIVE over-threshold steps.
# Single-sample thresholding over a ~2000-step episode makes an accidental crossing
# almost certain regardless of threshold (P(any hit) = 1-(1-p)^2000). Requiring a
# sustained run of ABORT_HOLD_STEPS drops accidental-trigger probability to ~p^10,
# forcing the policy to actually commit to the signal rather than get flagged by one
# lucky noise sample. Confirmed the bug in practice: job 138899 (single-sample,
# threshold~0.92 by iter ~286) already showed Episode_Termination/abort=27% -- far
# too high to be a learned decision this early (2026-08-27).
ABORT_PENALTY_WEIGHT = -2.0     # mild: well below -100 (tower_large_pertub), non-zero
ACTION_CLIP = 1.0


def abort_threshold_curriculum(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = min(env.common_step_counter / ABORT_CURRICULUM_STEPS, 1.0)
    scale = ABORT_THRESHOLD_START + (
        ABORT_THRESHOLD_END - ABORT_THRESHOLD_START
    ) * progress
    return torch.tensor(scale, device=env.device)


class AbortTriggered:
    """Stateful bool signal: fires only after ABORT_HOLD_STEPS consecutive steps with
    the abort action over the (curriculum-ramped) threshold. Lazily sized on first
    call, since cfg construction happens before `env`/num_envs exist -- mirrors
    DeltaBlockProgressReward's pattern above in the base (v1) file."""

    def __init__(self) -> None:
        self._hold: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        abort_action = env.action_manager.get_term("abort").raw_action[:, 0]
        if self._hold is None:
            self._hold = torch.zeros_like(abort_action, dtype=torch.long)
        over = abort_action > abort_threshold_curriculum(env)
        self._hold = torch.where(over, self._hold + 1, torch.zeros_like(self._hold))
        return self._hold >= ABORT_HOLD_STEPS

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._hold is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._hold[env_ids] = 0


class AbortRewardSignal(AbortTriggered):
    """Same hold-counter logic as AbortTriggered, exposed as float for reward/metric
    terms. Deliberately a SEPARATE instance from the termination's: both are pure
    functions of the same per-step inputs (raw_action, curriculum threshold) and each
    gets called exactly once per step by its own manager, so independent instances
    stay in lockstep without needing to share mutable state or coordinate call order."""

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return super().__call__(env).float()


def tower_shift_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    # v1.tower_com_shift is (num_envs,); observation terms need a feature dim.
    return v1.tower_com_shift(env).unsqueeze(-1)


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
        func=AbortTriggered(),
        time_out=True,
    )

    cfg.rewards["abort_penalty"] = RewardTermCfg(
        func=AbortRewardSignal(),
        weight=ABORT_PENALTY_WEIGHT,
    )

    cfg.metrics["abort_rate_mean"] = MetricsTermCfg(
        func=AbortRewardSignal(),
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
    cfg.experiment_name = "jenga_incomplete_randblock_abort_v2"
    return cfg


if __name__ == "__main__":
    cfg = jenga_env_cfg()
    print(list(cfg.actions.keys()))
    print(list(cfg.terminations.keys()))
    print(list(cfg.observations["actor"].terms.keys()))
