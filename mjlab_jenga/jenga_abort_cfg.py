"""Abort variant of the low-level extraction task.

Gives the policy the option to stop working on the selected block instead of being
forced to continue until success, tower damage, or the episode length runs out. A
block that cannot be extracted safely should be abandoned so another can be chosen,
which is what a human player does and what a sequential game requires.

The mechanism follows the design on the `timur` branch
(mjlab_jenga/jenga_incomplete_random_abort_cfg.py), which established two points worth
carrying over:

* The termination is declared `time_out=True`. Aborting is an artificial cutoff, not a
  failure state, so the value function must bootstrap at that point rather than treat
  the state as worthless.
* The abort signal must be sustained. Thresholding a single sample makes an accidental
  crossing near-certain over a 2000-step episode, and that variant was measured firing
  in 27% of episodes at iteration 286 -- far too early to be a learned decision.
  Requiring ABORT_HOLD_STEPS consecutive crossings reduces the accidental rate to
  p**ABORT_HOLD_STEPS.

This module differs from that one in a way that matters for reuse. That variant also
had to add a tower-shift observation, because its policy had nothing to ground the
decision in. Ours already observes `tower_state` and `support_presence`, so the
observation space is unchanged at 45 and only the action space grows from 4 to 5. An
existing checkpoint can therefore be warm-started by widening three tensors in the
actor -- see widen_checkpoint_for_abort.py -- instead of retraining from scratch.

Nothing in jenga_mjenv_cfg.py is modified. Deleting this file restores the previous
behaviour exactly.

Reward arithmetic at the current weights, for a block abandoned at 50 mm of progress:

    success                    +8.0 progress  +10 success   = +18.0
    tower damage               +8.0 progress  -20 damage    = -12.0
    full timeout               +3.6 progress   -5 timeout   =  -1.4
    abort                      +3.6 progress   -2 abort     =  +1.6

Attempting is worth 18 - 30p against an abort value of 1.6, where p is the probability
of damaging the tower, so the policy should abandon a block once p exceeds roughly
0.55. Because the progress term rewards the furthest point reached and is not undone
by aborting, the incentive is to push as far as is safe and only then stop.
"""

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


# Threshold the abort action must exceed. Starts where Gaussian noise cannot reach it
# and eases as the policy learns, so early training is unaffected.
ABORT_THRESHOLD_START = 0.98
ABORT_THRESHOLD_END = 0.70
# common_step_counter advances by one per env.step(), i.e. 32 per iteration at the
# configured rollout length, so 40_000 steps is about 1250 iterations.
ABORT_CURRICULUM_STEPS = 40_000

# Consecutive over-threshold steps required before the abort takes effect. At the
# configured policy standard deviation of 0.2 and a mean output of 0, a single step
# exceeds 0.98 with probability ~5e-7, so accidental triggering is impossible well
# below this value; 100 keeps that true even if the mean drifts, at a cost of one
# second of confirmation delay in a 20-second episode.
ABORT_HOLD_STEPS = 100

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
    """Fires once the abort action has been over threshold for ABORT_HOLD_STEPS steps.

    Sized lazily, because configuration objects are built before the environment and
    its env count exist.
    """

    def __init__(self) -> None:
        self._hold: torch.Tensor | None = None

    def _update(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        raw = env.action_manager.get_term("abort").raw_action[:, 0]
        if self._hold is None:
            self._hold = torch.zeros_like(raw, dtype=torch.long)
        # Guard against a manager that does not forward reset() to plain callables:
        # a freshly reset environment must not inherit a partial hold from the
        # episode before it.
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
    """The same condition as a float, for reward and metric terms.

    A separate instance from the termination's on purpose: both are pure functions of
    the same per-step inputs and each is called exactly once per step by its own
    manager, so they stay in lockstep without sharing mutable state.
    """

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
    """The previous action, excluding the abort signal.

    The base task observes its own previous action so that the actor can account for
    the action-rate penalty. Letting that observation grow with the new action would
    widen the actor input from 45 to 46 and the critic input from 126 to 127, and in
    the critic the extra element lands before block_all_pos, displacing 81 values. A
    checkpoint of the base task could then only be reused by inserting a column in the
    middle of the first layer -- exactly the kind of surgery that fails silently.

    Excluding it keeps both observation spaces unchanged, so warm-starting only has to
    widen the actor output. Little is lost: the abort signal drives no actuator, so it
    does not enter the action-rate penalty in the way the physical actions do, and the
    hold counter that actually decides the termination is not observable either way.

    Relies on abort being registered last, which dict insertion order guarantees.
    """
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
    cfg = base.jenga_ppo_runner_cfg()
    cfg.experiment_name = "jenga_abort"
    return cfg


if __name__ == "__main__":
    built = jenga_env_cfg()
    print("actions:     ", list(built.actions.keys()))
    print("terminations:", list(built.terminations.keys()))
    print("rewards:     ", list(built.rewards.keys()))
    print("threshold:   ", ABORT_THRESHOLD_START, "->", ABORT_THRESHOLD_END,
          "over", ABORT_CURRICULUM_STEPS, "steps")
