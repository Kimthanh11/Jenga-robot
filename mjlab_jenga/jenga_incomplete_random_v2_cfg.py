"""Incomplete-tower task with home-relative contact and yaw actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_jenga import jenga_incomplete_random_cfg as base

if base.TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


BLOCK_HALF_SIZE = base.BLOCK_HALF_SIZE

TOUCH_Y_LIMIT = BLOCK_HALF_SIZE[0]   # 0.025 m lateral
TOUCH_Z_LIMIT = BLOCK_HALF_SIZE[2]   # 0.015 m vertical

TOUCH_CURRICULUM_START = 0.0
TOUCH_CURRICULUM_END = 1.0
TOUCH_CURRICULUM_BEGIN_STEP = 10_000
TOUCH_CURRICULUM_STEPS = 50_000
YAW_CURRICULUM_START = 0.0
YAW_CURRICULUM_END = 1.0
YAW_CURRICULUM_BEGIN_STEP = 60_000
YAW_CURRICULUM_STEPS = 80_000
YAW_HOME_LIMIT = 1.0
YAW_STEP_SCALE = 0.05
ACTION_CLIP = 1.0


def _linear_curriculum_scale(
    env: ManagerBasedRlEnv,
    start: float,
    end: float,
    begin_step: int,
    steps: int,
) -> float:
    progress = min(max(env.common_step_counter - begin_step, 0) / steps, 1.0)
    return start + (end - start) * progress


def touch_curriculum_scale(env: ManagerBasedRlEnv) -> float:
    return _linear_curriculum_scale(
        env,
        TOUCH_CURRICULUM_START,
        TOUCH_CURRICULUM_END,
        TOUCH_CURRICULUM_BEGIN_STEP,
        TOUCH_CURRICULUM_STEPS,
    )


def yaw_curriculum_scale(env: ManagerBasedRlEnv) -> float:
    return _linear_curriculum_scale(
        env,
        YAW_CURRICULUM_START,
        YAW_CURRICULUM_END,
        YAW_CURRICULUM_BEGIN_STEP,
        YAW_CURRICULUM_STEPS,
    )


class JengaPushCommandV2(base.JengaPushCommand):
    """Same as v1 but exposes the per-env hook home pose (yaw, sx, sy, sz) the hook
    was teleported to, so action terms can offset targets around it."""

    def selected_hook_home(self) -> torch.Tensor:
        return self._hook_target[self._global_sel]        # [N,4] (yaw, sx, sy, sz)


@dataclass(kw_only=True)
class JengaPushCommandV2Cfg(base.JengaPushCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> JengaPushCommandV2:
        return JengaPushCommandV2(self, env)


def _push_cmd_v2(env: ManagerBasedRlEnv) -> JengaPushCommandV2:
    return env.command_manager.get_term("push")


@dataclass(kw_only=True)
class TaskFrameTouchActionCfg(ActionTermCfg):
    """Policy picks the contact point [lateral, vertical] on the selected block's
    push face; mapped to slide_y/slide_z position targets around the hook home."""

    scale: tuple[float, float] = (TOUCH_Y_LIMIT, TOUCH_Z_LIMIT)

    def build(self, env: ManagerBasedRlEnv) -> TaskFrameTouchAction:
        return TaskFrameTouchAction(self, env)


class TaskFrameTouchAction(ActionTerm):
    cfg: TaskFrameTouchActionCfg

    def __init__(self, cfg: TaskFrameTouchActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, joint_names = self._entity.find_joints(
            ("hook_slide_y", "hook_slide_z"),
            preserve_order=True,
        )
        self._target_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._target_names = joint_names
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._scale = torch.tensor(cfg.scale, device=self.device).view(1, 2)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)

        home = _push_cmd_v2(self._env).selected_hook_home()          # [N,4]
        offsets = self._raw_actions * self._scale * touch_curriculum_scale(self._env)
        self._processed_targets = home[:, 2:4] + offsets             # (sy, sz) + (dy, dz)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(
            self._processed_targets,
            joint_ids=self._target_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_targets[env_ids] = 0.0


@dataclass(kw_only=True)
class TaskFrameYawActionCfg(ActionTermCfg):
    """Relative yaw target, ramped by curriculum, clamped around the per-env home yaw."""

    scale: float = YAW_STEP_SCALE

    def build(self, env: ManagerBasedRlEnv) -> TaskFrameYawAction:
        return TaskFrameYawAction(self, env)


class TaskFrameYawAction(ActionTerm):
    cfg: TaskFrameYawActionCfg

    def __init__(self, cfg: TaskFrameYawActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, joint_names = self._entity.find_joints(
            ("hook_yaw",),
            preserve_order=True,
        )
        self._target_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._target_names = joint_names
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)

        home_yaw = _push_cmd_v2(self._env).selected_hook_home()[:, 0:1]   # [N,1]
        joint_pos = self._entity.data.joint_pos[:, self._target_ids]
        delta_yaw = self._raw_actions * self.cfg.scale * yaw_curriculum_scale(self._env)
        self._processed_targets = torch.clamp(
            joint_pos + delta_yaw,
            home_yaw - YAW_HOME_LIMIT,
            home_yaw + YAW_HOME_LIMIT,
        )

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(
            self._processed_targets,
            joint_ids=self._target_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_targets[env_ids] = 0.0


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = base._make_env_cfg()

    # Same 4-dim action space as v1, but touch/yaw are live (curriculum-gated).
    cfg.actions["touch"] = TaskFrameTouchActionCfg(
        entity_name="hook",
        scale=(TOUCH_Y_LIMIT, TOUCH_Z_LIMIT),
    )
    cfg.actions["yaw"] = TaskFrameYawActionCfg(
        entity_name="hook",
        scale=YAW_STEP_SCALE,
    )
    # Drop the frozen placeholders (their DOFs are now owned by the terms above).
    cfg.actions.pop("touch_y", None)
    cfg.actions.pop("touch_z", None)
    cfg.actions = {
        "x_velocity": cfg.actions["x_velocity"],
        "touch": cfg.actions["touch"],
        "yaw": cfg.actions["yaw"],
    }

    cfg.commands = {
        "push": JengaPushCommandV2Cfg(
            resampling_time_range=(1.0e9, 1.0e9),
            selectable_global_idx=base._SELECTABLE_GLOBAL,
        ),
    }

    return cfg


def jenga_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg()

    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False

    return cfg


def jenga_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = base.jenga_ppo_runner_cfg()
    cfg.experiment_name = "jenga_incomplete_randblock_v2"
    return cfg


if __name__ == "__main__":
    cfg = jenga_env_cfg()
    print(list(cfg.actions.keys()))
