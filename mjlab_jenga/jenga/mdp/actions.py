from __future__ import annotations

from dataclasses import dataclass, field
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mdp.commands import *
from scene import *
from constants import _TARGET_BLOCK_CFG


def block_vector_to_world(
    env: ManagerBasedRlEnv,
    vector_block: torch.Tensor,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    """Rotate a vector from the target block frame into the world frame."""
    _, block_quat_world = target_block_pose(env, asset_cfg)
    vector_block = vector_block.to(
        device=block_quat_world.device,
        dtype=block_quat_world.dtype,
    )
    if vector_block.ndim == 1:
        vector_block = vector_block.unsqueeze(0).repeat(env.num_envs, 1)
    return quat_apply(block_quat_world, vector_block)

def block_point_to_world(
    env: ManagerBasedRlEnv,
    point_block: torch.Tensor,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    """Transform a point from the target block frame into the world frame."""
    block_pos_world, _ = target_block_pose(env, asset_cfg)
    return block_pos_world + block_vector_to_world(env, point_block, asset_cfg)



def hook_slide_targets_for_tip_world(
    env: ManagerBasedRlEnv,
    tip_world: torch.Tensor,
) -> torch.Tensor:
    """Convert a desired hook-tip world point into slide joint coordinates."""
    hook_joint_pos = hook_joint_pos_ordered(env)
    yaw = hook_joint_pos[:, 3]

    base = torch.tensor(
        HOOK_BASE_POS,
        device=tip_world.device,
        dtype=tip_world.dtype,
    ).view(1, 3)
    rel = tip_world - base

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    slide = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1] - HOOK_TIP_LOCAL_X
    slide_y = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
    slide_z = rel[:, 2]

    return torch.stack((slide, slide_y, slide_z), dim=-1)

def block_contact_to_hook_yz_targets(
    env: ManagerBasedRlEnv,
    contact_block: torch.Tensor,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    contact_block = contact_block.clone()
    contact_block[:, 0] = torch.clamp(
        contact_block[:, 0],
        -CONTACT_X_LIMIT,
        CONTACT_X_LIMIT,
    )
    contact_block[:, 1] = torch.clamp(
        contact_block[:, 1],
        -CONTACT_Y_LIMIT,
        CONTACT_Y_LIMIT,
    )
    contact_block[:, 2] = torch.clamp(
        contact_block[:, 2],
        -CONTACT_Z_LIMIT,
        CONTACT_Z_LIMIT,
    )

    contact_world = block_point_to_world(env, contact_block, asset_cfg)
    target_slides = hook_slide_targets_for_tip_world(env, contact_world)
    target_y = target_slides[:, 1]
    target_z = target_slides[:, 2]

    target_y = torch.clamp(
        target_y,
        HOOK_SLIDE_Y_TARGET_RANGE[0],
        HOOK_SLIDE_Y_TARGET_RANGE[1],
    )
    target_z = torch.clamp(
        target_z,
        HOOK_SLIDE_Z_TARGET_RANGE[0],
        HOOK_SLIDE_Z_TARGET_RANGE[1],
    )

    return torch.stack([target_y, target_z], dim=-1)

def _linear_curriculum_scale(
    env: ManagerBasedRlEnv,
    start: float,
    end: float,
    begin_step: int,
    steps: int,
) -> torch.Tensor:
    progress = min(max(curriculum_step(env) - begin_step, 0) / steps, 1.0)
    scale = start + (end - start) * progress
    return torch.tensor(scale, device=env.device)

def yaw_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _linear_curriculum_scale(
        env,
        YAW_CURRICULUM_START,
        YAW_CURRICULUM_END,
        YAW_CURRICULUM_BEGIN_STEP,
        YAW_CURRICULUM_STEPS,
    )

def touch_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _linear_curriculum_scale(
        env,
        TOUCH_CURRICULUM_START,
        TOUCH_CURRICULUM_END,
        TOUCH_CURRICULUM_BEGIN_STEP,
        TOUCH_CURRICULUM_STEPS,
    )

@dataclass(kw_only=True)
class PushStopRetreatActionCfg(ActionTermCfg):
    """Signed velocity command with explicit push, stop, and retreat regions."""

    push_speed: float = PUSH_X_VELOCITY_SCALE
    retreat_speed: float = PUSH_X_VELOCITY_CLIP[1]
    deadzone: float = PUSH_ACTION_DEADZONE
    max_velocity_change: float = PUSH_VELOCITY_CHANGE_PER_STEP

    def build(self, env: ManagerBasedRlEnv) -> PushStopRetreatAction:
        return PushStopRetreatAction(self, env)


class PushStopRetreatAction(ActionTerm):
    cfg: PushStopRetreatActionCfg

    def __init__(self, cfg: PushStopRetreatActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, _ = self._entity.find_joints(("hook_slide",), preserve_order=True)
        self._target_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_velocity = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_velocity(self) -> torch.Tensor:
        return self._processed_velocity

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)
        magnitude = torch.clamp(
            (torch.abs(self._raw_actions) - self.cfg.deadzone)
            / (1.0 - self.cfg.deadzone),
            0.0,
            1.0,
        )
        speed = torch.where(
            self._raw_actions < 0.0,
            torch.full_like(self._raw_actions, self.cfg.push_speed),
            torch.full_like(self._raw_actions, self.cfg.retreat_speed),
        )
        desired = torch.sign(self._raw_actions) * magnitude * speed
        velocity_delta = torch.clamp(
            desired - self._processed_velocity,
            -self.cfg.max_velocity_change,
            self.cfg.max_velocity_change,
        )
        self._processed_velocity += velocity_delta

    def apply_actions(self) -> None:
        self._entity.set_joint_velocity_target(
            self._processed_velocity,
            joint_ids=self._target_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_velocity[env_ids] = 0.0

@dataclass(kw_only=True)
class BlockLocalHookYZActionCfg(ActionTermCfg):
    """Choose a contact point on the target block face."""

    scale: tuple[float, float] = (CONTACT_X_LIMIT, CONTACT_Z_LIMIT)
    contact_y: float = CONTACT_FACE_Y
    asset_cfg: SceneEntityCfg = field(
        default_factory=lambda: SceneEntityCfg("b6_1", body_names=("b6_1",))
    )

    def build(self, env: ManagerBasedRlEnv) -> BlockLocalHookYZAction:
        return BlockLocalHookYZAction(self, env)


class BlockLocalHookYZAction(ActionTerm):
    """Policy contact action [block_lateral, block_z] -> hook_slide_y/z targets."""

    cfg: BlockLocalHookYZActionCfg

    def __init__(self, cfg: BlockLocalHookYZActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, joint_names = self._entity.find_joints(
            ("hook_slide_y", "hook_slide_z"),
            preserve_order=True,
        )
        self._target_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._target_names = joint_names
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._contact_block = torch.zeros(self.num_envs, 3, device=self.device)
        self._scale = torch.tensor(cfg.scale, device=self.device).view(1, 2)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)

        contact_block = torch.zeros(self.num_envs, 3, device=self.device)
        scaled_actions = self._raw_actions * self._scale * touch_curriculum_scale(self._env)
        contact_block[:, 0] = scaled_actions[:, 0]
        cmd = target_command_or_none(self._env)
        if cmd is not None and self.cfg.asset_cfg.name == _TARGET_BLOCK_CFG.name:
            contact_block[:, 1] = cmd.selected_contact_face_y()
        else:
            contact_block[:, 1] = self.cfg.contact_y
        contact_block[:, 2] = scaled_actions[:, 1]
        self._contact_block[:] = contact_block

        self._processed_targets = block_contact_to_hook_yz_targets(
            self._env,
            contact_block,
            self.cfg.asset_cfg,
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

@dataclass(kw_only=True)
class CurriculumYawActionCfg(ActionTermCfg):
    """Relative yaw target whose effective scale ramps up during training."""

    scale: float = 0.05

    def build(self, env: ManagerBasedRlEnv) -> CurriculumYawAction:
        return CurriculumYawAction(self, env)


class CurriculumYawAction(ActionTerm):
    """Relative yaw target, integrated on the COMMANDED target.

    Integrating on the measured joint position instead makes the servo error identically
    zero whenever the action is small, so the joint free-wheels under contact load and
    the target ratchets along with it. Measured on b1_1: yaw drifted 90.1deg -> 85.7deg
    while the target tracked it to within 0.06deg. At the ~0.2 m lever arm that is ~15 mm
    of tip travel lost, and it cost ~25% of the delivered push force across all targets.
    """

    cfg: CurriculumYawActionCfg

    def __init__(self, cfg: CurriculumYawActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, joint_names = self._entity.find_joints(
            ("hook_yaw",),
            preserve_order=True,
        )
        self._target_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._target_names = joint_names
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # The action manager resets before the command manager resamples, so the new
        # home yaw is not known yet at reset time. Seed the target lazily instead.
        self._needs_home_init = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)

        delta_yaw = self._raw_actions * self.cfg.scale * yaw_curriculum_scale(self._env)
        cmd = target_command_or_none(self._env)
        if cmd is not None:
            home_yaw = cmd.selected_hook_home()[:, 3:4]
            self._processed_targets = torch.where(
                self._needs_home_init.unsqueeze(-1), home_yaw, self._processed_targets
            )
            self._needs_home_init[:] = False
            self._processed_targets = torch.clamp(
                self._processed_targets + delta_yaw,
                home_yaw - YAW_TARGET_LIMIT,
                home_yaw + YAW_TARGET_LIMIT,
            )
            return

        self._processed_targets = torch.clamp(
            self._processed_targets + delta_yaw,
            -YAW_TARGET_LIMIT,
            YAW_TARGET_LIMIT,
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
        self._needs_home_init[env_ids] = True

def push_velocity_target(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = env.action_manager.get_term("push_stop_retreat")
    if not isinstance(term, PushStopRetreatAction):
        raise TypeError("push_stop_retreat has an unexpected action term type")
    return term.processed_velocity.squeeze(-1)