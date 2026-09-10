from mjlab.envs import ManagerBasedRlEnv
from constants import _HOOK_ALL_CFG, _HOOK_JOINT_ORDER, _TARGET_BLOCK_CFG, CONTACT_FORCE_OBS_NORMALIZER, CONTACT_FORCE_OBS_CLIP, TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT, TOWER_DAMAGE_MAX_BLOCK_ROTATION, TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT
from mdp.commands import *
from utils import *
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

def hook_joint_pos_ordered(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hook joints in policy order: slide, slide_y, slide_z, yaw."""
    asset: Entity = env.scene[_HOOK_ALL_CFG.name]
    joint_ids, _ = asset.find_joints(_HOOK_JOINT_ORDER, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids]


def hook_tip_pos_in_task_frame(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    block_pos_world, _ = target_block_pose(env, asset_cfg)
    task_quat_world = target_task_quat_w(env, asset_cfg)
    return quat_apply_inverse(task_quat_world, hook_tip_pos(env) - block_pos_world)

def target_block_movement_in_task_frame(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    movement_world = target_block_relative_movement(env, asset_cfg)
    task_quat_world = target_task_quat_w(env, asset_cfg)
    return quat_apply_inverse(task_quat_world, movement_world)


def target_block_vel_in_task_frame(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    vel_world = target_block_vel(env, asset_cfg)
    task_quat_world = target_task_quat_w(env, asset_cfg)
    return quat_apply_inverse(task_quat_world, vel_world)

def hook_contact_force_in_task_frame(env: ManagerBasedRlEnv) -> torch.Tensor:
    return quat_apply_inverse(target_task_quat_w(env), hook_contact_force_world(env))


def hook_contact_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Normalized task-frame force plus a binary contact flag."""
    force = torch.clamp(
        hook_contact_force_in_task_frame(env) / CONTACT_FORCE_OBS_NORMALIZER,
        -CONTACT_FORCE_OBS_CLIP,
        CONTACT_FORCE_OBS_CLIP,
    )
    found = hook_contact_found(env).unsqueeze(-1).to(force.dtype)
    return torch.cat((force, found), dim=-1)

def tower_instability_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    horizontal = normalized_limit_excess(
        tower_max_block_horizontal_shift(env),
        TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT,
        TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT,
    )
    vertical = normalized_limit_excess(
        tower_max_block_vertical_shift(env),
        TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT,
        TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT,
    )
    rotation = normalized_limit_excess(
        tower_max_block_rotation(env),
        TOWER_SUCCESS_MAX_BLOCK_ROTATION,
        TOWER_DAMAGE_MAX_BLOCK_ROTATION,
    )
    return torch.maximum(torch.maximum(horizontal, vertical), rotation)

def tower_state_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """How close the tower is to the damage limits, as fractions of those limits.

    The actor is penalized for new instability and the episode terminates on damage,
    but neither quantity was observable. A value of 1.0 means the limit is reached.
    """
    return torch.stack(
        (
            tower_max_block_horizontal_shift(env)
            / TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT,
            tower_max_block_vertical_shift(env)
            / TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT,
            tower_max_block_rotation(env) / TOWER_DAMAGE_MAX_BLOCK_ROTATION,
            tower_instability_fraction(env),
        ),
        dim=-1,
    )

def last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Previous action. There is an action-rate penalty the actor could not account
    for without knowing what it did last step."""
    return env.action_manager.prev_action

def make_all_block_cfgs():
    all_block_cfgs = []
    for block in get_block_infos(): 
        name = block["name"]
        block_cfg = SceneEntityCfg(name, body_names=(name,))
        all_block_cfgs.append(block_cfg)
    return tuple(all_block_cfgs)

_ALL_BLOCK_CFGS = make_all_block_cfgs() #get block configs (for position/velocity)

def all_block_pos(env):
    """Critic observation: every block's world position, zeros for absent blocks.

    Two defects this avoids. Routing through target_block_pos() returned the CURRENTLY
    SELECTED block for the b6_1 slot, because that helper short-circuits on the target
    asset name -- so one of the 27 slots did not mean what its position implied. And
    blocks removed by the missing-block randomization are parked 1.5 m away rather than
    deleted, which fed the critic a metre-scale jump in three of the 81 inputs.
    """
    positions = []
    for block_cfg in _ALL_BLOCK_CFGS:
        asset: Entity = env.scene[block_cfg.name]
        pos = asset.data.body_link_pos_w[:, 0, :]
        present = ~current_missing_block_mask(env, block_cfg.name)
        positions.append(pos * present.unsqueeze(-1).to(pos.dtype))
    return torch.cat(positions, dim=-1)
