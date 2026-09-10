import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from constants import PUSH_ACTION_DEADZONE, _HOOK1_CFG
from scene import hook_joint_pos_ordered

def stop_action_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (torch.abs(env.action_manager.action[:, 0]) <= PUSH_ACTION_DEADZONE).float()

def retreat_action_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.action_manager.action[:, 0] > PUSH_ACTION_DEADZONE).float()

def hook_x_position(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HOOK1_CFG,
) -> torch.Tensor:
    del asset_cfg
    return hook_joint_pos_ordered(env)[:, 0]