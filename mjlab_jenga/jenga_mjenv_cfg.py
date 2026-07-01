from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
import math

from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.terrains import TerrainEntityCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import (
    JointEffortActionCfg,
    JointVelocityActionCfg,
    RelativeJointPositionActionCfg,
)
from mjlab.envs.mdp.rewards import joint_torques_l2, action_rate_l2
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Tower Configurations
LAYERS = 9
BLOCKS_PER_LAYER = 3

BLOCK_SIZE = (0.05, 0.152, 0.03)
BLOCK_HALF_SIZE = tuple(v / 2 for v in BLOCK_SIZE)

SIDE_SPACING = BLOCK_SIZE[0] + 0.0005
START_Z = (BLOCK_SIZE[2] / 2) + 0.0005
LAYER_HEIGHT = BLOCK_SIZE[2] + 0.0005

COLOR_A = (0.68, 0.85, 0.90, 1.0)
COLOR_B = (0.96, 0.96, 0.95, 1.0)



# get the Scene configurations
_JENGA_XML = Path(__file__).parent.parent / "jenga.xml"
_HOOK1_CFG = SceneEntityCfg("hook", joint_names=("hook_slide",))
_HOOK2_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide2",))
_HOOK3_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide3",))
_HOOK_Y_CFG = SceneEntityCfg("hook", joint_names=("hook_slide_y",))
_HOOK_Z_CFG = SceneEntityCfg("hook", joint_names=("hook_slide_z",))
_TARGET_BLOCK_CFG = SceneEntityCfg("b6_1", body_names=("b6_1",))
_REF_BLOCK_1_CFG = SceneEntityCfg("b6_2", body_names=("b6_2",))
_REF_BLOCK_2_CFG = SceneEntityCfg("b6_3", body_names=("b6_3",))
_HOOK_YAW_CFG = SceneEntityCfg("hook", joint_names=("hook_yaw",))
_HOOK_ALL_CFG = SceneEntityCfg(
    "hook",
    joint_names=("hook_slide", "hook_slide_y", "hook_slide_z", "hook_yaw"),
)
_HOOK_TIP_CFG = SceneEntityCfg("hook", site_names=("hook_tip",))



#block entities
def _vec(values) -> str:
    return " ".join(f"{v:g}" for v in values)


def _quat_from_z_rotation_deg(angle_deg: float) -> tuple[float, float, float, float]:
    angle = math.radians(angle_deg)
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def _get_block_infos():
    import random

    rng = random.Random(0)
    block_infos = []

    for layer in range(1, LAYERS + 1):
        for block in range(1, BLOCKS_PER_LAYER + 1):
            z = START_Z + (layer - 1) * LAYER_HEIGHT

            if layer % 2 == 1:
                x_positions = [-SIDE_SPACING, 0, SIDE_SPACING]
                x = x_positions[block - 1] + rng.uniform(-0.0005, 0.0005)
                y = 0.0 + rng.uniform(-0.0005, 0.0005)
                yaw_noise = rng.uniform(-1.0, 1.0)
                quat = _quat_from_z_rotation_deg(0.0 + yaw_noise)
            else:
                y_positions = [SIDE_SPACING, 0, -SIDE_SPACING]
                x = 0.0 + rng.uniform(-0.0005, 0.0005)
                y = y_positions[block - 1] + rng.uniform(-0.0005, 0.0005)
                yaw_noise = rng.uniform(-1.0, 1.0)
                quat = _quat_from_z_rotation_deg(90.0 + yaw_noise)

            if layer % 2 == 1:
                color = COLOR_A if block in (1, 3) else COLOR_B
            else:
                color = COLOR_B if block in (1, 3) else COLOR_A

            sliding = rng.uniform(0.2, 0.4)
            torsional = rng.uniform(0.01, 0.06)
            friction = (sliding, torsional, 0.001)

            block_infos.append({
                "name": f"b{layer}_{block}",
                "pos": (x, y, z),
                "quat": quat,
                "color": color,
                "friction": friction,
            })

    return block_infos


# loads the jenga_xml into an Mjspec, which is editable
def _spec_from_xml(xml: str) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(xml)


def _get_hook_spec() -> mujoco.MjSpec:
    xml = """
<mujoco model="hook">
  <compiler angle="degree" coordinate="local"/>

  <worldbody>
    <body name="hook" pos="0 0 0">
      <joint name="hook_slide" type="slide" axis="1 0 0" range="-0.09 0.02" limited="true" damping="2"/>
      <joint name="hook_slide_y" type="slide" axis="0 1 0" range="-0.01 0.01" limited="true" damping="2"/>
      <joint name="hook_slide_z" type="slide" axis="0 0 1" range="-0.002 0.07" limited="true" damping="2"/>
      <joint name="hook_yaw" type="hinge" axis="0 0 1" range="-60 60" limited="true" damping="2"/>

      <geom type="box"
            size="0.04 0.005 0.006"
            pos="0 0 0"
            rgba="0.1 0.1 0.9 1"
            density="2000"
            contype="0"
            conaffinity="0"/>

      <geom type="box"
            size="0.006 0.004 0.004"
            pos="-0.05 0 0"
            rgba="1 0 0 1"
            density="2000"/>
        <site name="hook_tip" pos="-0.056 0 0" size="0.003"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="hook_x_vel" joint="hook_slide" ctrlrange="-0.05 0.05" kv="100"/>    
    <position name="hook_y_pos" joint="hook_slide_y" ctrlrange="-0.01 0.01" kp="50"/>
    <position name="hook_z_pos" joint="hook_slide_z" ctrlrange="-0.002 0.07" kp="50"/>
    <position name="hook_yaw_pos" joint="hook_yaw" ctrlrange="-1 1" kp="20"/>
  </actuator>
</mujoco>
"""
    return _spec_from_xml(xml)


# tells mjlab those actuators are there. We DONT create a new object, unlike EntityCfg
_HOOK_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(target_names_expr=("hook_slide",)),
        XmlActuatorCfg(target_names_expr=("hook_slide_y",)),
        XmlActuatorCfg(target_names_expr=("hook_slide_z",)),
        XmlActuatorCfg(target_names_expr=("hook_yaw",)),
    ),
)

# blueprint for the Jenga-Entity (where is the model from and what are the actuators)
def _get_hook_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_hook_spec,
        articulation=_HOOK_ARTICULATION,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.15, 0.05, 0.16),
            joint_pos={
                "hook_slide": 0.0,
                "hook_slide_y": 0.0,
                "hook_slide_z": 0.0,
                "hook_yaw": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )



def _get_block_cfg(block_info) -> EntityCfg:
    def _get_block_spec() -> mujoco.MjSpec:
        xml = f"""
<mujoco model="{block_info["name"]}">
  <compiler angle="degree" coordinate="local"/>

    <default>
    <geom density="650"
            margin="0"
            gap="0"/>
    </default>

  <worldbody>
    <body name="{block_info["name"]}" pos="{_vec(block_info["pos"])}" quat="{_vec(block_info["quat"])}">
      <joint name="{block_info["name"]}_free" type="free"/>

      <geom type="box"
            size="{_vec(BLOCK_HALF_SIZE)}"
            rgba="{_vec(block_info["color"])}"
            friction="{_vec(block_info["friction"])}"/>
    </body>
  </worldbody>
</mujoco>
"""
        return _spec_from_xml(xml)

    return EntityCfg(
        spec_fn=_get_block_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=block_info["pos"],
            rot=block_info["quat"],
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


def _build_entities() -> dict[str, EntityCfg]:
    entities = {
        "hook": _get_hook_cfg(),
    }

    for block_info in _get_block_infos():
        entities[block_info["name"]] = _get_block_cfg(block_info)

    return entities


def make_all_block_cfgs():
    all_block_cfgs = []
    for block in _get_block_infos(): 
        name = block["name"]
        block_cfg = SceneEntityCfg(name, body_names=(name,))
        all_block_cfgs.append(block_cfg)
    return tuple(all_block_cfgs)

_ALL_BLOCK_CFGS = make_all_block_cfgs() #get block configs (for position/velocity)

def all_block_pos(env):
    positions = []
    for block_cfg in _ALL_BLOCK_CFGS:
        pos = target_block_pos(env, block_cfg)
        positions.append(pos)
    return torch.cat(positions, dim=-1)


# Observations


#custom reward functions for the position/velocity of the target block position
def target_block_pos(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    return position.squeeze(1)

def target_block_vel(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    velocity = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :]
    return velocity.squeeze(1)


# get COM of the tower
def get_com_per_block(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    get COM per block, expcept the target block. 
    """
    blocks = _get_block_infos()
    block_com_all = []
    target_block_name = asset_cfg.name
    for block in blocks:
        asset: Entity =env.scene[block["name"]]
        if block["name"] != target_block_name:
            block_com = asset.data.body_com_pos_w[:, 0, :]
            block_com_all.append(block_com)
        else:
            continue
    return torch.stack(block_com_all, dim=1)


def get_com_tower(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    get the COM of the tower.
    """
    block_com = get_com_per_block(env, asset_cfg)
    com_total_tower = torch.mean(block_com, dim=1)
    return com_total_tower

def initial_tower_com(
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    """
    Compute the initial tower COM from the initial block positions,
    excluding the target block.
    """
    block_positions = []
    target_block_name = asset_cfg.name

    for block in _get_block_infos():
        if block["name"] == target_block_name:
            continue

        block_positions.append(torch.tensor(block["pos"], dtype=torch.float32))

    block_positions = torch.stack(block_positions, dim=0)
    return torch.mean(block_positions, dim=0)


_START_TOWER_COM  = initial_tower_com()


def tower_com_shift(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    """
    Computes the horizontal shift of the COM of the Tower.
    """
    current = get_com_tower(env, asset_cfg)
    movement = current - _START_TOWER_COM.to(current.device)
    horizontal_shift = torch.norm(movement[:, :2], dim=-1)
    return horizontal_shift


#convert gripper to local coordinate frame of the block
def target_block_pose(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    extracts quaternion and position of block
    """
    asset: Entity = env.scene[asset_cfg.name]
    pose = asset.data.body_link_pose_w[:, asset_cfg.body_ids, :]
    pose = pose.squeeze(1)
    block_pos = pose[:, :3]
    block_quat = pose[:, 3:7]
    return block_pos, block_quat


def hook_tip_pos(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _HOOK_TIP_CFG) -> torch.Tensor:
    """
    get the position of the gripper
    """
    asset: Entity = env.scene[asset_cfg.name]
    hook_tip_position = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    return hook_tip_position.squeeze(1)


def hook_tip_pos_in_block_frame(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    Convert hook_tip_pos World coordinate system into a Block-local coordinate system.
    """
    block_pos_world, block_quat_world = target_block_pose(env, asset_cfg)
    hook_tip_pos_world = hook_tip_pos(env)

    position = hook_tip_pos_world - block_pos_world #vector from block_center to tip of the hook
    rotation = quat_apply_inverse(block_quat_world, position)

    return rotation


def _initial_block_pos(block_name: str) -> torch.Tensor:
    for block_info in _get_block_infos():
        if block_info["name"] == block_name:
            return torch.tensor(block_info["pos"])
    raise ValueError(f"Unknown block name: {block_name}")


_START_REF_POS = (_initial_block_pos("b6_2") + _initial_block_pos("b6_3")) / 2
_START_TARGET_REL_POS = _initial_block_pos("b6_1") - _START_REF_POS
# Rewards
def block_progress(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    ref_pos = get_block_ref_pos(env)
    target_pos = target_block_pos(env, asset_cfg)

    current_rel = target_pos - ref_pos
    movement_rel = current_rel - _START_TARGET_REL_POS.to(current_rel.device)

    extraction_direction = torch.tensor(
        [1.0, 0.0, 0.0],
        device=current_rel.device,
    )
    progress = torch.sum(movement_rel * extraction_direction, dim=-1)

    return progress


def tower_moderate_perturbation(env: ManagerBasedRlEnv) -> torch.Tensor:
    return tower_com_shift(env)


def tower_large_perturbation(env: ManagerBasedRlEnv) -> torch.Tensor:
    shift = tower_com_shift(env)
    return (shift > 0.02).float()


def action_norm(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.norm(env.action_manager.action, dim=-1)


def hook_x_position(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HOOK1_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)


def debug_reward_signals(env: ManagerBasedRlEnv) -> torch.Tensor:
    if env.common_step_counter % 500 == 0:
        print(
            "DEBUG_REWARD",
            f"step={env.common_step_counter}",
            f"progress_mean={block_progress(env).mean().item():.5f}",
            f"success_mean={success_block_reward(env).mean().item():.5f}",
            f"tower_shift_mean={tower_com_shift(env).mean().item():.5f}",
            f"large_mean={tower_large_perturbation(env).mean().item():.5f}",
            f"action_norm_mean={action_norm(env).mean().item():.5f}",
            f"hook_x_mean={hook_x_position(env).mean().item():.5f}",
            flush=True,
        )
    return torch.zeros(env.num_envs, device=env.device)


class DeltaBlockProgressReward:
    """Reward only new extraction progress since the previous environment step."""

    def __init__(self, asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG):
        self.asset_cfg = asset_cfg
        self.previous_progress: torch.Tensor | None = None
        self.needs_init: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        current_progress = block_progress(env, self.asset_cfg)

        if self.previous_progress is None:
            self.previous_progress = current_progress.clone()
            self.needs_init = torch.zeros_like(current_progress, dtype=torch.bool)
            return torch.zeros_like(current_progress)

        if self.needs_init is not None and torch.any(self.needs_init):
            self.previous_progress[self.needs_init] = current_progress[self.needs_init]
            self.needs_init[self.needs_init] = False

        delta_progress = current_progress - self.previous_progress
        self.previous_progress = current_progress.clone()

        return torch.clamp(delta_progress, min=0.0)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.previous_progress is None:
            return

        if self.needs_init is None:
            self.needs_init = torch.zeros_like(self.previous_progress, dtype=torch.bool)

        if env_ids is None:
            env_ids = slice(None)
        self.needs_init[env_ids] = True



def get_block_ref_pos(env : ManagerBasedRlEnv) -> torch.Tensor:
    ref1_block_pos = target_block_pos(env, _REF_BLOCK_1_CFG)
    ref2_block_pos = target_block_pos(env, _REF_BLOCK_2_CFG)
    ref_block_state_mean = (ref1_block_pos + ref2_block_pos) / 2
    return ref_block_state_mean


def success_block_extract(env : ManagerBasedRlEnv) -> torch.Tensor:
    progress = block_progress(env)
    return progress > 0.06


def success_block_reward(env : ManagerBasedRlEnv) -> torch.Tensor:
    return success_block_extract(env).float()


def tower_damage(env : ManagerBasedRlEnv) -> torch.Tensor:
    ref_pos = get_block_ref_pos(env)
    movement = ref_pos - _START_REF_POS.to(ref_pos.device)
    horizontal_movement = torch.norm(movement[:, :2], dim=-1)
    return horizontal_movement > 0.06





# Environment conifg


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
#observations actor + critic
    actor_terms = {
        "pusher_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _HOOK_ALL_CFG}
        ),
        "pusher_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _HOOK_ALL_CFG}
        ),
        "block_pos": ObservationTermCfg(
            func=target_block_pos,
            params={"asset_cfg": _TARGET_BLOCK_CFG}
        ),
    }

    critic_terms = {
        **actor_terms,
        "block_all_pos": ObservationTermCfg(
            func=all_block_pos,
        ),
        #"block_all_vel": ObservationTermCfg(
         #   func=target_block_vel,
          #  params={"asset_cfg": _ALL_BLOCK_CFGS},
        #),
    }


    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms),
    }


    #TODO Maybe swap effort (aka force) for velocity
    actions : dict[str, ActionTermCfg] = {
        "x_velocity": JointVelocityActionCfg(
            entity_name="hook",
            actuator_names=("hook_slide",),
            scale=0.03,
            clip={"hook_slide": (-0.03, 0.03)},
        ),
        "touch_y": RelativeJointPositionActionCfg(
            entity_name="hook",
            actuator_names=("hook_slide_y",),
            scale=0.000,
        ),
        "touch_z": RelativeJointPositionActionCfg(
            entity_name="hook",
            actuator_names=("hook_slide_z",),
            scale=0.000,
        ),
        "yaw" : RelativeJointPositionActionCfg(
            entity_name="hook",
            actuator_names=("hook_yaw",),
            scale=0.00,
        ),
    }


    hook_range = (-0.01, 0.01)
    events = {
        "reset_hook_x": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("hook", joint_names=("hook_slide",))
            }
        ),
        "reset_hook_y": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": hook_range,
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("hook", joint_names=("hook_slide_y",))
            }
        ),
        "reset_hook_z": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": hook_range,
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("hook", joint_names=("hook_slide_z",))
            }
        ),
    }


    rewards = {
        "delta_block_progress": RewardTermCfg(
            func=DeltaBlockProgressReward(),
            weight=40.0,
        ),
        # "torque_penalty": RewardTermCfg(
        #     func=joint_torques_l2,
        #     weight=-0.01,
        #     params={"asset_cfg": SceneEntityCfg("hook", joint_names=("hook_slide",))},
        # ),
        # "action_rate": RewardTermCfg( #to prevent the hook from wild jumping
        #     func=action_rate_l2,
        #     weight=-0.001,
        # ),
        "successful_extract": RewardTermCfg(
            func=success_block_reward,
            weight=900.0,
        ),
        "tower_moderate_pertub" : RewardTermCfg(
            func=tower_moderate_perturbation,
            weight=-0.2
        ),
        "tower_large_pertub" : RewardTermCfg(
            func=tower_large_perturbation,
            weight=-100.0
        ),
        "debug_reward_signals": RewardTermCfg(
            func=debug_reward_signals,
            weight=1e-12,
        ),
    }

    metrics = {
        "block_progress_last": MetricsTermCfg(
            func=block_progress,
            reduce="last",
        ),
        "delta_block_progress_mean": MetricsTermCfg(
            func=DeltaBlockProgressReward(),
            reduce="mean",
        ),
        "success_last": MetricsTermCfg(
            func=success_block_reward,
            reduce="last",
        ),
        "tower_com_shift_last": MetricsTermCfg(
            func=tower_com_shift,
            reduce="last",
        ),
        "tower_large_perturb_mean": MetricsTermCfg(
            func=tower_large_perturbation,
            reduce="mean",
        ),
        "action_norm_mean": MetricsTermCfg(
            func=action_norm,
            reduce="mean",
        ),
        "hook_x_position_last": MetricsTermCfg(
            func=hook_x_position,
            params={"asset_cfg": _HOOK1_CFG},
            reduce="last",
        ),
    }

    terminations = {
        "success": TerminationTermCfg(func=success_block_extract),
        #"tower_damage": TerminationTermCfg(func=tower_damage),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }


    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=_build_entities(),
            num_envs=8,
            env_spacing=4.0,
        ),
        #scale_rewards_by_dt=False,
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        metrics=metrics,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=1.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            nconmax=4096,
            njmax=4096,
            mujoco=MujocoCfg(timestep=0.002),
        ),
        decimation=5,
        episode_length_s=20.0,
    )



def jenga_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg()

    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False

    return cfg



def jenga_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="jenga",
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=2000,
  )




if __name__ == "__main__":
    entities = _build_entities()
    print(entities.keys())
    print(len(entities))
