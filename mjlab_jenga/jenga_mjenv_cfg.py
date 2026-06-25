from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
import math

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

BLOCK_SIZE = (0.05, 0.15, 0.03)
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
_REF_BLOCKS_CFG = SceneEntityCfg(
    "jenga",
    body_names=("b6_2", "b6_3"),
)


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
                x = x_positions[block - 1]
                y = 0.0
                quat = _quat_from_z_rotation_deg(0.0)
            else:
                y_positions = [SIDE_SPACING, 0, -SIDE_SPACING]
                x = 0.0
                y = y_positions[block - 1]
                quat = _quat_from_z_rotation_deg(90.0)

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
      <joint name="hook_slide" type="slide" axis="1 0 0" damping="2"/>
      <joint name="hook_slide_y" type="slide" axis="0 1 0" damping="2"/>
      <joint name="hook_slide_z" type="slide" axis="0 0 1" damping="2"/>

      <geom type="box"
            size="0.04 0.005 0.01"
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
    </body>
  </worldbody>

  <actuator>
    <motor name="hook_x_motor" joint="hook_slide" ctrlrange="-1 1" gear="5"/>
    <position name="hook_y_pos" joint="hook_slide_y" ctrlrange="-0.05 0.05" kp="50"/>
    <position name="hook_z_pos" joint="hook_slide_z" ctrlrange="-0.05 0.05" kp="50"/>
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


# Rewards



# get the block start position // TODO extract the start_pos with targe_block_pos() if the env is given
# block_start_pos = target_block_pos()
_START_BLOCK_POS = torch.tensor([0.0, 0.0505, 0.168])

def block_progress(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    block_current_pos = target_block_pos(env, asset_cfg)
    movement = block_current_pos - _START_BLOCK_POS

    extraction_direction = torch.Tensor([-1, 0, 0])
    progress = torch.sum(movement * extraction_direction, dim=-1)
    return progress



# Environment conifg


def _make_env_cfg() -> ManagerBasedRlEnvCfg:

    actor_terms = {
        "pusher_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _HOOK1_CFG}
        ),
        "pusher_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _HOOK1_CFG}
        ),
        "block_pos": ObservationTermCfg(
            func=target_block_pos,
            params={"asset_cfg": _TARGET_BLOCK_CFG}
        ),
        "block_vel": ObservationTermCfg(
            func=target_block_vel,
            params={"asset_cfg": _TARGET_BLOCK_CFG}
        )
    }


    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}),
    }


    #TODO Maybe swap effort (aka force) for velocity
    actions : dict[str, ActionTermCfg] = {
        "effort": JointEffortActionCfg(
            entity_name="hook",
            actuator_names=("hook_slide",),
            scale=1.0,
        ),
        "touch_y": RelativeJointPositionActionCfg(
            entity_name="hook",
            actuator_names=("hook_slide_y",),
            scale=1.0,
        ),
        "touch_z": RelativeJointPositionActionCfg(
            entity_name="hook",
            actuator_names=("hook_slide_z",),
            scale=1.0,
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
        "block_progress": RewardTermCfg(
            func=block_progress,
            weight=1.0,
        ),
        "torque_penalty": RewardTermCfg(
            func=joint_torques_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("hook", joint_names=("hook_slide",))},
        ),
        "action_rate": RewardTermCfg( #to prevent the hook from wild jumping
            func=action_rate_l2,
            weight=-0.001,
        )
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }


    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=_build_entities(),
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
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
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )




if __name__ == "__main__":
    entities = _build_entities()
    print(entities.keys())
    print(len(entities))
