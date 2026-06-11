from pathlib import Path
from mjlab.managers.scene_entity_config import SceneEntityCfg
import mujoco
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# get the configurations
_JENGA_XML = Path(__file__).parent / "jenga.xml"
_HOOK1_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide",))
_HOOK2_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide2",))
_HOOK3_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide3",))


# loads the jenga_xml into an Mjspec, which is editable
def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_JENGA_XML))


# tells mjlab those actuators are there (probably relevant for RL)
_JENGA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(target_names_expr=("hook_slide",)),
        XmlActuatorCfg(target_names_expr=("hook_slide2",)),
        XmlActuatorCfg(target_names_expr=("hook_slide3",)),
    ),
)

# blueprint for the Jenga-Entity (where is the model from and what are the actuators)
def _get_jenga_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_JENGA_ARTICULATION,
    )



















if __name__ == "__main__":
    spec = _get_spec()
    print(f"Loaded MuJoCo spec from {_JENGA_XML}")
    print(f"Model name: {spec.modelname}")
