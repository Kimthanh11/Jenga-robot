from __future__ import annotations

import math 
import torch
import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.envs import ManagerBasedRlEnv
import constants

def _rz_quat(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2))

def curriculum_step(env) -> int:
    """Return the curriculum clock persisted by the MJLab runner."""
    return env.common_step_counter

# Actuator utils
def _solref_attr() -> str:
    """XML attribute for BLOCK_SOLREF, or nothing when MuJoCo's default applies."""
    if constants.BLOCK_SOLREF is None:
        return ""
    return f' solref="{constants.BLOCK_SOLREF[0]:g} {constants.BLOCK_SOLREF[1]:g}"'


#block entities
def _vec(values) -> str:
    return " ".join(f"{v:g}" for v in values)

def _quat_from_z_rotation_deg(angle_deg: float) -> tuple[float, float, float, float]:
    angle = math.radians(angle_deg)
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))

def get_block_infos():
    import random

    rng = random.Random(0)
    block_infos = []

    for layer in range(1, constants.LAYERS + 1):
        for block in range(1, constants.BLOCKS_PER_LAYER + 1):
            z = constants.START_Z + (layer - 1) * constants.LAYER_HEIGHT

            if layer % 2 == 1:
                x_positions = [-constants.SIDE_SPACING, 0, constants.SIDE_SPACING]
                x = x_positions[block - 1] + rng.uniform(-0.0005, 0.0005)
                y = 0.0 + rng.uniform(-0.0005, 0.0005)
                yaw_noise = rng.uniform(-1.0, 1.0)
                quat = _quat_from_z_rotation_deg(0.0 + yaw_noise)
            else:
                y_positions = [constants.SIDE_SPACING, 0, -constants.SIDE_SPACING]
                x = 0.0 + rng.uniform(-0.0005, 0.0005)
                y = y_positions[block - 1] + rng.uniform(-0.0005, 0.0005)
                yaw_noise = rng.uniform(-1.0, 1.0)
                quat = _quat_from_z_rotation_deg(90.0 + yaw_noise)

            if layer % 2 == 1:
                color = constants.COLOR_A if block in (1, 3) else constants.COLOR_B
            else:
                color = constants.COLOR_B if block in (1, 3) else constants.COLOR_A

            sliding = rng.uniform(0.2, 0.4)
            torsional = rng.uniform(0.01, 0.06)
            friction = (sliding, torsional, 0.001)
            density = constants.BLOCK_DENSITY * rng.uniform(
                1.0 - constants.BLOCK_DENSITY_RANDOMIZATION,
                1.0 + constants.BLOCK_DENSITY_RANDOMIZATION,
            )
            size = tuple(
                nominal_size
                * rng.uniform(1.0 - randomization, 1.0 + randomization)
                for nominal_size, randomization in zip(
                    constants.BLOCK_SIZE,
                    constants.BLOCK_SIZE_RANDOMIZATION,
                    strict=True,
                )
            )

            block_infos.append({
                "name": f"b{layer}_{block}",
                "pos": (x, y, z),
                "quat": quat,
                "color": color,
                "friction": friction,
                "density": density,
                "half_size": tuple(value / 2 for value in size),
            })

    return block_infos

INITIAL_BLOCK_POS_BY_NAME = {
    block_info["name"]: torch.tensor(block_info["pos"], dtype=torch.float32)
    for block_info in get_block_infos()
}

def initial_block_pos(block_name: str) -> torch.Tensor:
    for block_info in get_block_infos():
        if block_info["name"] == block_name:
            return torch.tensor(block_info["pos"])
    raise ValueError(f"Unknown block name: {block_name}")

_START_REF_POS = (initial_block_pos("b6_2") + initial_block_pos("b6_3")) / 2
START_TARGET_REL_POS = initial_block_pos("b6_1") - _START_REF_POS

_INITIAL_BLOCK_POS_BY_NAME = {
    block_info["name"]: torch.tensor(block_info["pos"], dtype=torch.float32)
    for block_info in get_block_infos()
}

# loads the jenga_xml into an Mjspec, which is editable
def _spec_from_xml(xml: str) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(xml)

def _get_hook_spec() -> mujoco.MjSpec:
    xml = """
<mujoco model="hook">
  <compiler angle="degree" coordinate="local"/>

  <worldbody>
    <body name="hook" pos="0 0 0">
      <joint name="hook_yaw" type="hinge" axis="0 0 1" range="-60 150" limited="true" damping="2"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>

      <body name="hook_tool" pos="0 0 0">
        <joint name="hook_slide" type="slide" axis="1 0 0" range="-0.22 0.16" limited="true" damping="2"/>
        <joint name="hook_slide_y" type="slide" axis="0 1 0" range="-0.13 0.23" limited="true" damping="2"/>
        <joint name="hook_slide_z" type="slide" axis="0 0 1" range="-0.17 0.13" limited="true" damping="2"/>

        <geom type="box"
              size="0.04 0.005 0.006"
              pos="0 0 0"
              rgba="0.1 0.1 0.9 1"
              density="2000"/>

        <geom type="box"
              size="0.006 0.004 0.004"
              pos="-0.05 0 0"
              rgba="1 0 0 1"
              density="2000"/>
        <site name="hook_tip" pos="-0.056 0 0" size="0.003"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="hook_x_vel" joint="hook_slide" ctrlrange="-0.05 0.05" kv="150"/>    
    <position name="hook_y_pos" joint="hook_slide_y" ctrlrange="-0.13 0.23" kp="50"/>
    <position name="hook_z_pos" joint="hook_slide_z" ctrlrange="-0.17 0.13" kp="50"/>
    <position name="hook_yaw_pos" joint="hook_yaw" ctrlrange="-1.1 2.7" kp="20"/>
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
    <geom density="{block_info["density"]}"
            margin="0"
            gap="0"{_solref_attr()}/>
    </default>

  <worldbody>
    <body name="{block_info["name"]}" pos="{_vec(block_info["pos"])}" quat="{_vec(block_info["quat"])}">
      <joint name="{block_info["name"]}_free" type="free"/>

      <geom type="box"
            size="{_vec(block_info["half_size"])}"
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


def build_entities() -> dict[str, EntityCfg]:
    entities = {
        "hook": _get_hook_cfg(),
    }

    for block_info in get_block_infos():
        entities[block_info["name"]] = _get_block_cfg(block_info)

    return entities

def hook_joint_pos_ordered(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hook joints in policy order: slide, slide_y, slide_z, yaw."""
    asset: Entity = env.scene[constants._HOOK_ALL_CFG.name]
    joint_ids, _ = asset.find_joints(constants._HOOK_JOINT_ORDER, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids]

def hook_tip_pos(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = constants._HOOK_TIP_CFG) -> torch.Tensor:
    """
    get the position of the gripper
    """
    asset: Entity = env.scene[asset_cfg.name]
    hook_tip_position = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    return hook_tip_position.squeeze(1)

def _hook_contact_sensor(env: ManagerBasedRlEnv) -> ContactSensor:
    sensor = env.scene[constants.HOOK_CONTACT_SENSOR_NAME]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"{constants.HOOK_CONTACT_SENSOR_NAME} is not a ContactSensor")
    return sensor

def hook_contact_force_world(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Strongest net hook contact force observed during the current policy step."""
    data = _hook_contact_sensor(env).data
    if data.force is None:
        return torch.zeros(env.num_envs, 3, device=env.device)

    if data.force_history is None:
        return data.force[:, 0, :]

    history = data.force_history[:, 0, :, :]
    strongest_idx = torch.linalg.vector_norm(history, dim=-1).argmax(dim=1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    return history[env_ids, strongest_idx]

def _hook_contact_sensor(env: ManagerBasedRlEnv) -> ContactSensor:
    sensor = env.scene[constants.HOOK_CONTACT_SENSOR_NAME]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"{constants.HOOK_CONTACT_SENSOR_NAME} is not a ContactSensor")
    return sensor

def hook_contact_found(env: ManagerBasedRlEnv) -> torch.Tensor:
    data = _hook_contact_sensor(env).data
    current = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if data.found is not None:
        current = (data.found > 0).any(dim=1)
    if data.force_history is None:
        return current.float()
    history = torch.linalg.vector_norm(data.force_history, dim=-1) > 1.0e-8
    return (current | history.any(dim=(1, 2))).float()

def current_missing_block_mask(
    env: ManagerBasedRlEnv,
    block_name: str,
) -> torch.Tensor:
    mask = getattr(env, "_jenga_missing_block_mask", None)
    if mask is None or block_name not in constants.MISSING_BLOCK_CANDIDATES:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return mask[:, constants.MISSING_BLOCK_CANDIDATES.index(block_name)]

def get_com_tower(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = constants._TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    get the COM of all present tower blocks except the target block.
    """
    target_block_name = asset_cfg.name
    total_com = torch.zeros(env.num_envs, 3, device=env.device)
    present_count = torch.zeros(env.num_envs, device=env.device)

    for block in get_block_infos():
        block_name = block["name"]
        if block_name == target_block_name:
            continue

        asset: Entity = env.scene[block_name]
        block_com = asset.data.body_com_pos_w[:, 0, :]
        present = ~current_missing_block_mask(env, block_name)
        present_weight = present.to(dtype=block_com.dtype)
        total_com += block_com * present_weight.unsqueeze(-1)
        present_count += present_weight

    return total_com / present_count.clamp_min(1.0).unsqueeze(-1)

def normalized_limit_excess(
    value: torch.Tensor,
    safe_limit: float,
    damage_limit: float,
) -> torch.Tensor:
    return torch.clamp(
        (value - safe_limit) / (damage_limit - safe_limit),
        min=0.0,
        max=1.0,
    )

def ensure_missing_block_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "_jenga_missing_block_mask"):
        env._jenga_missing_block_mask = torch.zeros(
            env.num_envs,
            len(constants.MISSING_BLOCK_CANDIDATES),
            dtype=torch.bool,
            device=env.device,
        )
        env._jenga_missing_pattern_id = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    return env._jenga_missing_block_mask

def missing_block_max_count(env: ManagerBasedRlEnv) -> int:
    """Maximum number of missing blocks allowed at the current curriculum step."""

    if constants.FORCED_MISSING_BLOCK_COUNT is not None:
        return constants.FORCED_MISSING_BLOCK_COUNT

    if curriculum_step(env) >= constants.MISSING_BLOCK_TRIPLE_BEGIN_STEP:
        return 3

    if curriculum_step(env) >= constants.MISSING_BLOCK_DOUBLE_BEGIN_STEP:
        return 2

    if curriculum_step(env) >= constants.MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP:
        return 1

    return 0

def active_missing_pattern_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    if constants.FORCED_MISSING_PATTERN_IDS is not None:
        invalid = [
            idx
            for idx in constants.FORCED_MISSING_PATTERN_IDS
            if idx < 0 or idx >= len(constants.MISSING_BLOCK_PATTERNS)
        ]
        if invalid:
            raise ValueError(f"Invalid forced missing pattern ids: {invalid}")
        return torch.tensor(
            constants.FORCED_MISSING_PATTERN_IDS,
            dtype=torch.long,
            device=env.device,
        )

    max_count = missing_block_max_count(env)
    if constants.FORCED_MISSING_BLOCK_COUNT is None:
        active_ids = [
            idx
            for idx, pattern in enumerate(constants.MISSING_BLOCK_PATTERNS)
            if 0 < len(pattern) <= max_count
        ]
    else:
        active_ids = [
            idx
            for idx, pattern in enumerate(constants.MISSING_BLOCK_PATTERNS)
            if len(pattern) == max_count
        ]
    return torch.tensor(active_ids, dtype=torch.long, device=env.device)

def missing_block_randomization_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Probability that a reset uses a non-empty missing-block pattern."""
    progress = min(
        max(
            curriculum_step(env) - constants.MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP,
            0,
        )
        / constants.MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS,
        1.0,
    )
    probability = constants.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY + progress * (
        constants.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY
        - constants.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY
    )
    return torch.tensor(probability, device=env.device)