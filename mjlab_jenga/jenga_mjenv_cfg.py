from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
import math

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply
from mjlab.terrains import TerrainEntityCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  time_out,
)
from mjlab.envs.mdp.dr import geom_friction, pseudo_inertia
from mjlab.envs.mdp.rewards import action_rate_l2
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
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
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Tower Configurations
LAYERS = 9
BLOCKS_PER_LAYER = 3

BLOCK_SIZE = (0.05, 0.15, 0.03)
BLOCK_HALF_SIZE = tuple(v / 2 for v in BLOCK_SIZE)
# Per-block build-time domain randomization. Keep this small: larger shape
# variation can create unrealistic overlaps in the stacked tower.
BLOCK_DENSITY = 650.0
BLOCK_DENSITY_RANDOMIZATION = 0.0
BLOCK_SIZE_RANDOMIZATION = (0.0, 0.0, 0.0)
RESET_DENSITY_RANDOMIZATION = 0.15
RESET_FRICTION_SLIDING_RANGE = (0.28, 0.48)
RESET_FRICTION_TORSIONAL_RANGE = (0.012, 0.055)
RESET_FRICTION_ROLLING_RANGE = (0.001, 0.001)
TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT = 0.012
TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT = 0.008
# Out-of-plane tilt of any non-target block. These limits always described tipping;
# the measurement, not the numbers, was wrong.
TOWER_SUCCESS_MAX_BLOCK_ROTATION = math.radians(8.0)
TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT = 0.025
TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT = 0.015
TOWER_DAMAGE_MAX_BLOCK_ROTATION = math.radians(15.0)
CONTACT_X_LIMIT = 0.01
CONTACT_Y_LIMIT = BLOCK_HALF_SIZE[1]
CONTACT_Z_LIMIT = 0.006
CONTACT_FACE_Y = -CONTACT_Y_LIMIT
PUSH_X_VELOCITY_SCALE = 0.03
PUSH_X_VELOCITY_CLIP = (-0.05, 0.05)
PUSH_ACTION_DEADZONE = 0.08
PUSH_VELOCITY_CHANGE_PER_STEP = 0.006
HOOK_CONTACT_SENSOR_NAME = "hook_contact"
CONTACT_FORCE_OBS_NORMALIZER = 5.0
CONTACT_FORCE_OBS_CLIP = 2.0

SIDE_SPACING = BLOCK_SIZE[0] + 0.0005
START_Z = (BLOCK_SIZE[2] / 2) + 0.0005
LAYER_HEIGHT = BLOCK_SIZE[2] + 0.0005

# Every curriculum is a function of env.common_step_counter, which restarts at 0 on a
# resumed run -- so continuing a training would silently rewind the difficulty to its
# starting value. Setting this offset makes a resumed run pick the curriculum up where
# the previous one stopped.
CURRICULUM_STEP_OFFSET = 0

# Horizontal block displacement is measured against the nominal spawn pose, so a tower
# that slides across the floor as one rigid piece counts as damaged even though nothing
# about it came apart. With this enabled the bottom layer's drift is subtracted first,
# making the measure "how far has this block moved relative to the tower's base" --
# shear and blocks sliding out of their layer still count, rigid translation does not.
# Off by default until the A/B says it helps. Layer 1 is never a target and never a
# missing-block candidate, so the base reference is always intact.
TOWER_SHIFT_RELATIVE_TO_BASE = False

# Contact softness of the block geoms, as MuJoCo solref = (timeconst, dampratio).
# None keeps MuJoCo's default (0.02, 1). Lower timeconst means a stiffer contact;
# MuJoCo clamps it to at least 2 * timestep, so 0.004 is the floor here.
#
# This is the remaining lever on the drag ratio. impratio took it from 0.81 to 0.19 by
# stiffening the FRICTION constraints; solref governs how far the contacts deform in
# the first place. Success at full extraction needs drag below 0.107 (12 mm of
# neighbour movement over 112.5 mm of target movement), and the per-target drags
# predict the sweep outcomes exactly: b3_1 at 0.098 succeeds, b6_3 at 0.149, b6_1 at
# 0.181 and b7_1 at 0.252 all fail.
BLOCK_SOLREF: tuple[float, float] | None = None


def curriculum_step(env) -> int:
    """Curriculum clock: wall step count plus the resume offset."""
    return env.common_step_counter + CURRICULUM_STEP_OFFSET


MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 600_000
MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.05
MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.35
MISSING_BLOCK_DOUBLE_BEGIN_STEP = 250_000
MISSING_BLOCK_TRIPLE_BEGIN_STEP = 500_000
FORCED_MISSING_BLOCK_COUNT: int | None = None
MISSING_BLOCK_PARK_OFFSET = (1.5, 1.5, 0.5)
MISSING_BLOCK_PARK_SPACING = 0.2
RANDOM_TARGET_BLOCK_BEGIN_STEP = 0
RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
RANDOM_TARGET_BLOCK_START_PROBABILITY = 1.0
RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 0
RANDOM_TARGET_WITH_MISSING_RAMP_STEPS = 1
RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 1.0
RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 1.0
FIXED_TARGET_BLOCK_NAME = "b6_1"
# b1_1 and b1_3 are deliberately absent: the scripted full-push controller cannot
# extract them. They end in step_cap rather than damage -- the tower holds, the block
# simply stops after ~25 mm with the actuator at 5-6 N, at or above its stall force.
# Layer 1 carries the whole tower, so this is a real force limit, not a policy failure,
# and keeping them would hand PPO episodes it cannot win. Revisit if the push actuator
# is ever given a larger force budget.
RANDOM_TARGET_BLOCK_NAMES = (
    "b2_1",
    "b2_2",
    "b2_3",
    "b3_1",
    "b6_1",
    "b6_2",
    "b6_3",
    "b7_1",
    "b7_3",
    "b9_1",
    "b9_2",
    "b9_3",
)
HOOK_BASE_POS = (0.15, 0.05, 0.16)
HOOK_TIP_LOCAL_X = -0.056
HOOK_APPROACH_GAP = 0.02
HOOK_BOTTOM_LAYER_Z_LIFT = 0.006

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
    preserve_order=True,
)
_HOOK_TIP_CFG = SceneEntityCfg("hook", site_names=("hook_tip",))
_HOOK_JOINT_ORDER = ("hook_slide", "hook_slide_y", "hook_slide_z", "hook_yaw")



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
            density = BLOCK_DENSITY * rng.uniform(
                1.0 - BLOCK_DENSITY_RANDOMIZATION,
                1.0 + BLOCK_DENSITY_RANDOMIZATION,
            )
            size = tuple(
                nominal_size
                * rng.uniform(1.0 - randomization, 1.0 + randomization)
                for nominal_size, randomization in zip(
                    BLOCK_SIZE,
                    BLOCK_SIZE_RANDOMIZATION,
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


MISSING_BLOCK_SINGLE_PATTERNS = (
    ("b4_1",),
    ("b4_3",),
    ("b5_2",),
)
MISSING_BLOCK_DOUBLE_PATTERNS = (
    ("b4_1", "b5_2"),
    ("b4_3", "b5_2"),
    ("b3_2", "b5_1"),
    ("b3_3", "b5_3"),
)
MISSING_BLOCK_TRIPLE_PATTERNS = (
    ("b3_2", "b4_1", "b5_2"),
    ("b3_3", "b4_3", "b5_2"),
    ("b4_1", "b5_2", "b8_2"),
)
MISSING_BLOCK_PATTERNS = (
    (),
    *MISSING_BLOCK_SINGLE_PATTERNS,
    *MISSING_BLOCK_DOUBLE_PATTERNS,
    *MISSING_BLOCK_TRIPLE_PATTERNS,
)
MISSING_BLOCK_CANDIDATES = tuple(
    sorted({block_name for pattern in MISSING_BLOCK_PATTERNS for block_name in pattern})
)
_INITIAL_BLOCK_POS_BY_NAME = {
    block_info["name"]: torch.tensor(block_info["pos"], dtype=torch.float32)
    for block_info in _get_block_infos()
}


def random_target_block_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Probability that a reset uses a random target block instead of b6_1."""
    progress = (
        max(
            curriculum_step(env) - RANDOM_TARGET_BLOCK_BEGIN_STEP,
            0,
        )
        / RANDOM_TARGET_BLOCK_RAMP_STEPS
    )
    probability = RANDOM_TARGET_BLOCK_START_PROBABILITY + min(progress, 1.0) * (
        RANDOM_TARGET_BLOCK_END_PROBABILITY - RANDOM_TARGET_BLOCK_START_PROBABILITY
    )
    return torch.tensor(
        probability,
        device=env.device,
    )


def random_target_with_missing_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Probability that random targets are allowed in already incomplete towers."""
    progress = (
        max(
            curriculum_step(env) - RANDOM_TARGET_WITH_MISSING_BEGIN_STEP,
            0,
        )
        / RANDOM_TARGET_WITH_MISSING_RAMP_STEPS
    )
    probability = RANDOM_TARGET_WITH_MISSING_START_PROBABILITY + min(
        progress,
        1.0,
    ) * (
        RANDOM_TARGET_WITH_MISSING_END_PROBABILITY
        - RANDOM_TARGET_WITH_MISSING_START_PROBABILITY
    )
    return torch.tensor(probability, device=env.device)


def _rz_quat(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2))


def _target_block_entries() -> tuple[list[str], list[dict]]:
    entries = []
    names = []
    long_half = BLOCK_SIZE[1] / 2
    tip_offset = -HOOK_TIP_LOCAL_X

    for block_info in _get_block_infos():
        name = block_info["name"]
        names.append(name)
        cx, cy, cz = block_info["pos"]
        layer, slot = (int(part) for part in name[1:].split("_"))
        even_layer = layer % 2 == 0
        hook_center_z = cz + (HOOK_BOTTOM_LAYER_Z_LIFT if layer == 1 else 0.0)

        if even_layer:
            yaw_home = 0.0
            extraction_w = (-1.0, 0.0, 0.0)
            contact_face_y = CONTACT_FACE_Y
            slide_home = cx + long_half + HOOK_APPROACH_GAP + tip_offset - HOOK_BASE_POS[0]
            slide_y_home = cy - HOOK_BASE_POS[1]
            task_quat = _rz_quat(math.pi)
        else:
            yaw_home = math.pi / 2
            extraction_w = (0.0, -1.0, 0.0)
            contact_face_y = CONTACT_Y_LIMIT
            slide_home = cy + long_half + HOOK_APPROACH_GAP + tip_offset - HOOK_BASE_POS[1]
            slide_y_home = HOOK_BASE_POS[0] - cx
            task_quat = _rz_quat(-math.pi / 2)

        entries.append(
            {
                "name": name,
                "layer": layer,
                "slot": slot,
                "start_pos": (cx, cy, cz),
                "start_quat": block_info["quat"],
                "extraction_w": extraction_w,
                "contact_face_y": contact_face_y,
                "task_quat": task_quat,
                # Order matches our hook joint order: slide, slide_y, slide_z, yaw.
                "hook_home": (
                    slide_home,
                    slide_y_home,
                    hook_center_z - HOOK_BASE_POS[2],
                    yaw_home,
                ),
            }
        )

    by_layer: dict[int, list[int]] = {}
    for idx, entry in enumerate(entries):
        by_layer.setdefault(entry["layer"], []).append(idx)

    for idx, entry in enumerate(entries):
        entry["neighbors"] = [other for other in by_layer[entry["layer"]] if other != idx]

    # Fixed 8 slots describing the target's structural surroundings:
    #   [same-layer x2, layer above x3, layer below x3]
    # Slot meaning is identical for every target, so the policy can read them the same
    # way regardless of which block it was assigned. Slots that cannot exist (no layer
    # above for layer 9, none below for layer 1) are marked invalid rather than absent.
    for idx, entry in enumerate(entries):
        layer = entry["layer"]
        groups = (
            ([other for other in by_layer[layer] if other != idx], 2),
            (by_layer.get(layer + 1, []), 3),
            (by_layer.get(layer - 1, []), 3),
        )
        support_idx, support_valid = [], []
        for members, width in groups:
            for slot in range(width):
                if slot < len(members):
                    support_idx.append(members[slot])
                    support_valid.append(1.0)
                else:
                    support_idx.append(0)
                    support_valid.append(0.0)
        entry["support_idx"] = support_idx
        entry["support_valid"] = support_valid

    return names, entries


class TargetBlockCommand(CommandTerm):
    """Curriculum command for target-block selection and hook teleport.

    At the beginning, every env keeps the old fixed target b6_1. Once
    random_target_block_scale becomes non-zero, some reset envs sample a safe target
    block and the hook is teleported in front of that block's push face.
    """

    cfg: "TargetBlockCommandCfg"

    def __init__(self, cfg: "TargetBlockCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        names, entries = _target_block_entries()
        self._all_names = names
        self._blocks = [env.scene[name] for name in names]
        self._num_blocks = len(names)
        self._env_arange = torch.arange(self.num_envs, device=self.device)

        self._start_pos = torch.tensor(
            [entry["start_pos"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._start_quat = torch.tensor(
            [entry["start_quat"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._extraction = torch.tensor(
            [entry["extraction_w"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._contact_face_y = torch.tensor(
            [entry["contact_face_y"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._task_quat = torch.tensor(
            [entry["task_quat"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._hook_home = torch.tensor(
            [entry["hook_home"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._target_features = torch.tensor(
            [
                (
                    (entry["layer"] - 1) / max(LAYERS - 1, 1),
                    entry["slot"] - 2,
                    1.0 if entry["layer"] % 2 == 0 else -1.0,
                )
                for entry in entries
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._neighbor_idx = torch.tensor(
            [entry["neighbors"] for entry in entries],
            dtype=torch.long,
            device=self.device,
        )
        self._base_idx = torch.tensor(
            [idx for idx, entry in enumerate(entries) if entry["layer"] == 1],
            dtype=torch.long,
            device=self.device,
        )
        self._support_idx = torch.tensor(
            [entry["support_idx"] for entry in entries],
            dtype=torch.long,
            device=self.device,
        )
        self._support_valid = torch.tensor(
            [entry["support_valid"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._cur_present = torch.ones(
            self.num_envs, len(names), dtype=torch.bool, device=self.device
        )

        name_to_idx = {name: idx for idx, name in enumerate(names)}
        self._fixed_idx = name_to_idx[cfg.fixed_target_name]
        self._force_target_idx: int | None = None
        if cfg.force_target_name is not None:
            if cfg.force_target_name not in name_to_idx:
                raise ValueError(f"Unknown forced target block: {cfg.force_target_name}")
            self._force_target_idx = name_to_idx[cfg.force_target_name]
        self._force_target_per_env: torch.Tensor | None = None
        if cfg.force_target_names:
            unknown = [n for n in cfg.force_target_names if n not in name_to_idx]
            if unknown:
                raise ValueError(f"Unknown forced target blocks: {unknown}")
            cycled = [
                name_to_idx[cfg.force_target_names[i % len(cfg.force_target_names)]]
                for i in range(self.num_envs)
            ]
            self._force_target_per_env = torch.tensor(
                cycled, dtype=torch.long, device=self.device
            )
        selectable = [
            name_to_idx[name]
            for name in cfg.selectable_target_names
            if name in name_to_idx and name not in MISSING_BLOCK_CANDIDATES
        ]
        self._selectable = torch.tensor(selectable, dtype=torch.long, device=self.device)
        self._num_selectable = int(self._selectable.numel())
        if self._num_selectable == 0:
            raise ValueError("TargetBlockCommand needs at least one selectable block.")

        self._hook = env.scene["hook"]
        hook_home_joint_ids, _ = self._hook.find_joints(
            ("hook_slide", "hook_slide_y", "hook_slide_z", "hook_yaw"),
            preserve_order=True,
        )
        self._hook_home_joint_ids = torch.tensor(
            hook_home_joint_ids,
            dtype=torch.long,
            device=self.device,
        )

        self.selected_block_idx = torch.full(
            (self.num_envs,),
            self._fixed_idx,
            dtype=torch.long,
            device=self.device,
        )
        self.selected_is_random = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._cur_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._cur_target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._cur_target_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._cur_ref_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._cur_movement_rel = torch.zeros(self.num_envs, 3, device=self.device)
        self._cur_progress = torch.zeros(self.num_envs, device=self.device)
        self._cur_tower_shift = torch.zeros(self.num_envs, device=self.device)
        self._cur_max_block_horizontal_shift = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._cur_max_block_vertical_shift = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._cur_max_block_rotation = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.metrics["selected_block"] = self.selected_block_idx.float()
        self.metrics["random_target"] = self.selected_is_random.float()
        self.metrics["progress"] = self._cur_progress

    @property
    def command(self) -> torch.Tensor:
        return self.selected_extraction_w()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        count = weights.sum(dim=1)
        mean = (values * weights).sum(dim=1) / count.clamp_min(1.0)
        return torch.where(count > 0, mean, torch.zeros_like(mean))

    def _present_by_block(self) -> torch.Tensor:
        present = torch.ones(
            self.num_envs,
            self._num_blocks,
            dtype=torch.bool,
            device=self.device,
        )
        for block_idx, block_name in enumerate(self._all_names):
            present[:, block_idx] = ~current_missing_block_mask(self._env, block_name)
        return present

    def selected_target_pos_w(self) -> torch.Tensor:
        return self._cur_target_pos

    def selected_target_vel_w(self) -> torch.Tensor:
        return self._cur_target_vel

    def selected_target_pose_w(self) -> torch.Tensor:
        return self._cur_target_pose

    def selected_ref_pos_w(self) -> torch.Tensor:
        return self._cur_ref_pos

    def selected_relative_movement(self) -> torch.Tensor:
        return self._cur_movement_rel

    def selected_progress(self) -> torch.Tensor:
        return self._cur_progress

    def selected_tower_shift(self) -> torch.Tensor:
        return self._cur_tower_shift

    def selected_max_block_horizontal_shift(self) -> torch.Tensor:
        return self._cur_max_block_horizontal_shift

    def selected_max_block_vertical_shift(self) -> torch.Tensor:
        return self._cur_max_block_vertical_shift

    def selected_max_block_rotation(self) -> torch.Tensor:
        return self._cur_max_block_rotation

    def selected_support_presence(self) -> torch.Tensor:
        """Presence of the blocks structurally around the target, per fixed slot.

        1 = present, 0 = removed, -1 = the slot cannot exist for this target. The three
        values are distinct on purpose: "there is no layer below me" is a different
        situation from "the block below me was taken away".
        """
        idx = self._support_idx[self.selected_block_idx]
        valid = self._support_valid[self.selected_block_idx]
        present = torch.gather(self._cur_present.float(), 1, idx)
        return present * valid + (valid - 1.0)

    def selected_extraction_w(self) -> torch.Tensor:
        return self._extraction[self.selected_block_idx]

    def selected_contact_face_y(self) -> torch.Tensor:
        return self._contact_face_y[self.selected_block_idx]

    def selected_hook_home(self) -> torch.Tensor:
        return self._hook_home[self.selected_block_idx]

    def selected_task_quat_w(self) -> torch.Tensor:
        return self._task_quat[self.selected_block_idx]

    def selected_target_features(self) -> torch.Tensor:
        target_features = self._target_features[self.selected_block_idx]
        random_flag = self.selected_is_random.to(dtype=torch.float32).unsqueeze(-1)
        return torch.cat((target_features, random_flag), dim=-1)

    def selected_block_count_summary(self, max_items: int = 6) -> str:
        unique, counts = torch.unique(self.selected_block_idx, return_counts=True)
        order = torch.argsort(counts, descending=True)
        items = []
        for item_idx in order[:max_items].tolist():
            block_idx = int(unique[item_idx].item())
            count = int(counts[item_idx].item())
            items.append(f"{self._all_names[block_idx]}:{count}")
        if unique.numel() > max_items:
            items.append("...")
        return ",".join(items)

    def _update_metrics(self) -> None:
        all_pos = torch.stack(
            [block.data.body_link_pos_w[:, 0, :] for block in self._blocks],
            dim=0,
        )
        all_vel = torch.stack(
            [block.data.body_link_vel_w[:, 0, :] for block in self._blocks],
            dim=0,
        )
        all_pose = torch.stack(
            [block.data.body_link_pose_w[:, 0, :] for block in self._blocks],
            dim=0,
        )
        selected = self.selected_block_idx
        env_ids = self._env_arange
        self._cur_target_pos = all_pos[selected, env_ids]
        self._cur_target_vel = all_vel[selected, env_ids]
        self._cur_target_pose = all_pose[selected, env_ids]

        neighbor_idx = self._neighbor_idx[selected]
        present_by_block = self._present_by_block()
        self._cur_present = present_by_block
        neighbor_present = present_by_block[env_ids.unsqueeze(1), neighbor_idx]
        neighbor_pos = all_pos[neighbor_idx, env_ids.unsqueeze(1)]
        neighbor_start = self._start_pos[neighbor_idx]
        ref_pos = self._masked_mean(neighbor_pos, neighbor_present)
        start_ref_pos = self._masked_mean(neighbor_start, neighbor_present)
        start_target_rel = self._start_pos[selected] - start_ref_pos

        self._cur_ref_pos = ref_pos
        self._cur_movement_rel = (self._cur_target_pos - ref_pos) - start_target_rel
        self._cur_progress = torch.sum(
            self._cur_movement_rel * self.selected_extraction_w(),
            dim=-1,
        )

        present_for_com = present_by_block.clone()
        present_for_com[env_ids, selected] = False
        weights = present_for_com.to(all_pos.dtype).transpose(0, 1).unsqueeze(-1)
        current_com = (all_pos * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
        start_weights = present_for_com.to(self._start_pos.dtype).unsqueeze(-1)
        start_com = (
            self._start_pos.unsqueeze(0) * start_weights
        ).sum(dim=1) / start_weights.sum(dim=1).clamp_min(1.0)
        self._cur_tower_shift = torch.norm((current_com - start_com)[:, :2], dim=-1)

        stability_mask = present_for_com.transpose(0, 1)
        position_delta = all_pos - self._start_pos.unsqueeze(1)
        if TOWER_SHIFT_RELATIVE_TO_BASE:
            base_drift = (
                all_pos[self._base_idx].mean(dim=0)
                - self._start_pos[self._base_idx].mean(dim=0).unsqueeze(0)
            )
            position_delta = position_delta - base_drift.unsqueeze(0)
        horizontal_shift = torch.norm(position_delta[:, :, :2], dim=-1)
        vertical_shift = torch.abs(position_delta[:, :, 2])
        # Tilt, not total rotation. The previous form took the full quaternion angle
        # against the spawn pose, which is dominated by rotation about the vertical
        # axis: measured 77-96% yaw (b6_1 3.62 deg total of which 0.15 deg tilt, b7_1
        # 2.80/0.12). A Jenga block that turns in its own plane still lies flat and
        # still carries the layer above it -- that is not instability. Tipping is.
        # Conflating them made the stability gate fire on a harmless quantity: training
        # stalled at max_block_rot_deg_mean 6.7 against an 8 deg limit while actual tilt
        # was around 0.2 deg.
        #
        # Blocks spawn flat, so measuring against world +Z rather than the spawn
        # quaternion is equivalent and cheaper: for q = (w, x, y, z) the z-component of
        # R*[0,0,1] is 1 - 2*(x^2 + y^2).
        current_quat = all_pose[:, :, 3:7]
        up_z = 1.0 - 2.0 * (
            current_quat[:, :, 1] ** 2 + current_quat[:, :, 2] ** 2
        )
        rotation = torch.acos(up_z.clamp(-1.0, 1.0))
        zeros = torch.zeros_like(horizontal_shift)
        self._cur_max_block_horizontal_shift = torch.where(
            stability_mask,
            horizontal_shift,
            zeros,
        ).max(dim=0).values
        self._cur_max_block_vertical_shift = torch.where(
            stability_mask,
            vertical_shift,
            zeros,
        ).max(dim=0).values
        self._cur_max_block_rotation = torch.where(
            stability_mask,
            rotation,
            zeros,
        ).max(dim=0).values

        self.metrics["selected_block"] = selected.float()
        self.metrics["random_target"] = self.selected_is_random.float()
        self.metrics["progress"] = self._cur_progress

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        if self._force_target_per_env is not None:
            selected = self._force_target_per_env[env_ids]
            use_random = torch.ones(num_resets, dtype=torch.bool, device=self.device)
        elif self._force_target_idx is None:
            random_probability = random_target_block_scale(self._env)
            use_random = torch.rand(num_resets, device=self.device) < random_probability
            present_by_block = self._present_by_block()
            envs_without_missing = present_by_block[env_ids].all(dim=1)
            allow_random_with_missing = (
                torch.rand(num_resets, device=self.device)
                < random_target_with_missing_scale(self._env)
            )
            use_random &= envs_without_missing | allow_random_with_missing
            selected = torch.full(
                (num_resets,),
                self._fixed_idx,
                dtype=torch.long,
                device=self.device,
            )
            if torch.any(use_random):
                random_choices = torch.randint(
                    0,
                    self._num_selectable,
                    (int(use_random.sum().item()),),
                    device=self.device,
                )
                selected[use_random] = self._selectable[random_choices]
        else:
            use_random = torch.ones(num_resets, dtype=torch.bool, device=self.device)
            selected = torch.full(
                (num_resets,),
                self._force_target_idx,
                dtype=torch.long,
                device=self.device,
            )

        self.selected_block_idx[env_ids] = selected
        self.selected_is_random[env_ids] = use_random

        target = self._hook_home[selected].clone()
        target[:, :3] += torch.empty_like(target[:, :3]).uniform_(-0.002, 0.002)
        target[:, 3] += torch.empty(
            target.shape[0],
            device=self.device,
            dtype=target.dtype,
        ).uniform_(-0.02, 0.02)
        self._hook.write_joint_position_to_sim(
            target,
            joint_ids=self._hook_home_joint_ids,
            env_ids=env_ids,
        )
        self._hook.write_joint_velocity_to_sim(
            torch.zeros_like(target),
            joint_ids=self._hook_home_joint_ids,
            env_ids=env_ids,
        )
        if self.cfg.debug_target_reset:
            self._print_reset_debug(env_ids, selected, use_random, target)

    def _update_command(self) -> None:
        pass

    def _print_reset_debug(
        self,
        env_ids: torch.Tensor,
        selected: torch.Tensor,
        use_random: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        max_items = min(8, int(env_ids.numel()))
        for row in range(max_items):
            block_idx = int(selected[row].item())
            face_y = float(self._contact_face_y[block_idx].item())
            target_name = self._all_names[block_idx]
            layer, slot = (int(part) for part in target_name[1:].split("_"))
            expected_z = HOOK_BOTTOM_LAYER_Z_LIFT if layer == 1 else 0.0
            expected_y = face_y + math.copysign(HOOK_APPROACH_GAP, face_y)
            home = target[row].detach().cpu().tolist()
            print(
                "DEBUG_TARGET_RESET",
                f"env={int(env_ids[row].item())}",
                f"target={target_name}",
                f"layer_slot=({layer},{slot})",
                f"random={bool(use_random[row].item())}",
                f"home_slide_y_z_yaw=({home[0]:.5f},{home[1]:.5f},{home[2]:.5f},{home[3]:.5f})",
                f"face_y={face_y:.5f}",
                f"expected_tip_block=(0.00000,{expected_y:.5f},{expected_z:.5f})",
                flush=True,
            )


@dataclass(kw_only=True)
class TargetBlockCommandCfg(CommandTermCfg):
    fixed_target_name: str = FIXED_TARGET_BLOCK_NAME
    selectable_target_names: tuple[str, ...] = RANDOM_TARGET_BLOCK_NAMES
    force_target_name: str | None = None
    force_target_names: tuple[str, ...] = ()
    """Per-env forced targets, cycled over the envs. Lets one vectorized rollout cover
    several target blocks at once (feasibility sweeps). Takes precedence over
    ``force_target_name``; empty means "not used"."""
    debug_target_reset: bool = False

    def build(self, env: ManagerBasedRlEnv) -> TargetBlockCommand:
        return TargetBlockCommand(self, env)


def _target_command_or_none(env: ManagerBasedRlEnv) -> TargetBlockCommand | None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None
    try:
        return command_manager.get_term("target_block")
    except Exception:
        return None


def missing_block_randomization_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Probability that a reset uses a non-empty missing-block pattern."""
    progress = min(
        max(
            curriculum_step(env) - MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP,
            0,
        )
        / MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS,
        1.0,
    )
    probability = MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY + progress * (
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY
        - MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY
    )
    return torch.tensor(probability, device=env.device)


def missing_block_max_count(env: ManagerBasedRlEnv) -> int:
    """Maximum number of missing blocks allowed at the current curriculum step."""
    if FORCED_MISSING_BLOCK_COUNT is not None:
        return FORCED_MISSING_BLOCK_COUNT
    if curriculum_step(env) >= MISSING_BLOCK_TRIPLE_BEGIN_STEP:
        return 3
    if curriculum_step(env) >= MISSING_BLOCK_DOUBLE_BEGIN_STEP:
        return 2
    if curriculum_step(env) >= MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP:
        return 1
    return 0


def _active_missing_pattern_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    max_count = missing_block_max_count(env)
    if FORCED_MISSING_BLOCK_COUNT is None:
        active_ids = [
            idx
            for idx, pattern in enumerate(MISSING_BLOCK_PATTERNS)
            if 0 < len(pattern) <= max_count
        ]
    else:
        active_ids = [
            idx
            for idx, pattern in enumerate(MISSING_BLOCK_PATTERNS)
            if len(pattern) == max_count
        ]
    return torch.tensor(active_ids, dtype=torch.long, device=env.device)


def _ensure_missing_block_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "_jenga_missing_block_mask"):
        env._jenga_missing_block_mask = torch.zeros(
            env.num_envs,
            len(MISSING_BLOCK_CANDIDATES),
            dtype=torch.bool,
            device=env.device,
        )
        env._jenga_missing_pattern_id = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    return env._jenga_missing_block_mask


def current_missing_block_mask(
    env: ManagerBasedRlEnv,
    block_name: str,
) -> torch.Tensor:
    mask = getattr(env, "_jenga_missing_block_mask", None)
    if mask is None or block_name not in MISSING_BLOCK_CANDIDATES:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return mask[:, MISSING_BLOCK_CANDIDATES.index(block_name)]


def missing_pattern_count_summary(env: ManagerBasedRlEnv, max_items: int = 6) -> str:
    pattern_ids = getattr(env, "_jenga_missing_pattern_id", None)
    if pattern_ids is None:
        return "none"

    unique, counts = torch.unique(pattern_ids, return_counts=True)
    order = torch.argsort(counts, descending=True)
    items = []
    for item_idx in order[:max_items].tolist():
        pattern_idx = int(unique[item_idx].item())
        count = int(counts[item_idx].item())
        pattern = MISSING_BLOCK_PATTERNS[pattern_idx]
        label = "none" if len(pattern) == 0 else "+".join(pattern)
        items.append(f"{label}:{count}")
    if unique.numel() > max_items:
        items.append("...")
    return ",".join(items)


def randomize_missing_blocks(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    num_resets = len(env_ids)
    pattern_ids = torch.zeros(num_resets, dtype=torch.long, device=env.device)
    missing_probability = missing_block_randomization_scale(env)
    active_pattern_ids = _active_missing_pattern_ids(env)

    if active_pattern_ids.numel() > 0 and missing_probability.item() > 0.0:
        use_missing_pattern = torch.rand(num_resets, device=env.device) < missing_probability
        num_missing_patterns = int(use_missing_pattern.sum().item())
        if num_missing_patterns > 0:
            active_choice_ids = torch.randint(
                0,
                active_pattern_ids.numel(),
                (num_missing_patterns,),
                device=env.device,
            )
            pattern_ids[use_missing_pattern] = active_pattern_ids[active_choice_ids]

    missing_mask = _ensure_missing_block_state(env)
    missing_mask[env_ids] = False
    env._jenga_missing_pattern_id[env_ids] = pattern_ids

    base_park_offset = torch.tensor(
        MISSING_BLOCK_PARK_OFFSET,
        device=env.device,
        dtype=torch.float32,
    )

    for candidate_idx, block_name in enumerate(MISSING_BLOCK_CANDIDATES):
        missing_for_block = torch.zeros(num_resets, dtype=torch.bool, device=env.device)
        for pattern_idx, pattern in enumerate(MISSING_BLOCK_PATTERNS):
            if block_name in pattern:
                missing_for_block |= pattern_ids == pattern_idx

        missing_mask[env_ids, candidate_idx] = missing_for_block

        asset: Entity = env.scene[block_name]
        root_state = asset.data.default_root_state[env_ids].clone()
        park_offset = base_park_offset.clone()
        park_offset[0] += MISSING_BLOCK_PARK_SPACING * candidate_idx
        root_state[missing_for_block, :3] += park_offset
        root_state[:, 7:] = 0.0
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)


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



def _solref_attr() -> str:
    """XML attribute for BLOCK_SOLREF, or nothing when MuJoCo's default applies."""
    if BLOCK_SOLREF is None:
        return ""
    return f' solref="{BLOCK_SOLREF[0]:g} {BLOCK_SOLREF[1]:g}"'


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


def _density_alpha_range() -> tuple[float, float]:
    low = 0.5 * math.log(1.0 - RESET_DENSITY_RANDOMIZATION)
    high = 0.5 * math.log(1.0 + RESET_DENSITY_RANDOMIZATION)
    return low, high


@requires_model_fields(
    "geom_friction",
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    recompute=RecomputeLevel.set_const,
)
def randomize_block_physics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
) -> None:
    if env_ids is not None and isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    friction_ranges = {
        0: RESET_FRICTION_SLIDING_RANGE,
        1: RESET_FRICTION_TORSIONAL_RANGE,
        2: RESET_FRICTION_ROLLING_RANGE,
    }
    alpha_range = _density_alpha_range()

    for block_info in _get_block_infos():
        block_name = block_info["name"]
        geom_friction(
            env,
            env_ids,
            friction_ranges,
            asset_cfg=SceneEntityCfg(block_name),
            axes=[0, 1, 2],
            operation="abs",
        )
        pseudo_inertia(
            env,
            env_ids,
            alpha_range=alpha_range,
            asset_cfg=SceneEntityCfg(block_name, body_names=(block_name,)),
        )

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


# Observations


#custom reward functions for the position/velocity of the target block position
def target_block_pos(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_target_pos_w()

    asset: Entity = env.scene[asset_cfg.name]
    position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    return position.squeeze(1)

def target_block_vel(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_target_vel_w()[:, :3]

    asset: Entity = env.scene[asset_cfg.name]
    velocity = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :]
    return velocity.squeeze(1)[:, :3]


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
    get the COM of all present tower blocks except the target block.
    """
    target_block_name = asset_cfg.name
    total_com = torch.zeros(env.num_envs, 3, device=env.device)
    present_count = torch.zeros(env.num_envs, device=env.device)

    for block in _get_block_infos():
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


def initial_tower_com_for_current_missing_pattern(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    target_block_name = asset_cfg.name
    total_com = torch.zeros(env.num_envs, 3, device=env.device)
    present_count = torch.zeros(env.num_envs, device=env.device)

    for block_name, block_pos in _INITIAL_BLOCK_POS_BY_NAME.items():
        if block_name == target_block_name:
            continue

        present = ~current_missing_block_mask(env, block_name)
        present_weight = present.to(dtype=total_com.dtype)
        total_com += block_pos.to(env.device).unsqueeze(0) * present_weight.unsqueeze(-1)
        present_count += present_weight

    return total_com / present_count.clamp_min(1.0).unsqueeze(-1)


def tower_com_shift(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    """
    Computes the horizontal shift of the COM of the Tower.
    """
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_tower_shift()

    current = get_com_tower(env, asset_cfg)
    start = initial_tower_com_for_current_missing_pattern(env, asset_cfg)
    movement = current - start
    horizontal_shift = torch.norm(movement[:, :2], dim=-1)
    return horizontal_shift


def tower_max_block_horizontal_shift(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_horizontal_shift()
    return tower_com_shift(env)


def tower_max_block_vertical_shift(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_vertical_shift()
    return torch.zeros(env.num_envs, device=env.device)


def tower_max_block_rotation(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_rotation()
    return torch.zeros(env.num_envs, device=env.device)


def tower_stable_for_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (
        (tower_max_block_horizontal_shift(env) < TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT)
        & (tower_max_block_vertical_shift(env) < TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT)
        & (tower_max_block_rotation(env) < TOWER_SUCCESS_MAX_BLOCK_ROTATION)
    )


#convert gripper to local coordinate frame of the block
def target_block_pose(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    extracts quaternion and position of block
    """
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        pose = cmd.selected_target_pose_w()
        return pose[:, :3], pose[:, 3:7]

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


def hook_joint_pos_ordered(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hook joints in policy order: slide, slide_y, slide_z, yaw."""
    asset: Entity = env.scene[_HOOK_ALL_CFG.name]
    joint_ids, _ = asset.find_joints(_HOOK_JOINT_ORDER, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids]


def hook_tip_pos_in_block_frame(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    Convert hook_tip_pos World coordinate system into a Block-local coordinate system.
    """
    block_pos_world, block_quat_world = target_block_pose(env, asset_cfg)
    hook_tip_pos_world = hook_tip_pos(env)

    position = hook_tip_pos_world - block_pos_world #vector from block_center to tip of the hook
    hook_tip_pos_block  = quat_apply_inverse(block_quat_world, position)

    return hook_tip_pos_block 


def target_extraction_direction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    """World direction in which the selected target block should be extracted."""
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_extraction_w()

    return torch.tensor(
        [-1.0, 0.0, 0.0],
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1)


def target_task_quat_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_task_quat_w()

    return torch.tensor(
        _rz_quat(math.pi),
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(env.num_envs, 1)


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


def _hook_contact_sensor(env: ManagerBasedRlEnv) -> ContactSensor:
    sensor = env.scene[HOOK_CONTACT_SENSOR_NAME]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"{HOOK_CONTACT_SENSOR_NAME} is not a ContactSensor")
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


def hook_contact_force_norm(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(hook_contact_force_world(env), dim=-1)


def hook_contact_found(env: ManagerBasedRlEnv) -> torch.Tensor:
    data = _hook_contact_sensor(env).data
    current = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if data.found is not None:
        current = (data.found > 0).any(dim=1)
    if data.force_history is None:
        return current.float()
    history = torch.linalg.vector_norm(data.force_history, dim=-1) > 1.0e-8
    return (current | history.any(dim=(1, 2))).float()


def hook_joint_pos_relative_to_target_home(env: ManagerBasedRlEnv) -> torch.Tensor:
    hook_joint_pos = hook_joint_pos_ordered(env)
    cmd = _target_command_or_none(env)
    if cmd is None:
        return hook_joint_pos
    return hook_joint_pos - cmd.selected_hook_home()


def target_selection_features(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return cmd.selected_target_features()


def target_support_presence(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Which blocks around the target are still there (8 fixed slots).

    Without this the actor cannot tell an intact tower from one missing a supporting
    block, while being penalized for instability and terminated on damage. Only the
    critic saw the tower, so it could recognise danger during training that the actor
    could not react to at execution time.
    """
    cmd = _target_command_or_none(env)
    if cmd is None:
        return torch.zeros(env.num_envs, 8, device=env.device)
    return cmd.selected_support_presence()


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


def target_contact_face_y(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is None:
        return torch.full((env.num_envs,), CONTACT_FACE_Y, device=env.device)
    return cmd.selected_contact_face_y()


def _initial_block_pos(block_name: str) -> torch.Tensor:
    for block_info in _get_block_infos():
        if block_info["name"] == block_name:
            return torch.tensor(block_info["pos"])
    raise ValueError(f"Unknown block name: {block_name}")


_START_REF_POS = (_initial_block_pos("b6_2") + _initial_block_pos("b6_3")) / 2
_START_TARGET_REL_POS = _initial_block_pos("b6_1") - _START_REF_POS
# Fraction of the block length that counts as extracted. START == END made this a
# no-op: 0.75 * 0.15 m = 112.5 mm was required from step 0, and the scripted controller
# needs 400-900 contact steps to get there, so early policies never saw a success at
# all. Ramping from 2 cm gives every target a reachable signal early -- even the ones
# that only reach 79-106 mm before tripping tower_damage.
SUCCESS_CURRICULUM_START = 0.1333   # 2.0 cm
SUCCESS_CURRICULUM_END = 0.75       # 11.25 cm
# 200k was slower than learning needed at the easy end: the first run reached the
# maximum attainable return within ~70 iterations at 2 cm and held it all the way
# through 4.1 cm at iteration 1400. Since a 12-hour job only covers ~1000
# iterations, that slack costs whole days of wall clock. Raise it again if the
# return starts lagging the ramp.
SUCCESS_CURRICULUM_STEPS = 120_000
TOUCH_CURRICULUM_START = 1.0
TOUCH_CURRICULUM_END = 1.0
TOUCH_CURRICULUM_BEGIN_STEP = 0
TOUCH_CURRICULUM_STEPS = 1
YAW_CURRICULUM_START = 0.10
YAW_CURRICULUM_END = 0.6
YAW_CURRICULUM_BEGIN_STEP = 0
YAW_CURRICULUM_STEPS = 600_000
YAW_ACTION_SCALE = 0.06
YAW_TARGET_LIMIT = 0.6
ACTION_CLIP = 1.0
HOOK_SLIDE_Y_TARGET_RANGE = (-0.13, 0.23)
HOOK_SLIDE_Z_TARGET_RANGE = (-0.17, 0.13)
PROGRESS_REWARD_WEIGHT = 8.0
SUCCESS_REWARD_WEIGHT = 10.0
TOWER_INSTABILITY_REWARD_WEIGHT = -2.0
TOWER_DAMAGE_REWARD_WEIGHT = -20.0
TIMEOUT_REWARD_WEIGHT = -5.0
STUCK_REWARD_WEIGHT = -0.005
ACTION_RATE_REWARD_WEIGHT = -0.0005
ACTION_MAGNITUDE_REWARD_WEIGHT = -0.00005
STUCK_CONTACT_FORCE_THRESHOLD = 2.0
STUCK_BLOCK_SPEED_THRESHOLD = 0.002
STUCK_GRACE_STEPS = 20
TOWER_INSTABILITY_GRACE_STEPS = 10

# Rewards
def target_block_relative_movement(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_relative_movement()

    ref_pos = get_block_ref_pos(env)
    target_pos = target_block_pos(env, asset_cfg)

    current_rel = target_pos - ref_pos
    return current_rel - _START_TARGET_REL_POS.to(current_rel.device)


def block_progress(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_progress()

    movement_rel = target_block_relative_movement(env, asset_cfg)

    extraction_direction = torch.tensor(
        [-1.0, 0.0, 0.0],
        device=movement_rel.device,
    )
    progress = torch.sum(movement_rel * extraction_direction, dim=-1)

    return progress


def _normalized_limit_excess(
    value: torch.Tensor,
    safe_limit: float,
    damage_limit: float,
) -> torch.Tensor:
    return torch.clamp(
        (value - safe_limit) / (damage_limit - safe_limit),
        min=0.0,
        max=1.0,
    )


def tower_instability_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    horizontal = _normalized_limit_excess(
        tower_max_block_horizontal_shift(env),
        TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT,
        TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT,
    )
    vertical = _normalized_limit_excess(
        tower_max_block_vertical_shift(env),
        TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT,
        TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT,
    )
    rotation = _normalized_limit_excess(
        tower_max_block_rotation(env),
        TOWER_SUCCESS_MAX_BLOCK_ROTATION,
        TOWER_DAMAGE_MAX_BLOCK_ROTATION,
    )
    return torch.maximum(torch.maximum(horizontal, vertical), rotation)


def success_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = min(curriculum_step(env) / SUCCESS_CURRICULUM_STEPS, 1.0)
    scale = SUCCESS_CURRICULUM_START + (
        SUCCESS_CURRICULUM_END - SUCCESS_CURRICULUM_START
    ) * progress
    return torch.tensor(scale, device=env.device)


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


def touch_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _linear_curriculum_scale(
        env,
        TOUCH_CURRICULUM_START,
        TOUCH_CURRICULUM_END,
        TOUCH_CURRICULUM_BEGIN_STEP,
        TOUCH_CURRICULUM_STEPS,
    )


def yaw_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _linear_curriculum_scale(
        env,
        YAW_CURRICULUM_START,
        YAW_CURRICULUM_END,
        YAW_CURRICULUM_BEGIN_STEP,
        YAW_CURRICULUM_STEPS,
    )


def success_done_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    return BLOCK_SIZE[1] * success_curriculum_scale(env)


def action_norm(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.norm(env.action_manager.action, dim=-1)


def action_magnitude_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


def push_velocity_target(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = env.action_manager.get_term("push_stop_retreat")
    if not isinstance(term, PushStopRetreatAction):
        raise TypeError("push_stop_retreat has an unexpected action term type")
    return term.processed_velocity.squeeze(-1)


def stop_action_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (torch.abs(env.action_manager.action[:, 0]) <= PUSH_ACTION_DEADZONE).float()


def retreat_action_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.action_manager.action[:, 0] > PUSH_ACTION_DEADZONE).float()


def stuck_contact_signal(env: ManagerBasedRlEnv) -> torch.Tensor:
    contact = hook_contact_found(env) > 0.0
    force_high = hook_contact_force_norm(env) > STUCK_CONTACT_FORCE_THRESHOLD
    extraction_speed = target_block_vel_in_task_frame(env)[:, 0]
    nearly_stationary = torch.abs(extraction_speed) < STUCK_BLOCK_SPEED_THRESHOLD
    return (contact & force_high & nearly_stationary).float()


def hook_x_position(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HOOK1_CFG,
) -> torch.Tensor:
    del asset_cfg
    return hook_joint_pos_ordered(env)[:, 0]


def debug_reward_signals(env: ManagerBasedRlEnv) -> torch.Tensor:
    if env.common_step_counter % 500 == 0:
        action = env.action_manager.action
        hook_joint_pos = hook_joint_pos_ordered(env)
        movement_rel = target_block_relative_movement(env)
        progress = block_progress(env)
        extracted = target_extraction_reached(env)
        success = success_block_extract(env)
        success_distance = success_done_distance(env)
        tower_shift = tower_com_shift(env)
        tower_max_xy = tower_max_block_horizontal_shift(env)
        tower_max_z = tower_max_block_vertical_shift(env)
        tower_max_rotation = tower_max_block_rotation(env)
        tower_instability = tower_instability_fraction(env)
        tower_damaged = tower_damage(env)
        contact_found = hook_contact_found(env)
        contact_force = hook_contact_force_in_task_frame(env)
        contact_force_norm = torch.linalg.vector_norm(contact_force, dim=-1)
        stuck = stuck_contact_signal(env)
        hook_x = hook_x_position(env)
        hook_tip_block = hook_tip_pos_in_block_frame(env)
        touch_raw = torch.clamp(action[:, 1:3], -ACTION_CLIP, ACTION_CLIP)
        contact_block = torch.zeros(env.num_envs, 3, device=env.device)
        contact_block[:, 0] = touch_raw[:, 0] * CONTACT_X_LIMIT * touch_curriculum_scale(env)
        contact_block[:, 1] = target_contact_face_y(env)
        contact_block[:, 2] = touch_raw[:, 1] * CONTACT_Z_LIMIT * touch_curriculum_scale(env)
        touch_target = block_contact_to_hook_yz_targets(env, contact_block)
        block_pos_world, block_quat_world = target_block_pose(env)
        contact_world = block_point_to_world(env, contact_block)
        contact_roundtrip = quat_apply_inverse(
            block_quat_world,
            contact_world - block_pos_world,
        )
        roundtrip_error = contact_roundtrip - contact_block
        tip_contact_error_block = hook_tip_block - contact_block
        x_velocity_target = push_velocity_target(env)
        hook_x_joint = hook_joint_pos[:, 0]
        missing_mask = getattr(env, "_jenga_missing_block_mask", None)
        if missing_mask is None:
            missing_env_count = 0
            missing_block_count = 0
        else:
            missing_env_count = int(torch.any(missing_mask, dim=1).sum().item())
            missing_block_count = int(missing_mask.sum().item())
        cmd = _target_command_or_none(env)
        if cmd is None:
            random_target_env_count = 0
            random_missing_env_count = 0
            selected_block_counts = "none"
        else:
            random_target_env_count = int(cmd.selected_is_random.sum().item())
            if missing_mask is None:
                random_missing_env_count = 0
            else:
                random_missing_env_count = int(
                    (cmd.selected_is_random & torch.any(missing_mask, dim=1)).sum().item()
                )
            selected_block_counts = cmd.selected_block_count_summary()
        missing_pattern_counts = missing_pattern_count_summary(env)
        yaw_scale = yaw_curriculum_scale(env)
        yaw_step_max = YAW_ACTION_SCALE * yaw_scale
        best_env = int(torch.argmax(progress).item())
        worst_env = int(torch.argmin(progress).item())
        print(
            "DEBUG_REWARD",
            f"step={env.common_step_counter}",
            f"curriculum(success_dist={success_distance.item():.5f}, touch={touch_curriculum_scale(env).item():.3f}, yaw={yaw_scale.item():.3f}, yaw_step_max={yaw_step_max.item():.5f}, missing={missing_block_randomization_scale(env).item():.3f}, missing_max={missing_block_max_count(env)}, random_target={random_target_block_scale(env).item():.3f}, random_missing={random_target_with_missing_scale(env).item():.3f})",
            f"reward_cfg(progress_total_max={PROGRESS_REWARD_WEIGHT:.2f}, success={SUCCESS_REWARD_WEIGHT:.2f}, instability_total_max={TOWER_INSTABILITY_REWARD_WEIGHT:.2f}, instability_grace={TOWER_INSTABILITY_GRACE_STEPS}, stuck_per_step={STUCK_REWARD_WEIGHT:.3f}, stuck_grace={STUCK_GRACE_STEPS}, timeout={TIMEOUT_REWARD_WEIGHT:.2f}, damage={TOWER_DAMAGE_REWARD_WEIGHT:.2f}, dt_scaled=False)",
            f"progress(mean={progress.mean().item():.5f}, min={progress.min().item():.5f}, max={progress.max().item():.5f}, extracted_count={int(extracted.sum().item())}/{env.num_envs}, safe_success_count={int(success.sum().item())}/{env.num_envs})",
            f"movement(mean_xyz=({movement_rel[:, 0].mean().item():.5f},{movement_rel[:, 1].mean().item():.5f},{movement_rel[:, 2].mean().item():.5f}))",
            f"tower(com_shift_mean={tower_shift.mean().item():.5f}, max_block_xy_mean={tower_max_xy.mean().item():.5f}, max_block_z_mean={tower_max_z.mean().item():.5f}, max_block_rot_deg_mean={torch.rad2deg(tower_max_rotation).mean().item():.2f}, instability_mean={tower_instability.mean().item():.3f}, damage_count={int(tower_damaged.sum().item())}/{env.num_envs}, missing_envs={missing_env_count}/{env.num_envs}, missing_blocks={missing_block_count})",
            f"target(random_envs={random_target_env_count}/{env.num_envs}, random_missing_envs={random_missing_env_count}/{env.num_envs}, counts={selected_block_counts})",
            f"missing_patterns({missing_pattern_counts})",
            f"action(mean_xyzyaw=({action[:, 0].mean().item():.5f},{action[:, 1].mean().item():.5f},{action[:, 2].mean().item():.5f},{action[:, 3].mean().item():.5f}), norm={action_norm(env).mean().item():.5f}, x_vel_target={x_velocity_target.mean().item():.5f})",
            f"contact_sensor(found={int(contact_found.sum().item())}/{env.num_envs}, force_task_mean=({contact_force[:, 0].mean().item():.3f},{contact_force[:, 1].mean().item():.3f},{contact_force[:, 2].mean().item():.3f}), force_norm_mean={contact_force_norm.mean().item():.3f}, force_norm_max={contact_force_norm.max().item():.3f}, stuck={int(stuck.sum().item())}/{env.num_envs})",
            f"contact(desired_block_xz=({contact_block[:, 0].mean().item():.5f},{contact_block[:, 2].mean().item():.5f}), face_y={contact_block[:, 1].mean().item():.5f}, raw_xz=({touch_raw[:, 0].mean().item():.5f},{touch_raw[:, 1].mean().item():.5f}))",
            f"tracking(tip_block_yz=({hook_tip_block[:, 1].mean().item():.5f},{hook_tip_block[:, 2].mean().item():.5f}), err_yz=({tip_contact_error_block[:, 1].mean().item():.5f},{tip_contact_error_block[:, 2].mean().item():.5f}), target_yz=({touch_target[:, 0].mean().item():.5f},{touch_target[:, 1].mean().item():.5f}))",
            f"transform(roundtrip_err={torch.norm(roundtrip_error, dim=-1).mean().item():.8f})",
            f"hook(joint_x_mean={hook_x_joint.mean().item():.5f}, joint_x_minmax=({hook_x_joint.min().item():.5f},{hook_x_joint.max().item():.5f}), hook_x_mean={hook_x.mean().item():.5f}, hook_x_minmax=({hook_x.min().item():.5f},{hook_x.max().item():.5f}), joint_yz=({hook_joint_pos[:, 1].mean().item():.5f},{hook_joint_pos[:, 2].mean().item():.5f}), joint_yaw={hook_joint_pos[:, 3].mean().item():.5f})",
            f"env_compare(best={best_env}:progress={progress[best_env].item():.5f},hook_x={hook_x_joint[best_env].item():.5f},act_x={action[best_env, 0].item():.5f}; worst={worst_env}:progress={progress[worst_env].item():.5f},hook_x={hook_x_joint[worst_env].item():.5f},act_x={action[worst_env, 0].item():.5f})",
            flush=True,
        )
    return torch.zeros(env.num_envs, device=env.device)


class NormalizedDeltaBlockProgressReward:
    """Reward increases of the furthest normalized extraction progress."""

    def __init__(self, asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG):
        self.asset_cfg = asset_cfg
        self.best_progress_fraction: torch.Tensor | None = None
        self.needs_init: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        success_distance = success_done_distance(env).clamp_min(1.0e-6)
        current_fraction = torch.clamp(
            block_progress(env, self.asset_cfg) / success_distance,
            min=0.0,
            max=1.0,
        )

        if self.best_progress_fraction is None:
            self.best_progress_fraction = current_fraction.clone()
            self.needs_init = torch.zeros_like(current_fraction, dtype=torch.bool)
            return torch.zeros_like(current_fraction)

        if self.needs_init is not None and torch.any(self.needs_init):
            self.best_progress_fraction[self.needs_init] = current_fraction[self.needs_init]
            self.needs_init[self.needs_init] = False

        new_progress = torch.clamp(
            current_fraction - self.best_progress_fraction,
            min=0.0,
        )
        self.best_progress_fraction = torch.maximum(
            self.best_progress_fraction,
            current_fraction,
        )
        return new_progress

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.best_progress_fraction is None:
            return

        if self.needs_init is None:
            self.needs_init = torch.zeros_like(
                self.best_progress_fraction,
                dtype=torch.bool,
            )

        if env_ids is None:
            env_ids = slice(None)
        self.needs_init[env_ids] = True


class SustainedStuckPenalty:
    """Activate after forceful contact without block motion persists."""

    def __init__(self, grace_steps: int = STUCK_GRACE_STEPS):
        self.grace_steps = grace_steps
        self.consecutive_steps: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        stuck = stuck_contact_signal(env) > 0.0
        if self.consecutive_steps is None:
            self.consecutive_steps = torch.zeros(
                env.num_envs,
                device=env.device,
                dtype=torch.long,
            )

        self.consecutive_steps = torch.where(
            stuck,
            self.consecutive_steps + 1,
            torch.zeros_like(self.consecutive_steps),
        )
        return (self.consecutive_steps > self.grace_steps).float()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.consecutive_steps is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self.consecutive_steps[env_ids] = 0


class NewTowerInstabilityPenalty:
    """Penalize only increases in peak instability after passive settling."""

    def __init__(self, grace_steps: int = TOWER_INSTABILITY_GRACE_STEPS):
        self.grace_steps = grace_steps
        self.episode_steps: torch.Tensor | None = None
        self.peak_instability: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self.episode_steps is None:
            self.episode_steps = torch.zeros(
                env.num_envs,
                device=env.device,
                dtype=torch.long,
            )
            self.peak_instability = torch.zeros(env.num_envs, device=env.device)

        self.episode_steps += 1
        current = tower_instability_fraction(env)
        baseline = self.episode_steps == self.grace_steps + 1
        active = self.episode_steps > self.grace_steps + 1
        assert self.peak_instability is not None
        self.peak_instability = torch.where(
            baseline,
            current,
            self.peak_instability,
        )
        new_instability = torch.clamp(current - self.peak_instability, min=0.0)
        self.peak_instability = torch.where(
            active,
            torch.maximum(self.peak_instability, current),
            self.peak_instability,
        )
        return torch.where(active, new_instability, torch.zeros_like(current))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.episode_steps is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self.episode_steps[env_ids] = 0
        if self.peak_instability is not None:
            self.peak_instability[env_ids] = 0.0



def get_block_ref_pos(env : ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_ref_pos_w()

    ref1_block_pos = target_block_pos(env, _REF_BLOCK_1_CFG)
    ref2_block_pos = target_block_pos(env, _REF_BLOCK_2_CFG)
    ref_block_state_mean = (ref1_block_pos + ref2_block_pos) / 2
    return ref_block_state_mean

def target_extraction_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = block_progress(env)
    return progress > success_done_distance(env)


def success_block_extract(env: ManagerBasedRlEnv) -> torch.Tensor:
    return target_extraction_reached(env) & tower_stable_for_success(env)


def success_block_reward(env : ManagerBasedRlEnv) -> torch.Tensor:
    return success_block_extract(env).float()


def tower_damage(env : ManagerBasedRlEnv) -> torch.Tensor:
    return (
        (tower_max_block_horizontal_shift(env) > TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT)
        | (tower_max_block_vertical_shift(env) > TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT)
        | (tower_max_block_rotation(env) > TOWER_DAMAGE_MAX_BLOCK_ROTATION)
    )


def tower_damage_signal(env: ManagerBasedRlEnv) -> torch.Tensor:
    return tower_damage(env).float()


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
        cmd = _target_command_or_none(self._env)
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
        cmd = _target_command_or_none(self._env)
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


# Environment configuration


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "pusher_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _HOOK_ALL_CFG}
        ),
        "pusher_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _HOOK_ALL_CFG}
        ),
        "pusher_target_home_error": ObservationTermCfg(
            func=hook_joint_pos_relative_to_target_home,
        ),
        "target_selection": ObservationTermCfg(
            func=target_selection_features,
        ),
        "hook_tip_task_position": ObservationTermCfg(
            func=hook_tip_pos_in_task_frame,
        ),
        "target_task_movement": ObservationTermCfg(
            func=target_block_movement_in_task_frame,
        ),
        "target_task_velocity": ObservationTermCfg(
            func=target_block_vel_in_task_frame,
        ),
        "hook_contact": ObservationTermCfg(
            func=hook_contact_observation,
        ),
        "support_presence": ObservationTermCfg(
            func=target_support_presence,
        ),
        "tower_state": ObservationTermCfg(
            func=tower_state_observation,
        ),
        "last_action": ObservationTermCfg(
            func=last_action,
        ),
    }

    critic_terms = {
        **actor_terms,
        "block_all_pos": ObservationTermCfg(
            func=all_block_pos,
        ),
    }


    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms),
    }


    actions : dict[str, ActionTermCfg] = {
        "push_stop_retreat": PushStopRetreatActionCfg(
            entity_name="hook",
        ),
        "block_local_touch": BlockLocalHookYZActionCfg(
            entity_name="hook",
            scale=(CONTACT_X_LIMIT, CONTACT_Z_LIMIT),
            asset_cfg=_TARGET_BLOCK_CFG,
        ),
        "yaw" : CurriculumYawActionCfg(
            entity_name="hook",
            scale=YAW_ACTION_SCALE,
        ),
    }


    events = {
        "randomize_block_physics": EventTermCfg(
            func=randomize_block_physics,
            mode="reset",
        ),
        "randomize_missing_blocks": EventTermCfg(
            func=randomize_missing_blocks,
            mode="reset",
        ),
    }


    rewards = {
        "normalized_new_progress": RewardTermCfg(
            func=NormalizedDeltaBlockProgressReward(),
            weight=PROGRESS_REWARD_WEIGHT,
        ),
        "action_rate": RewardTermCfg(
            func=action_rate_l2,
            weight=ACTION_RATE_REWARD_WEIGHT,
        ),
        "action_magnitude": RewardTermCfg(
            func=action_magnitude_l2,
            weight=ACTION_MAGNITUDE_REWARD_WEIGHT,
        ),
        "sustained_stuck": RewardTermCfg(
            func=SustainedStuckPenalty(),
            weight=STUCK_REWARD_WEIGHT,
        ),
        "successful_extract": RewardTermCfg(
            func=success_block_reward,
            weight=SUCCESS_REWARD_WEIGHT,
        ),
        "tower_instability": RewardTermCfg(
            func=NewTowerInstabilityPenalty(),
            weight=TOWER_INSTABILITY_REWARD_WEIGHT,
        ),
        "tower_damage": RewardTermCfg(
            func=tower_damage_signal,
            weight=TOWER_DAMAGE_REWARD_WEIGHT,
        ),
        "timeout": RewardTermCfg(
            func=time_out,
            weight=TIMEOUT_REWARD_WEIGHT,
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
        "extraction_reached_last": MetricsTermCfg(
            func=target_extraction_reached,
            reduce="last",
        ),
        "normalized_new_progress_mean": MetricsTermCfg(
            func=NormalizedDeltaBlockProgressReward(),
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
        "tower_instability_mean": MetricsTermCfg(
            func=tower_instability_fraction,
            reduce="mean",
        ),
        "tower_damage_mean": MetricsTermCfg(
            func=tower_damage_signal,
            reduce="mean",
        ),
        "timeout_last": MetricsTermCfg(
            func=time_out,
            reduce="last",
        ),
        "tower_max_block_horizontal_shift_last": MetricsTermCfg(
            func=tower_max_block_horizontal_shift,
            reduce="last",
        ),
        "tower_max_block_vertical_shift_last": MetricsTermCfg(
            func=tower_max_block_vertical_shift,
            reduce="last",
        ),
        "tower_max_block_rotation_last": MetricsTermCfg(
            func=tower_max_block_rotation,
            reduce="last",
        ),
        "action_norm_mean": MetricsTermCfg(
            func=action_norm,
            reduce="mean",
        ),
        "hook_contact_force_mean": MetricsTermCfg(
            func=hook_contact_force_norm,
            reduce="mean",
        ),
        "hook_contact_found_mean": MetricsTermCfg(
            func=hook_contact_found,
            reduce="mean",
        ),
        "stuck_contact_mean": MetricsTermCfg(
            func=stuck_contact_signal,
            reduce="mean",
        ),
        "stop_action_mean": MetricsTermCfg(
            func=stop_action_fraction,
            reduce="mean",
        ),
        "retreat_action_mean": MetricsTermCfg(
            func=retreat_action_fraction,
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
        "tower_damage": TerminationTermCfg(func=tower_damage),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    commands = {
        "target_block": TargetBlockCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
        ),
    }


    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=_build_entities(),
            sensors=(
                ContactSensorCfg(
                    name=HOOK_CONTACT_SENSOR_NAME,
                    primary=ContactMatch(
                        mode="subtree",
                        pattern="hook_tool",
                        entity="hook",
                    ),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                    history_length=5,
                ),
            ),
            num_envs=512,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        scale_rewards_by_dt=False,
        metrics=metrics,
        terminations=terminations,
        commands=commands,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=1.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            nconmax=4096,
            njmax=4096,
            # impratio is the stiffness of friction constraints relative to normal
            # ones. At MuJoCo's default of 1.0 they are equally soft, so contacts creep
            # tangentially well below the friction limit and the whole tower shears:
            # pushing b6_1 moved b8_1 -- two layers up -- by 89% as far, and twisted the
            # top of the tower by 8.45 deg. tower_damage (25 mm for any non-target block)
            # then fired at ~28 mm of target progress against a 112.5 mm success
            # distance, leaving 12 of 14 targets unextractable.
            #
            # Measured mean drag ratio over b6_1/b6_3/b3_1/b7_1 (pyramidal cone):
            #
            #        mu     impratio=1   impratio=10   impratio=30
            #      0.20          0.566         0.323         0.270
            #      0.28          0.615         0.272         0.196
            #      0.48          0.896         0.285         0.194
            #
            # Note the sign flip: at impratio=1 more friction means more drag, because
            # what is being measured is compliance scaling with contact load. Once the
            # compliance is gone, more friction means less drag -- the neighbours hold
            # each other. So the existing friction range is already right; impratio was
            # the wrong parameter. 30 sits inside MuJoCo's recommended 10-100 band for
            # friction-critical contact. Extractable targets went from 2/14 to 7/14.
            #
            # elliptic cone was measured too and is unusable here: it pushes the reset
            # settling transient past the damage limit (tower_damage fires at step 6
            # with the target unmoved) and locks other targets solid.
            mujoco=MujocoCfg(timestep=0.002, impratio=30.0),
        ),
        decimation=5,
        episode_length_s=20.0,
    )


def apply_low_level_stage(stage: str) -> None:
    global MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP
    global MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS
    global MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY
    global MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY
    global MISSING_BLOCK_DOUBLE_BEGIN_STEP
    global MISSING_BLOCK_TRIPLE_BEGIN_STEP
    global RANDOM_TARGET_BLOCK_BEGIN_STEP
    global RANDOM_TARGET_BLOCK_RAMP_STEPS
    global RANDOM_TARGET_BLOCK_START_PROBABILITY
    global RANDOM_TARGET_BLOCK_END_PROBABILITY
    global RANDOM_TARGET_WITH_MISSING_BEGIN_STEP
    global RANDOM_TARGET_WITH_MISSING_RAMP_STEPS
    global RANDOM_TARGET_WITH_MISSING_START_PROBABILITY
    global RANDOM_TARGET_WITH_MISSING_END_PROBABILITY
    global FORCED_MISSING_BLOCK_COUNT

    MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
    MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 600_000
    MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.05
    MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.35
    MISSING_BLOCK_DOUBLE_BEGIN_STEP = 250_000
    MISSING_BLOCK_TRIPLE_BEGIN_STEP = 500_000
    RANDOM_TARGET_BLOCK_BEGIN_STEP = 0
    RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
    RANDOM_TARGET_BLOCK_START_PROBABILITY = 1.0
    RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
    RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 0
    RANDOM_TARGET_WITH_MISSING_RAMP_STEPS = 1
    RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 1.0
    RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 1.0
    FORCED_MISSING_BLOCK_COUNT = None

    if stage == "fixed":
        MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
        RANDOM_TARGET_BLOCK_START_PROBABILITY = 0.0
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.0
        RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 0.0
        RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 0.0
    elif stage == "target":
        MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
        RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 0.0
        RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 0.0
    elif stage == "missing1":
        MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
        MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 1
        MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.50
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
        MISSING_BLOCK_DOUBLE_BEGIN_STEP = 10**12
        MISSING_BLOCK_TRIPLE_BEGIN_STEP = 10**12
        RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 1.0
    elif stage == "missing2":
        MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
        MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 1
        MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.50
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
        MISSING_BLOCK_DOUBLE_BEGIN_STEP = 0
        MISSING_BLOCK_TRIPLE_BEGIN_STEP = 10**12
        RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 1.0
    elif stage == "missing3":
        MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
        MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 1
        MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.50
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
        MISSING_BLOCK_DOUBLE_BEGIN_STEP = 0
        MISSING_BLOCK_TRIPLE_BEGIN_STEP = 0
        RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 1.0
    elif stage == "full":
        pass
    else:
        raise ValueError(
            "Unknown stage. Use fixed, target, missing1, missing2, missing3, or full."
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
      # The critic reads 81 raw block coordinates alongside joint angles and contact
      # forces, which span very different scales.
      obs_normalization=True,
      # rsl_rl defaults std_range to (1e-6, 1e6) and learns std, while the entropy
      # bonus pushes it up and clip_actions clips only AFTER sampling -- so PPO keeps
      # scoring log-probs of samples that all execute as the same boundary action.
      # Observed std of 117 and 743, with mean action norm exactly 2.0 (the maximum for
      # four actions in [-1,1]) and an episode return of exactly
      # -0.00005 * 4 * 2000 = -0.40, i.e. pure action-magnitude penalty and nothing else.
      # Bounding std_range is the fix; "log" is better conditioned than "scalar".
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.25,
        "std_type": "log",
        "std_range": (0.05, 1.0),
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # A/B at equal iteration (800): entropy 0.01 -> reward 17.76 with std pinned
      # at the 1.0 cap since iteration ~400; entropy 0.002 -> reward 17.49 with std
      # falling to 0.16. Same return, but the weaker bonus lets the policy gradient
      # pull std down instead of the entropy term pushing it into the clip region.
      entropy_coef=0.002,
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
    clip_actions=1.0,
    # A 12-hour job reaches roughly 900-1400 iterations, so saving every 500 threw
    # away 411 iterations when the first run was cut off at 911. This task needs
    # several chained resume jobs to finish its curriculum, so that loss compounds.
    # Checkpoints are ~220 KB.
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=10000,
  )




if __name__ == "__main__":
    entities = _build_entities()
    print(entities.keys())
    print(len(entities))
