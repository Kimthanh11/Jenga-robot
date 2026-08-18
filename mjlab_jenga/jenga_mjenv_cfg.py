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
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.dr import geom_friction, pseudo_inertia
from mjlab.envs.mdp.actions import (
    JointEffortActionCfg,
    JointVelocityActionCfg,
    RelativeJointPositionActionCfg,
)
from mjlab.envs.mdp.rewards import joint_torques_l2, action_rate_l2
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
RESET_FRICTION_SLIDING_RANGE = (0.35, 0.60)
RESET_FRICTION_TORSIONAL_RANGE = (0.02, 0.08)
RESET_FRICTION_ROLLING_RANGE = (0.001, 0.001)
CONTACT_X_LIMIT = 0.01
CONTACT_Y_LIMIT = BLOCK_HALF_SIZE[1]
CONTACT_Z_LIMIT = 0.006
CONTACT_FACE_Y = -CONTACT_Y_LIMIT
PUSH_X_VELOCITY_SCALE = 0.03
PUSH_X_VELOCITY_CLIP = (-0.05, 0.05)

SIDE_SPACING = BLOCK_SIZE[0] + 0.0005
START_Z = (BLOCK_SIZE[2] / 2) + 0.0005
LAYER_HEIGHT = BLOCK_SIZE[2] + 0.0005

MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 170_000
MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 120_000
MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
MISSING_BLOCK_DOUBLE_BEGIN_STEP = 520_000
MISSING_BLOCK_TRIPLE_BEGIN_STEP = 760_000
FORCED_MISSING_BLOCK_COUNT: int | None = None
MISSING_BLOCK_PARK_OFFSET = (1.5, 1.5, 0.5)
MISSING_BLOCK_PARK_SPACING = 0.2
RANDOM_TARGET_BLOCK_BEGIN_STEP = 350_000
RANDOM_TARGET_BLOCK_RAMP_STEPS = 240_000
RANDOM_TARGET_BLOCK_START_PROBABILITY = 0.0
RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.15
RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 620_000
RANDOM_TARGET_WITH_MISSING_RAMP_STEPS = 240_000
FIXED_TARGET_BLOCK_NAME = "b6_1"
RANDOM_TARGET_BLOCK_NAMES = (
    "b1_1",
    "b1_3",
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
            env.common_step_counter - RANDOM_TARGET_BLOCK_BEGIN_STEP,
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
            env.common_step_counter - RANDOM_TARGET_WITH_MISSING_BEGIN_STEP,
            0,
        )
        / RANDOM_TARGET_WITH_MISSING_RAMP_STEPS
    )
    return torch.tensor(min(progress, 1.0), device=env.device)


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

        name_to_idx = {name: idx for idx, name in enumerate(names)}
        self._fixed_idx = name_to_idx[cfg.fixed_target_name]
        self._force_target_idx: int | None = None
        if cfg.force_target_name is not None:
            if cfg.force_target_name not in name_to_idx:
                raise ValueError(f"Unknown forced target block: {cfg.force_target_name}")
            self._force_target_idx = name_to_idx[cfg.force_target_name]
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

        self.metrics["selected_block"] = selected.float()
        self.metrics["random_target"] = self.selected_is_random.float()
        self.metrics["progress"] = self._cur_progress

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        if self._force_target_idx is None:
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

        if torch.any(use_random):
            random_env_ids = env_ids[use_random]
            target = self._hook_home[selected[use_random]].clone()
            target[:, :3] += torch.empty_like(target[:, :3]).uniform_(-0.002, 0.002)
            target[:, 3] += torch.empty(
                target.shape[0],
                device=self.device,
                dtype=target.dtype,
            ).uniform_(-0.02, 0.02)
            self._hook.write_joint_position_to_sim(
                target,
                joint_ids=self._hook_home_joint_ids,
                env_ids=random_env_ids,
            )
            self._hook.write_joint_velocity_to_sim(
                torch.zeros_like(target),
                joint_ids=self._hook_home_joint_ids,
                env_ids=random_env_ids,
            )

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class TargetBlockCommandCfg(CommandTermCfg):
    fixed_target_name: str = FIXED_TARGET_BLOCK_NAME
    selectable_target_names: tuple[str, ...] = RANDOM_TARGET_BLOCK_NAMES
    force_target_name: str | None = None

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
            env.common_step_counter - MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP,
            0,
        )
        / MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS,
        1.0,
    )
    return torch.tensor(
        progress * MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY,
        device=env.device,
    )


def missing_block_max_count(env: ManagerBasedRlEnv) -> int:
    """Maximum number of missing blocks allowed at the current curriculum step."""
    if FORCED_MISSING_BLOCK_COUNT is not None:
        return FORCED_MISSING_BLOCK_COUNT
    if env.common_step_counter >= MISSING_BLOCK_TRIPLE_BEGIN_STEP:
        return 3
    if env.common_step_counter >= MISSING_BLOCK_DOUBLE_BEGIN_STEP:
        return 2
    if env.common_step_counter >= MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP:
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

      <body name="hook_tool" pos="0 0 0">
        <joint name="hook_slide" type="slide" axis="1 0 0" range="-0.22 0.16" limited="true" damping="2"/>
        <joint name="hook_slide_y" type="slide" axis="0 1 0" range="-0.13 0.23" limited="true" damping="2"/>
        <joint name="hook_slide_z" type="slide" axis="0 0 1" range="-0.17 0.13" limited="true" damping="2"/>

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
            gap="0"/>
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
    positions = []
    for block_cfg in _ALL_BLOCK_CFGS:
        pos = target_block_pos(env, block_cfg)
        positions.append(pos)
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


def hook_joint_pos_relative_to_target_home(env: ManagerBasedRlEnv) -> torch.Tensor:
    hook_asset: Entity = env.scene[_HOOK_ALL_CFG.name]
    hook_joint_pos = hook_asset.data.joint_pos[:, _HOOK_ALL_CFG.joint_ids]
    cmd = _target_command_or_none(env)
    if cmd is None:
        return hook_joint_pos
    return hook_joint_pos - cmd.selected_hook_home()


def target_selection_features(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = _target_command_or_none(env)
    if cmd is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return cmd.selected_target_features()


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
PERTURBATION_CURRICULUM_START = 0.1
PERTURBATION_CURRICULUM_STEPS = 100_000
SUCCESS_CURRICULUM_START = 0.75
SUCCESS_CURRICULUM_END = 0.75
SUCCESS_CURRICULUM_STEPS = 1
TOUCH_CURRICULUM_START = 1.0
TOUCH_CURRICULUM_END = 1.0
TOUCH_CURRICULUM_BEGIN_STEP = 0
TOUCH_CURRICULUM_STEPS = 1
YAW_CURRICULUM_START = 0.0
YAW_CURRICULUM_END = 0.6
YAW_CURRICULUM_BEGIN_STEP = 80_000
YAW_CURRICULUM_STEPS = 160_000
YAW_ACTION_SCALE = 0.06
YAW_TARGET_LIMIT = 0.6
ACTION_CLIP = 1.0
HOOK_SLIDE_Y_TARGET_RANGE = (-0.13, 0.23)
HOOK_SLIDE_Z_TARGET_RANGE = (-0.17, 0.13)
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


def tower_moderate_perturbation(env: ManagerBasedRlEnv) -> torch.Tensor:
    return tower_com_shift(env)


def tower_large_perturbation(env: ManagerBasedRlEnv) -> torch.Tensor:
    shift = tower_com_shift(env)
    return (shift > 0.02).float()


def perturbation_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = min(env.common_step_counter / PERTURBATION_CURRICULUM_STEPS, 1.0)
    scale = PERTURBATION_CURRICULUM_START + (
        1.0 - PERTURBATION_CURRICULUM_START
    ) * progress
    return torch.tensor(scale, device=env.device)


def success_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = min(env.common_step_counter / SUCCESS_CURRICULUM_STEPS, 1.0)
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
    progress = min(max(env.common_step_counter - begin_step, 0) / steps, 1.0)
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


def progress_towards_success_distance_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dense reward for being closer to the configured extraction success distance."""
    progress = block_progress(env)
    target = success_done_distance(env)
    progress_fraction = torch.clamp(progress / target, 0.0, 1.0)
    return progress_fraction ** 2


def tower_moderate_perturbation_curriculum(env: ManagerBasedRlEnv) -> torch.Tensor:
    return tower_moderate_perturbation(env) * perturbation_curriculum_scale(env)


def tower_large_perturbation_curriculum(env: ManagerBasedRlEnv) -> torch.Tensor:
    return tower_large_perturbation(env) * perturbation_curriculum_scale(env)


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
        action = env.action_manager.action
        hook_asset: Entity = env.scene[_HOOK_ALL_CFG.name]
        hook_joint_pos = hook_asset.data.joint_pos[:, _HOOK_ALL_CFG.joint_ids]
        movement_rel = target_block_relative_movement(env)
        progress = block_progress(env)
        success = success_block_extract(env)
        success_distance = success_done_distance(env)
        tower_shift = tower_com_shift(env)
        tower_large = tower_large_perturbation(env)
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
        x_velocity_target = torch.clamp(
            action[:, 0] * PUSH_X_VELOCITY_SCALE,
            PUSH_X_VELOCITY_CLIP[0],
            PUSH_X_VELOCITY_CLIP[1],
        )
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
            f"curriculum(success_dist={success_distance.item():.5f}, perturb={perturbation_curriculum_scale(env).item():.3f}, touch={touch_curriculum_scale(env).item():.3f}, yaw={yaw_scale.item():.3f}, yaw_step_max={yaw_step_max.item():.5f}, missing={missing_block_randomization_scale(env).item():.3f}, missing_max={missing_block_max_count(env)}, random_target={random_target_block_scale(env).item():.3f}, random_missing={random_target_with_missing_scale(env).item():.3f})",
            f"progress(mean={progress.mean().item():.5f}, min={progress.min().item():.5f}, max={progress.max().item():.5f}, success_count={int(success.sum().item())}/{env.num_envs})",
            f"movement(mean_xyz=({movement_rel[:, 0].mean().item():.5f},{movement_rel[:, 1].mean().item():.5f},{movement_rel[:, 2].mean().item():.5f}))",
            f"tower(shift_mean={tower_shift.mean().item():.5f}, large_count={int(tower_large.sum().item())}/{env.num_envs}, missing_envs={missing_env_count}/{env.num_envs}, missing_blocks={missing_block_count})",
            f"target(random_envs={random_target_env_count}/{env.num_envs}, random_missing_envs={random_missing_env_count}/{env.num_envs}, counts={selected_block_counts})",
            f"missing_patterns({missing_pattern_counts})",
            f"action(mean_xyzyaw=({action[:, 0].mean().item():.5f},{action[:, 1].mean().item():.5f},{action[:, 2].mean().item():.5f},{action[:, 3].mean().item():.5f}), norm={action_norm(env).mean().item():.5f}, x_vel_target={x_velocity_target.mean().item():.5f})",
            f"contact(desired_block_xz=({contact_block[:, 0].mean().item():.5f},{contact_block[:, 2].mean().item():.5f}), face_y={contact_block[:, 1].mean().item():.5f}, raw_xz=({touch_raw[:, 0].mean().item():.5f},{touch_raw[:, 1].mean().item():.5f}))",
            f"tracking(tip_block_yz=({hook_tip_block[:, 1].mean().item():.5f},{hook_tip_block[:, 2].mean().item():.5f}), err_yz=({tip_contact_error_block[:, 1].mean().item():.5f},{tip_contact_error_block[:, 2].mean().item():.5f}), target_yz=({touch_target[:, 0].mean().item():.5f},{touch_target[:, 1].mean().item():.5f}))",
            f"transform(roundtrip_err={torch.norm(roundtrip_error, dim=-1).mean().item():.8f})",
            f"hook(joint_x_mean={hook_x_joint.mean().item():.5f}, joint_x_minmax=({hook_x_joint.min().item():.5f},{hook_x_joint.max().item():.5f}), hook_x_mean={hook_x.mean().item():.5f}, hook_x_minmax=({hook_x.min().item():.5f},{hook_x.max().item():.5f}), joint_yz=({hook_joint_pos[:, 1].mean().item():.5f},{hook_joint_pos[:, 2].mean().item():.5f}), joint_yaw={hook_joint_pos[:, 3].mean().item():.5f})",
            f"env_compare(best={best_env}:progress={progress[best_env].item():.5f},hook_x={hook_x_joint[best_env].item():.5f},act_x={action[best_env, 0].item():.5f}; worst={worst_env}:progress={progress[worst_env].item():.5f},hook_x={hook_x_joint[worst_env].item():.5f},act_x={action[worst_env, 0].item():.5f})",
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
    cmd = _target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_ref_pos_w()

    ref1_block_pos = target_block_pos(env, _REF_BLOCK_1_CFG)
    ref2_block_pos = target_block_pos(env, _REF_BLOCK_2_CFG)
    ref_block_state_mean = (ref1_block_pos + ref2_block_pos) / 2
    return ref_block_state_mean

def success_block_extract(env : ManagerBasedRlEnv) -> torch.Tensor:
    progress = block_progress(env)
    return progress > success_done_distance(env)


def success_block_reward(env : ManagerBasedRlEnv) -> torch.Tensor:
    return success_block_extract(env).float()


def tower_damage(env : ManagerBasedRlEnv) -> torch.Tensor:
    ref_pos = get_block_ref_pos(env)
    movement = ref_pos - _START_REF_POS.to(ref_pos.device)
    horizontal_movement = torch.norm(movement[:, :2], dim=-1)
    return horizontal_movement > 0.06


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
    hook_asset = env.scene[_HOOK_ALL_CFG.name]
    hook_joint_pos = hook_asset.data.joint_pos[:, _HOOK_ALL_CFG.joint_ids]
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

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -ACTION_CLIP, ACTION_CLIP)

        joint_pos = self._entity.data.joint_pos[:, self._target_ids]
        delta_yaw = self._raw_actions * self.cfg.scale * yaw_curriculum_scale(self._env)
        cmd = _target_command_or_none(self._env)
        if cmd is not None:
            home_yaw = cmd.selected_hook_home()[:, 3:4]
            self._processed_targets = torch.clamp(
                joint_pos + delta_yaw,
                home_yaw - YAW_TARGET_LIMIT,
                home_yaw + YAW_TARGET_LIMIT,
            )
            return

        self._processed_targets = torch.clamp(
            joint_pos + delta_yaw,
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
            scale=PUSH_X_VELOCITY_SCALE,
            clip={"hook_slide": PUSH_X_VELOCITY_CLIP},
        ),
        "block_local_touch": BlockLocalHookYZActionCfg( #yields 2 actions
            entity_name="hook",
            scale=(CONTACT_X_LIMIT, CONTACT_Z_LIMIT),
            asset_cfg=_TARGET_BLOCK_CFG,
        ),
        "yaw" : CurriculumYawActionCfg(
            entity_name="hook",
            scale=YAW_ACTION_SCALE,
        ),
    }


    hook_range = (-0.01, 0.01)
    events = {
        "randomize_block_physics": EventTermCfg(
            func=randomize_block_physics,
            mode="reset",
        ),
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
        "randomize_missing_blocks": EventTermCfg(
            func=randomize_missing_blocks,
            mode="reset",
        ),
    }


    rewards = {
        "delta_block_progress": RewardTermCfg(
            func=DeltaBlockProgressReward(),
            weight=120.0,
        ),
        "progress_towards_success_distance": RewardTermCfg(
            func=progress_towards_success_distance_reward,
            weight=2.0,
        ),
        # "torque_penalty": RewardTermCfg(
        #     func=joint_torques_l2,
        #     weight=-0.01,
        #     params={"asset_cfg": SceneEntityCfg("hook", joint_names=("hook_slide",))},
        # ),
        "action_rate": RewardTermCfg(
            func=action_rate_l2,
            weight=-0.0002,
        ),
        "successful_extract": RewardTermCfg(
            func=success_block_reward,
            weight=900.0,
        ),
        "tower_moderate_pertub" : RewardTermCfg(
            func=tower_moderate_perturbation_curriculum,
            weight=-0.2
        ),
        "tower_large_pertub" : RewardTermCfg(
            func=tower_large_perturbation_curriculum,
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

    commands = {
        "target_block": TargetBlockCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
        ),
    }


    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=_build_entities(),
            num_envs=512,
            env_spacing=4.0,
        ),
        #scale_rewards_by_dt=False,
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
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
            mujoco=MujocoCfg(timestep=0.002),
        ),
        decimation=5,
        episode_length_s=20.0,
    )


def apply_low_level_stage(stage: str) -> None:
    global MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP
    global MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY
    global MISSING_BLOCK_DOUBLE_BEGIN_STEP
    global MISSING_BLOCK_TRIPLE_BEGIN_STEP
    global RANDOM_TARGET_BLOCK_BEGIN_STEP
    global RANDOM_TARGET_BLOCK_RAMP_STEPS
    global RANDOM_TARGET_BLOCK_START_PROBABILITY
    global RANDOM_TARGET_BLOCK_END_PROBABILITY
    global RANDOM_TARGET_WITH_MISSING_BEGIN_STEP
    global FORCED_MISSING_BLOCK_COUNT

    MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 170_000
    MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
    MISSING_BLOCK_DOUBLE_BEGIN_STEP = 520_000
    MISSING_BLOCK_TRIPLE_BEGIN_STEP = 760_000
    RANDOM_TARGET_BLOCK_BEGIN_STEP = 350_000
    RANDOM_TARGET_BLOCK_RAMP_STEPS = 240_000
    RANDOM_TARGET_BLOCK_START_PROBABILITY = 0.0
    RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.15
    RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 620_000
    FORCED_MISSING_BLOCK_COUNT = None

    if stage == "fixed":
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.0
        RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 10**12
    elif stage == "target":
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
        RANDOM_TARGET_BLOCK_BEGIN_STEP = 0
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 250_000
        RANDOM_TARGET_BLOCK_START_PROBABILITY = 0.30
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.85
        RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 10**12
    elif stage == "missing1":
        MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
        MISSING_BLOCK_DOUBLE_BEGIN_STEP = 10**12
        MISSING_BLOCK_TRIPLE_BEGIN_STEP = 10**12
        RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 10**12
    elif stage == "missing2":
        MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
        MISSING_BLOCK_DOUBLE_BEGIN_STEP = 0
        MISSING_BLOCK_TRIPLE_BEGIN_STEP = 10**12
        RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 220_000
    elif stage == "missing3":
        MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
        MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.50
        MISSING_BLOCK_DOUBLE_BEGIN_STEP = 0
        MISSING_BLOCK_TRIPLE_BEGIN_STEP = 0
        RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
        RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
        RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
        RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 220_000
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
    max_iterations=10000,
  )




if __name__ == "__main__":
    entities = _build_entities()
    print(entities.keys())
    print(len(entities))
