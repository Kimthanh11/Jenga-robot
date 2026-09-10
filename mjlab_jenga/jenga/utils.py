from __future__ import annotations

import math 
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from dataclasses import dataclass, field
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from utils.constants import *

def target_command_or_none(env):
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None

    try:
        return command_manager.get_term("target_block")
    except Exception:
        return None

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

def initial_block_pos(block_name: str) -> torch.Tensor:
    for block_info in _get_block_infos():
        if block_info["name"] == block_name:
            return torch.tensor(block_info["pos"])
    raise ValueError(f"Unknown block name: {block_name}")











    


#  Utils for events
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






# rewards utils


def success_done_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    return BLOCK_SIZE[1] * success_curriculum_scale(env)


















# Others
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
    global FORCED_MISSING_PATTERN_IDS
    global FORCED_MISSING_PATTERN_OFFSET

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
    FORCED_MISSING_PATTERN_IDS = None
    FORCED_MISSING_PATTERN_OFFSET = 0

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

def target_block_entries() -> tuple[list[str], list[dict]]:
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

