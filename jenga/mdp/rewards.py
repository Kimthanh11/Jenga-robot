import torch 
import math
from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.sensor import ContactSensor
from constants import *
from utils import get_block_infos, _rz_quat, _INITIAL_BLOCK_POS_BY_NAME, hook_joint_pos_ordered
from mdp.commands import *
from mdp.actions import _linear_curriculum_scale, target_block_pose, block_point_to_world, hook_slide_targets_for_tip_world, push_velocity_target, yaw_curriculum_scale

def target_block_pos(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_target_pos_w()

    asset: Entity = env.scene[asset_cfg.name]
    position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    return position.squeeze(1)

def get_block_ref_pos(env : ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_ref_pos_w()

    ref1_block_pos = target_block_pos(env, _REF_BLOCK_1_CFG)
    ref2_block_pos = target_block_pos(env, _REF_BLOCK_2_CFG)
    ref_block_state_mean = (ref1_block_pos + ref2_block_pos) / 2
    return ref_block_state_mean

def target_block_relative_movement(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_relative_movement()

    ref_pos = get_block_ref_pos(env)
    target_pos = target_block_pos(env, asset_cfg)

    current_rel = target_pos - ref_pos
    return current_rel - START_TARGET_REL_POS.to(current_rel.device)

def block_progress(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_progress()

    movement_rel = target_block_relative_movement(env, asset_cfg)

    extraction_direction = torch.tensor(
        [-1.0, 0.0, 0.0],
        device=movement_rel.device,
    )
    progress = torch.sum(movement_rel * extraction_direction, dim=-1)

    return progress

def success_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = min(curriculum_step(env) / SUCCESS_CURRICULUM_STEPS, 1.0)
    scale = SUCCESS_CURRICULUM_START + (
        SUCCESS_CURRICULUM_END - SUCCESS_CURRICULUM_START
    ) * progress
    return torch.tensor(scale, device=env.device)

def success_done_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    return BLOCK_SIZE[1] * success_curriculum_scale(env)

def target_task_quat_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_task_quat_w()

    return torch.tensor(
        _rz_quat(math.pi),
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(env.num_envs, 1)

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

def target_block_vel_in_task_frame(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    vel_world = target_block_vel(env, asset_cfg)
    task_quat_world = target_task_quat_w(env, asset_cfg)
    return quat_apply_inverse(task_quat_world, vel_world)

def target_block_vel(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_target_vel_w()[:, :3]

    asset: Entity = env.scene[asset_cfg.name]
    velocity = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :]
    return velocity.squeeze(1)[:, :3]

def stuck_contact_signal(env: ManagerBasedRlEnv) -> torch.Tensor:
    contact = hook_contact_found(env) > 0.0
    force_high = hook_contact_force_norm(env) > STUCK_CONTACT_FORCE_THRESHOLD
    extraction_speed = target_block_vel_in_task_frame(env)[:, 0]
    nearly_stationary = torch.abs(extraction_speed) < STUCK_BLOCK_SPEED_THRESHOLD
    return (contact & force_high & nearly_stationary).float()

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

def action_magnitude_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=-1)

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

def target_extraction_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
    progress = block_progress(env)
    return progress > success_done_distance(env)

def get_com_tower(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
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
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_tower_shift()

    current = get_com_tower(env, asset_cfg)
    start = initial_tower_com_for_current_missing_pattern(env, asset_cfg)
    movement = current - start
    horizontal_shift = torch.norm(movement[:, :2], dim=-1)
    return horizontal_shift

def tower_max_block_horizontal_shift(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_horizontal_shift()
    return tower_com_shift(env)

def tower_max_block_vertical_shift(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_vertical_shift()
    return torch.zeros(env.num_envs, device=env.device)

def tower_stable_for_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (
        (tower_max_block_horizontal_shift(env) < TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT)
        & (tower_max_block_vertical_shift(env) < TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT)
        & (tower_max_block_rotation(env) < TOWER_SUCCESS_MAX_BLOCK_ROTATION)
    )

def success_block_extract(env: ManagerBasedRlEnv) -> torch.Tensor:
    return target_extraction_reached(env) & tower_stable_for_success(env)

def success_block_reward(env : ManagerBasedRlEnv) -> torch.Tensor:
    return success_block_extract(env).float()

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

def tower_max_block_rotation(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_rotation()
    return torch.zeros(env.num_envs, device=env.device)


def tower_damage(env : ManagerBasedRlEnv) -> torch.Tensor:
    return (
        (tower_max_block_horizontal_shift(env) > TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT)
        | (tower_max_block_vertical_shift(env) > TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT)
        | (tower_max_block_rotation(env) > TOWER_DAMAGE_MAX_BLOCK_ROTATION)
    )

def tower_damage_signal(env: ManagerBasedRlEnv) -> torch.Tensor:
    return tower_damage(env).float()

def touch_curriculum_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _linear_curriculum_scale(
        env,
        TOUCH_CURRICULUM_START,
        TOUCH_CURRICULUM_END,
        TOUCH_CURRICULUM_BEGIN_STEP,
        TOUCH_CURRICULUM_STEPS,
    )

def hook_x_position(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HOOK1_CFG,
) -> torch.Tensor:
    del asset_cfg
    return hook_joint_pos_ordered(env)[:, 0]

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

def target_contact_face_y(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is None:
        return torch.full((env.num_envs,), CONTACT_FACE_Y, device=env.device)
    return cmd.selected_contact_face_y()

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

def action_norm(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.norm(env.action_manager.action, dim=-1)

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
        cmd = target_command_or_none(env)
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