from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
import torch
import math
from dataclasses import dataclass
from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from constants import *
from scene import *

def _rz_quat(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2))

def current_missing_block_mask(
    env: ManagerBasedRlEnv,
    block_name: str,
) -> torch.Tensor:
    mask = getattr(env, "_jenga_missing_block_mask", None)
    if mask is None or block_name not in MISSING_BLOCK_CANDIDATES:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return mask[:, MISSING_BLOCK_CANDIDATES.index(block_name)]

def target_block_entries() -> tuple[list[str], list[dict]]:
    entries = []
    names = []
    long_half = BLOCK_SIZE[1] / 2
    tip_offset = -HOOK_TIP_LOCAL_X

    for block_info in get_block_infos():
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



class TargetBlockCommand(CommandTerm):
    """Curriculum command for target-block selection and hook teleport.

    At the beginning, every env keeps the old fixed target b6_1. Once
    random_target_block_scale becomes non-zero, some reset envs sample a safe target
    block and the hook is teleported in front of that block's push face.
    """

    cfg: "TargetBlockCommandCfg"

    def __init__(self, cfg: "TargetBlockCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        names, entries = target_block_entries()
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

def target_command_or_none(env: ManagerBasedRlEnv) -> TargetBlockCommand | None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None
    try:
        return command_manager.get_term("target_block")
    except Exception:
        return None

#convert gripper to local coordinate frame of the block
def target_block_pos(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_target_pos_w()

    asset: Entity = env.scene[asset_cfg.name]
    position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    return position.squeeze(1)

def target_block_vel(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        return cmd.selected_target_vel_w()[:, :3]

    asset: Entity = env.scene[asset_cfg.name]
    velocity = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :]
    return velocity.squeeze(1)[:, :3]

def initial_tower_com_for_current_missing_pattern(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _TARGET_BLOCK_CFG,
) -> torch.Tensor:
    target_block_name = asset_cfg.name
    total_com = torch.zeros(env.num_envs, 3, device=env.device)
    present_count = torch.zeros(env.num_envs, device=env.device)

    for block_name, block_pos in INITIAL_BLOCK_POS_BY_NAME.items():
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

def tower_max_block_rotation(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_max_block_rotation()
    return torch.zeros(env.num_envs, device=env.device)

def target_block_pose(env : ManagerBasedRlEnv, asset_cfg : SceneEntityCfg = _TARGET_BLOCK_CFG) -> torch.Tensor:
    """
    extracts quaternion and position of block
    """
    cmd = target_command_or_none(env)
    if cmd is not None and asset_cfg.name == _TARGET_BLOCK_CFG.name:
        pose = cmd.selected_target_pose_w()
        return pose[:, :3], pose[:, 3:7]

    asset: Entity = env.scene[asset_cfg.name]
    pose = asset.data.body_link_pose_w[:, asset_cfg.body_ids, :]
    pose = pose.squeeze(1)
    block_pos = pose[:, :3]
    block_quat = pose[:, 3:7]
    return block_pos, block_quat

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

def hook_joint_pos_relative_to_target_home(env: ManagerBasedRlEnv) -> torch.Tensor:
    hook_joint_pos = hook_joint_pos_ordered(env)
    cmd = target_command_or_none(env)
    if cmd is None:
        return hook_joint_pos
    return hook_joint_pos - cmd.selected_hook_home()

def target_selection_features(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
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
    cmd = target_command_or_none(env)
    if cmd is None:
        return torch.zeros(env.num_envs, 8, device=env.device)
    return cmd.selected_support_presence()

def target_contact_face_y(env: ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is None:
        return torch.full((env.num_envs,), CONTACT_FACE_Y, device=env.device)
    return cmd.selected_contact_face_y()

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

def get_block_ref_pos(env : ManagerBasedRlEnv) -> torch.Tensor:
    cmd = target_command_or_none(env)
    if cmd is not None:
        return cmd.selected_ref_pos_w()

    ref1_block_pos = target_block_pos(env, _REF_BLOCK_1_CFG)
    ref2_block_pos = target_block_pos(env, _REF_BLOCK_2_CFG)
    ref_block_state_mean = (ref1_block_pos + ref2_block_pos) / 2
    return ref_block_state_mean



def target_support_presence(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Which blocks around the target are still there (8 fixed slots).

    Without this the actor cannot tell an intact tower from one missing a supporting
    block, while being penalized for instability and terminated on damage. Only the
    critic saw the tower, so it could recognise danger during training that the actor
    could not react to at execution time.
    """
    cmd = target_command_or_none(env)
    if cmd is None:
        return torch.zeros(env.num_envs, 8, device=env.device)
    return cmd.selected_support_presence()

def hook_joint_pos_relative_to_target_home(env: ManagerBasedRlEnv) -> torch.Tensor:
    hook_joint_pos = hook_joint_pos_ordered(env)
    cmd = target_command_or_none(env)
    if cmd is None:
        return hook_joint_pos
    return hook_joint_pos - cmd.selected_hook_home()

