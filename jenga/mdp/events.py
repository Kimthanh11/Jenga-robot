from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr import geom_friction, pseudo_inertia
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

import torch 
import math

from constants import RESET_FRICTION_SLIDING_RANGE, RESET_FRICTION_TORSIONAL_RANGE, RESET_FRICTION_ROLLING_RANGE, RESET_DENSITY_RANDOMIZATION, FORCED_MISSING_PATTERN_OFFSET, FORCED_MISSING_PATTERN_IDS, MISSING_BLOCK_PARK_OFFSET, MISSING_BLOCK_CANDIDATES, MISSING_BLOCK_PARK_SPACING, MISSING_BLOCK_PATTERNS
from utils import *

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

    for block_info in get_block_infos():
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
    active_pattern_ids = active_missing_pattern_ids(env)

    if FORCED_MISSING_PATTERN_IDS is not None:
        if active_pattern_ids.numel() == 0:
            raise ValueError("FORCED_MISSING_PATTERN_IDS must not be empty")
        choice = torch.remainder(
            env_ids + FORCED_MISSING_PATTERN_OFFSET,
            active_pattern_ids.numel(),
        )
        pattern_ids = active_pattern_ids[choice]
    elif active_pattern_ids.numel() > 0 and missing_probability.item() > 0.0:
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

    missing_mask = ensure_missing_block_state(env)
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