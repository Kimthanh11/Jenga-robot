"""Shared definitions for reproducible Jenga evaluation runs."""

from __future__ import annotations

import csv
import math
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path


TARGET_SELECTOR_NAMES = (
    "trained",
    "trained-legal",
    "heldout-legal",
    "legal",
    "tower",
)


def git_commit() -> str:
    """Return the checked-out commit without making evaluation depend on Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_rows(path: str | Path | None, rows: Sequence[dict]) -> None:
    """Append complete rows and create a header for a new result file."""
    if path is None or not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output.exists()
    with output.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def block_layer(name: str) -> int:
    """Return the one-based tower layer encoded in a block name such as b4_2."""
    try:
        return int(name[1:].split("_", maxsplit=1)[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid block name: {name!r}") from exc


def target_groups(cfg) -> dict[str, tuple[str, ...]]:
    """Build the named target sets used by training and evaluation."""
    tower = tuple(
        f"b{layer}_{slot}"
        for layer in range(1, cfg.LAYERS + 1)
        for slot in range(1, cfg.BLOCKS_PER_LAYER + 1)
    )
    legal = tuple(name for name in tower if block_layer(name) < cfg.LAYERS)
    trained = tuple(cfg.RANDOM_TARGET_BLOCK_NAMES)
    trained_set = set(trained)
    trained_legal = tuple(name for name in trained if name in legal)
    heldout_legal = tuple(name for name in legal if name not in trained_set)
    return {
        "trained": trained,
        "trained-legal": trained_legal,
        "heldout-legal": heldout_legal,
        "legal": legal,
        "tower": tower,
    }


def resolve_targets(value: str, cfg) -> tuple[str, tuple[str, ...]]:
    """Resolve a named target set or a comma-separated explicit block list."""
    selector = value.strip().lower().replace("_", "-")
    groups = target_groups(cfg)
    if selector == "all":
        raise ValueError(
            "Target selector 'all' is ambiguous. Use trained, trained-legal, "
            "heldout-legal, legal, or tower."
        )
    if selector in groups:
        return selector, groups[selector]

    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    if not requested:
        raise ValueError("At least one target block is required.")
    unknown = sorted(set(requested) - set(groups["tower"]))
    if unknown:
        raise ValueError(f"Unknown target blocks: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("Target list contains duplicates.")
    return "explicit", requested


def parse_int_csv(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated integer list without duplicates."""
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("At least one integer value is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate values are not allowed: {value}")
    return values


def valid_missing_pattern_ids(cfg, target: str, missing_level: int) -> tuple[int, ...]:
    """Return exact-level patterns that leave the selected target in the tower."""
    if missing_level < 0:
        raise ValueError("missing_level must be non-negative")
    pattern_ids = tuple(
        idx
        for idx, pattern in enumerate(cfg.MISSING_BLOCK_PATTERNS)
        if len(pattern) == missing_level and target not in pattern
    )
    if not pattern_ids:
        raise ValueError(
            f"No valid level-{missing_level} missing pattern leaves {target} present."
        )
    return pattern_ids


def configure_evaluation_case(
    cfg,
    target: str,
    missing_level: int,
    *,
    pattern_offset: int = 0,
) -> tuple[int, ...]:
    """Pin curriculum globals for a balanced, target-valid evaluation case."""
    pattern_ids = valid_missing_pattern_ids(cfg, target, missing_level)
    cfg.FORCED_MISSING_BLOCK_COUNT = missing_level
    cfg.FORCED_MISSING_PATTERN_IDS = pattern_ids
    cfg.FORCED_MISSING_PATTERN_OFFSET = pattern_offset
    cfg.MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = -1
    cfg.MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 1
    probability = 1.0 if missing_level > 0 else 0.0
    cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = probability
    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = probability
    cfg.RANDOM_TARGET_BLOCK_BEGIN_STEP = 10**12
    cfg.RANDOM_TARGET_BLOCK_START_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 0.0
    return pattern_ids


def validate_evaluation_reset(env, cfg, target: str, active_count: int) -> None:
    """Fail loudly if a forced target or missing pattern was not applied."""
    command = env.command_manager.get_term("target_block")
    selected = [
        command._all_names[index]
        for index in command.selected_block_idx[:active_count].tolist()
    ]
    wrong_targets = [name for name in selected if name != target]
    if wrong_targets:
        raise RuntimeError(
            f"Forced target {target} was not applied; selected {selected}."
        )

    pattern_ids = env._jenga_missing_pattern_id[:active_count].tolist()
    for pattern_id in pattern_ids:
        pattern = cfg.MISSING_BLOCK_PATTERNS[pattern_id]
        if len(pattern) != cfg.FORCED_MISSING_BLOCK_COUNT:
            raise RuntimeError(
                f"Expected {cfg.FORCED_MISSING_BLOCK_COUNT} missing blocks, "
                f"got pattern {pattern}."
            )
        if target in pattern:
            raise RuntimeError(f"Target {target} was removed by missing pattern {pattern}.")


def pattern_label(cfg, pattern_id: int) -> str:
    pattern = cfg.MISSING_BLOCK_PATTERNS[pattern_id]
    return "none" if not pattern else "+".join(pattern)


def evaluation_seed(base_seed: int, batch_index: int) -> int:
    """Derive a stable reset seed for one vectorized evaluation batch."""
    return int((base_seed + 1_000_003 * batch_index) % 2_147_483_647)


def scenario_id(
    target: str,
    missing_level: int,
    base_seed: int,
    batch_index: int,
    env_index: int,
) -> str:
    return (
        f"{target}-m{missing_level}-s{base_seed}-"
        f"b{batch_index}-e{env_index}"
    )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a Bernoulli proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return centre - half_width, centre + half_width


def mean(rows: Sequence[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def bool_count(rows: Iterable[dict], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def summarize_episode_rows(rows: list[dict]) -> dict:
    """Aggregate one controller/target/missing-level result group."""
    if not rows:
        raise ValueError("Cannot summarize an empty episode group.")
    successes = bool_count(rows, "safe_success")
    extractions = bool_count(rows, "extracted")
    damages = bool_count(rows, "tower_damage")
    total = len(rows)
    success_low, success_high = wilson_interval(successes, total)
    extraction_low, extraction_high = wilson_interval(extractions, total)
    damage_low, damage_high = wilson_interval(damages, total)
    first = rows[0]
    return {
        "controller": first["controller"],
        "checkpoint": first["checkpoint"],
        "commit": first["commit"],
        "target_set": first["target_set"],
        "target": first["target"],
        "layer": first["layer"],
        "is_trained": first["is_trained"],
        "is_legal": first["is_legal"],
        "missing_level": first["missing_level"],
        "episodes": total,
        "extraction_rate": extractions / total,
        "extraction_ci95_low": extraction_low,
        "extraction_ci95_high": extraction_high,
        "success_rate": successes / total,
        "success_ci95_low": success_low,
        "success_ci95_high": success_high,
        "tower_damage_rate": damages / total,
        "tower_damage_ci95_low": damage_low,
        "tower_damage_ci95_high": damage_high,
        "progress_final_mean": mean(rows, "progress_final"),
        "progress_max_mean": mean(rows, "progress_max"),
        "episode_length_mean": mean(rows, "steps"),
        "tower_xy_final_mean": mean(rows, "tower_xy_final"),
        "tower_xy_max_mean": mean(rows, "tower_xy_max"),
        "tower_xy_recovery_mean": mean(rows, "tower_xy_recovery"),
        "tower_z_final_mean": mean(rows, "tower_z_final"),
        "tower_z_max_mean": mean(rows, "tower_z_max"),
        "tower_z_recovery_mean": mean(rows, "tower_z_recovery"),
        "tower_rot_deg_final_mean": mean(rows, "tower_rot_deg_final"),
        "tower_rot_deg_max_mean": mean(rows, "tower_rot_deg_max"),
        "tower_rot_deg_recovery_mean": mean(rows, "tower_rot_deg_recovery"),
        "contact_rate": mean(rows, "contact_rate"),
        "contact_force_mean": mean(rows, "contact_force_mean"),
        "contact_force_max_mean": mean(rows, "contact_force_max"),
        "stuck_rate": mean(rows, "stuck_rate"),
        "stop_rate": mean(rows, "stop_rate"),
        "retreat_rate": mean(rows, "retreat_rate"),
    }
