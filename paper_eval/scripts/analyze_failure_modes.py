"""Break down how episodes end, per controller, from episode CSVs (pure Python, CSV only).

    python paper_eval/scripts/analyze_failure_modes.py "logs/paper_eval/eval-*-episodes.csv" \
        "logs/paper_eval/baseline-*-episodes.csv" --layers 6,7,8

For every controller (or every input file with --by-file) the report gives the share of
episodes ending in success, damage and timeout; among damaged episodes which damage limits
were exceeded (each combination of horizontal, vertical and rotation) with the median of
each maximum displacement; and among timeouts whether the extraction distance was reached.

Units assumed, as written by evaluate_low_level.py and evaluate_scripted_baselines.py:
tower_xy_max and tower_z_max in metres (compared with 0.025 m and 0.015 m, printed in mm),
tower_rot_deg_max in degrees (compared with 15 deg). A damaged episode ends at its first
crossing of a limit, so in the regular protocol one limit is usually the one that tripped;
with --settle-steps the maxima run on to the end of the settling period.
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from collections import Counter, defaultdict

DAMAGE_XY_M = 0.025
DAMAGE_Z_M = 0.015
DAMAGE_ROT_DEG = 15.0
TIMEOUT_REASONS = {"timeout", "step_cap"}
REQUIRED_COLUMNS = {
    "controller", "target", "layer", "reason", "extracted", "safe_success",
    "tower_damage", "tower_xy_max", "tower_z_max", "tower_rot_deg_max",
}


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def outcome(row: dict) -> str:
    """success, damage, timeout or other; success wins, then damage, then timeout."""
    if truth(row["safe_success"]):
        return "success"
    if truth(row["tower_damage"]):
        return "damage"
    if row["reason"] in TIMEOUT_REASONS:
        return "timeout"
    return "other"


def tripped_limits(row: dict) -> tuple[str, ...]:
    """Which damage limits the recorded maxima exceed: any of xy, z, rot; () if none."""
    tripped = []
    if float(row["tower_xy_max"]) > DAMAGE_XY_M:
        tripped.append("xy")
    if float(row["tower_z_max"]) > DAMAGE_Z_M:
        tripped.append("z")
    if float(row["tower_rot_deg_max"]) > DAMAGE_ROT_DEG:
        tripped.append("rot")
    return tuple(tripped)


def summarize(rows: list[dict]) -> dict:
    """Counts and medians for one group of episodes (all rows are used as given)."""
    total = len(rows)
    counts = Counter(outcome(row) for row in rows)
    damaged = [row for row in rows if outcome(row) == "damage"]
    timeouts = [row for row in rows if outcome(row) == "timeout"]
    combos = Counter(tripped_limits(row) for row in damaged)
    summary = {
        "n": total,
        "counts": {name: counts.get(name, 0) for name in ("success", "damage", "timeout", "other")},
        "damage_combos": dict(combos),
        "damage_median": None,
        "timeout_extracted": sum(truth(row["extracted"]) for row in timeouts),
        "timeout_n": len(timeouts),
    }
    if damaged:
        summary["damage_median"] = {
            "xy_mm": 1000.0 * statistics.median(float(r["tower_xy_max"]) for r in damaged),
            "z_mm": 1000.0 * statistics.median(float(r["tower_z_max"]) for r in damaged),
            "rot_deg": statistics.median(float(r["tower_rot_deg_max"]) for r in damaged),
        }
    return summary


def format_summary(label: str, summary: dict) -> str:
    n = summary["n"]
    counts = summary["counts"]

    def pct(count: int, of: int) -> str:
        return f"{100.0 * count / of:5.1f}%" if of else "  n/a "

    lines = [
        f"{label}: n={n} success={pct(counts['success'], n)} damage={pct(counts['damage'], n)} "
        f"timeout={pct(counts['timeout'], n)} other={pct(counts['other'], n)}"
    ]
    damaged = counts["damage"]
    if damaged:
        median = summary["damage_median"]
        lines.append(
            f"  damaged (n={damaged}): median max xy={median['xy_mm']:.1f} mm, "
            f"z={median['z_mm']:.1f} mm, rot={median['rot_deg']:.1f} deg"
        )
        for combo, count in sorted(summary["damage_combos"].items(), key=lambda item: -item[1]):
            name = "+".join(combo) if combo else "none exceeded"
            lines.append(f"    limits {name:<12} {count:5d} ({pct(count, damaged).strip()} of damaged)")
    if summary["timeout_n"]:
        lines.append(
            f"  timeouts (n={summary['timeout_n']}): extraction reached in "
            f"{summary['timeout_extracted']} ({pct(summary['timeout_extracted'], summary['timeout_n']).strip()})"
        )
    return "\n".join(lines)


def load_rows(paths: list[str]) -> list[tuple[str, dict]]:
    """Read all CSVs matching the given paths or globs as (file, row) pairs."""
    files: list[str] = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"no file matches {pattern}")
        files.extend(matches)
    rows: list[tuple[str, dict]] = []
    for path in files:
        with open(path, newline="") as file:
            reader = csv.DictReader(file)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is not an episode CSV; missing columns {sorted(missing)}")
            rows.extend((path, row) for row in reader)
    return rows


def group_rows(
    rows: list[tuple[str, dict]],
    *,
    layers: set[int] | None = None,
    targets: set[str] | None = None,
    by_file: bool = False,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path, row in rows:
        if layers is not None and int(row["layer"]) not in layers:
            continue
        if targets is not None and row["target"] not in targets:
            continue
        label = f"{row['controller']} [{path}]" if by_file else row["controller"]
        groups[label].append(row)
    return dict(groups)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode_csv", nargs="+", help="episode CSV files or globs")
    parser.add_argument("--layers", help="Comma-separated layers to keep, e.g. 6,7,8.")
    parser.add_argument("--targets", help="Comma-separated blocks to keep, e.g. b6_1,b7_2.")
    parser.add_argument("--by-file", action="store_true", help="Report every input file separately.")
    args = parser.parse_args(argv)
    try:
        layers = {int(item) for item in args.layers.split(",") if item.strip()} if args.layers else None
        targets = {item.strip() for item in args.targets.split(",") if item.strip()} if args.targets else None
        rows = load_rows(args.episode_csv)
    except ValueError as exc:
        parser.error(str(exc))
    groups = group_rows(rows, layers=layers, targets=targets, by_file=args.by_file)
    if not groups:
        parser.error("no episodes left after filtering")
    for label, group in sorted(groups.items()):
        print(format_summary(label, summarize(group)))


if __name__ == "__main__":
    main()
