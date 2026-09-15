"""Compare evaluation conditions target by target: checkpoints, controllers, protocols.

A condition is a named set of episode CSVs, optionally restricted to one controller:

    python scripts/compare_conditions.py \
        --condition "report=logs/eval/eval-146820_*-episodes.csv" \
        --condition "straight@straight=logs/eval/baseline-146824_*-episodes.csv" \
        --targets b1_1,b1_2 --seeds 1,2,3,4,5

The first condition is the reference for the paired comparisons. Two pairings are
reported, because they answer different questions:

* by target: the mean over targets of the per-target rate difference, with a t interval
  over targets. Needs no episode pairing and is the right test when conditions were
  sharded differently.
* by scenario: exact McNemar over scenario ids present in both conditions. Only valid
  when both were evaluated with the same sharding, so that equal ids mean equal resets.

Pure CSV processing; safe to run on a login node.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import math
from collections import defaultdict
from pathlib import Path

# Loaded by path: importing the mjlab_jenga package registers tasks and pulls in the
# simulator, which this script does not need.
_UTILS_PATH = Path(__file__).resolve().parents[1] / "mjlab_jenga" / "evaluation_utils.py"
_SPEC = importlib.util.spec_from_file_location("evaluation_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_utils)
wilson_interval = _utils.wilson_interval
block_layer = _utils.block_layer

# Two-sided 97.5 % quantiles of Student's t for 1..30 degrees of freedom.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)
TIMEOUT_REASONS = {"timeout", "step_cap"}


def t_quantile_975(dof: int) -> float:
    if dof < 1:
        return math.nan
    return _T975[dof - 1] if dof <= len(_T975) else 1.960


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def exact_mcnemar(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    # Integer true division: with a float factor the tail overflows beyond ~1000 pairs.
    return min(1.0, (2 * tail) / 2**n)


def paired_mean_difference(differences: list[float]) -> tuple[float, float, float]:
    """Mean of paired differences with a two-sided 95 % t interval."""
    n = len(differences)
    if n == 0:
        return math.nan, math.nan, math.nan
    mean = sum(differences) / n
    if n == 1:
        return mean, math.nan, math.nan
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    half = t_quantile_975(n - 1) * math.sqrt(variance / n)
    return mean, mean - half, mean + half


def parse_condition(value: str) -> tuple[str, str | None, list[str]]:
    if "=" not in value:
        raise ValueError(f"condition must look like NAME[@controller]=GLOB[,GLOB]: {value}")
    label, patterns = value.split("=", 1)
    controller = None
    if "@" in label:
        label, controller = label.split("@", 1)
    files: list[str] = []
    for pattern in patterns.split(","):
        matched = sorted(glob.glob(pattern.strip()))
        if not matched:
            raise ValueError(f"{label}: pattern matched no file: {pattern}")
        files.extend(matched)
    return label, controller, files


def load_condition(
    label: str,
    controller: str | None,
    files: list[str],
    targets: set[str] | None,
    seeds: set[str] | None,
) -> dict[str, dict]:
    """Rows of one condition keyed by scenario id, after filtering."""
    rows: dict[str, dict] = {}
    for path in files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if controller is not None and row["controller"] != controller:
                    continue
                if targets is not None and row["target"] not in targets:
                    continue
                if seeds is not None and row["base_seed"] not in seeds:
                    continue
                key = row["scenario_id"]
                if key in rows:
                    raise ValueError(
                        f"{label}: scenario {key} appears twice ({path}). Select files so "
                        "that each episode is counted once."
                    )
                rows[key] = row
    if not rows:
        raise ValueError(f"{label}: no rows left after filtering")
    return rows


def rates(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    success = sum(truth(r["safe_success"]) for r in rows)
    damage = sum(truth(r["tower_damage"]) for r in rows)
    timeout = sum(r["reason"] in TIMEOUT_REASONS for r in rows)
    result = {
        "n": n,
        "success": success / n,
        "damage": damage / n,
        "timeout": timeout / n,
        "success_count": success,
        "damage_count": damage,
    }
    if "reached_success" in rows[0]:
        reached = sum(truth(r["reached_success"]) for r in rows)
        result["reached"] = reached / n
        result["reached_count"] = reached
        result["damaged_after"] = sum(truth(r["damaged_after_success"]) for r in rows) / n
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument("--targets", help="Comma-separated blocks; default: all present in every condition.")
    parser.add_argument("--seeds", help="Comma-separated base seeds to keep.")
    parser.add_argument("--csv", help="Write the per-target table here.")
    args = parser.parse_args()

    targets = {t.strip() for t in args.targets.split(",")} if args.targets else None
    seeds = {s.strip() for s in args.seeds.split(",")} if args.seeds else None
    try:
        conditions = []
        for value in args.condition:
            label, controller, files = parse_condition(value)
            conditions.append((label, load_condition(label, controller, files, targets, seeds)))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    by_target: dict[str, dict[str, list[dict]]] = {}
    for label, rows in conditions:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows.values():
            grouped[row["target"]].append(row)
        by_target[label] = grouped
    common = sorted(
        set.intersection(*(set(g) for g in by_target.values())),
        key=lambda t: (block_layer(t), t),
    )
    if not common:
        parser.error("no target is present in every condition")
    labels = [label for label, _ in conditions]

    print("PER TARGET  success / damage / timeout in %, n episodes")
    header = f"{'target':7}" + "".join(f"{label:>26}" for label in labels)
    print(header)
    table_rows = []
    for target in common:
        line = f"{target:7}"
        for label in labels:
            r = rates(by_target[label][target])
            line += f"{r['success']*100:9.1f}{r['damage']*100:6.1f}{r['timeout']*100:6.1f}{r['n']:5d}"
            table_rows.append({"target": target, "layer": block_layer(target), "condition": label, **r})
        print(line)

    print("\nAGGREGATE over the common targets")
    print(f"{'condition':22} {'n':>6}  {'success [95% CI]':>22}  {'macro':>6}  {'damage [95% CI]':>22}  {'timeout':>7}")
    for label in labels:
        pooled = [row for target in common for row in by_target[label][target]]
        r = rates(pooled)
        s_low, s_high = wilson_interval(r["success_count"], r["n"])
        d_low, d_high = wilson_interval(r["damage_count"], r["n"])
        macro = sum(rates(by_target[label][t])["success"] for t in common) / len(common)
        print(
            f"{label:22} {r['n']:6d}  {r['success']*100:5.1f} [{s_low*100:5.1f}, {s_high*100:5.1f}]"
            f"  {macro*100:6.1f}  {r['damage']*100:5.1f} [{d_low*100:5.1f}, {d_high*100:5.1f}]"
            f"  {r['timeout']*100:7.1f}"
        )
        if "reached" in r:
            lost = r["reached_count"] - r["success_count"]
            print(
                f"{'':22} settling: reached {r['reached']*100:.1f} %, safe after settling "
                f"{r['success']*100:.1f} %, lost after success {lost} of {r['reached_count']}, "
                f"damaged after success {r['damaged_after']*100:.1f} %"
            )

    reference_label, reference_rows = conditions[0]
    print(f"\nPAIRED AGAINST {reference_label!r}")
    for label, rows in conditions[1:]:
        for metric in ("success", "damage"):
            differences = [
                rates(by_target[label][t])[metric] - rates(by_target[reference_label][t])[metric]
                for t in common
            ]
            mean, low, high = paired_mean_difference(differences)
            print(
                f"{label:22} {metric:8} by target (n={len(common)}): "
                f"{mean*100:+6.1f} pp  [{low*100:+6.1f}, {high*100:+6.1f}]"
            )
        shared = sorted(
            key for key in set(reference_rows) & set(rows) if rows[key]["target"] in common
        )
        if not shared:
            print(f"{label:22} by scenario: no shared scenario ids")
            continue
        only_ref = sum(
            truth(reference_rows[k]["safe_success"]) and not truth(rows[k]["safe_success"])
            for k in shared
        )
        only_this = sum(
            truth(rows[k]["safe_success"]) and not truth(reference_rows[k]["safe_success"])
            for k in shared
        )
        print(
            f"{label:22} by scenario (n={len(shared)}): {reference_label} only {only_ref}, "
            f"{label} only {only_this}, exact McNemar p={exact_mcnemar(only_ref, only_this):.3g}"
        )

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in table_rows for key in row}, key=lambda k: (k not in ("target", "layer", "condition"), k))
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
