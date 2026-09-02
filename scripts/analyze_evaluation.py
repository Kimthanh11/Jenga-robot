"""Summarize policy and baseline episode CSVs, including paired comparisons."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from mjlab_jenga.evaluation_utils import wilson_interval


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(discordant_a, discordant_b) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def _rate(rows: list[dict], key: str = "safe_success") -> tuple[int, int, float]:
    successes = sum(_truth(row[key]) for row in rows)
    return successes, len(rows), successes / len(rows)


def _load(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for value in paths:
        path = Path(value)
        with path.open(newline="") as file:
            file_rows = list(csv.DictReader(file))
        if not file_rows:
            raise ValueError(f"No episode rows in {path}")
        required = {"controller", "scenario_id", "target", "target_set", "missing_level"}
        missing = required - set(file_rows[0])
        if missing:
            raise ValueError(f"{path} is not an episode CSV; missing columns {sorted(missing)}")
        rows.extend(file_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_csv", nargs="+")
    parser.add_argument("--reference", default="policy")
    args = parser.parse_args()

    try:
        rows = _load(args.episode_csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["controller"], row["scenario_id"])
        if key in indexed:
            parser.error(
                f"Duplicate controller/scenario pair {key}; do not combine repeated runs."
            )
        indexed[key] = row

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    target_groups: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        level = int(row["missing_level"])
        groups[(row["controller"], row["target_set"], level)].append(row)
        target_groups[(row["controller"], row["target_set"], level, row["target"])].append(row)

    print("\nAGGREGATE (micro rate; macro gives every target equal weight)")
    print("controller target_set       miss      n  success [95% CI]       macro")
    aggregate_rates: dict[tuple[str, str, int], float] = {}
    for key, values in sorted(groups.items()):
        successes, total, rate = _rate(values)
        low, high = wilson_interval(successes, total)
        per_target = [
            _rate(target_rows)[2]
            for target_key, target_rows in target_groups.items()
            if target_key[:3] == key
        ]
        macro = sum(per_target) / len(per_target)
        aggregate_rates[key] = macro
        controller, target_set, level = key
        print(
            f"{controller:10} {target_set:16} {level:4d} {total:6d}  "
            f"{rate:6.3f} [{low:5.3f}, {high:5.3f}]  {macro:6.3f}"
        )

    print("\nPER TARGET")
    print("controller target_set       miss target      n  success  damage")
    for key, values in sorted(target_groups.items()):
        success = _rate(values)[2]
        damage = _rate(values, "tower_damage")[2]
        controller, target_set, level, target = key
        print(
            f"{controller:10} {target_set:16} {level:4d} {target:6} "
            f"{len(values):6d}  {success:6.3f}  {damage:6.3f}"
        )

    print("\nGENERALIZATION GAP (macro trained-legal minus heldout-legal)")
    found_gap = False
    controllers = sorted({row["controller"] for row in rows})
    levels = sorted({int(row["missing_level"]) for row in rows})
    for controller in controllers:
        for level in levels:
            train_key = (controller, "trained-legal", level)
            heldout_key = (controller, "heldout-legal", level)
            if train_key in aggregate_rates and heldout_key in aggregate_rates:
                found_gap = True
                gap = aggregate_rates[train_key] - aggregate_rates[heldout_key]
                print(f"{controller:10} missing={level}: {gap:+.3f}")
    if not found_gap:
        print("not available: evaluate both trained-legal and heldout-legal")

    by_controller: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_controller[row["controller"]][row["scenario_id"]] = row
    reference = by_controller.get(args.reference)
    print(f"\nPAIRED AGAINST {args.reference!r}")
    if reference is None:
        print("reference controller not present")
        return
    for controller, candidates in sorted(by_controller.items()):
        if controller == args.reference:
            continue
        shared = sorted(set(reference) & set(candidates))
        if not shared:
            print(f"{controller:10}: no shared scenario_id values")
            continue
        reference_only = 0
        candidate_only = 0
        both = 0
        neither = 0
        for scenario in shared:
            a = _truth(reference[scenario]["safe_success"])
            b = _truth(candidates[scenario]["safe_success"])
            both += a and b
            reference_only += a and not b
            candidate_only += b and not a
            neither += not a and not b
        p_value = _exact_mcnemar(reference_only, candidate_only)
        print(
            f"{controller:10}: n={len(shared)} both={both} "
            f"{args.reference}_only={reference_only} {controller}_only={candidate_only} "
            f"neither={neither} exact_McNemar_p={p_value:.4g}"
        )


if __name__ == "__main__":
    main()
