from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_failure_modes.py"
SPEC = importlib.util.spec_from_file_location("analyze_failure_modes", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
afm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(afm)

COLUMNS = [
    "controller", "target", "layer", "reason", "extracted", "safe_success",
    "tower_damage", "tower_xy_max", "tower_z_max", "tower_rot_deg_max",
]


def row(controller="policy", target="b6_1", layer=6, reason="timeout", extracted=False,
        success=False, damage=False, xy=0.001, z=0.001, rot=1.0) -> dict:
    return {
        "controller": controller, "target": target, "layer": layer, "reason": reason,
        "extracted": extracted, "safe_success": success, "tower_damage": damage,
        "tower_xy_max": xy, "tower_z_max": z, "tower_rot_deg_max": rot,
    }


class AnalyzeFailureModesTest(unittest.TestCase):
    def test_outcome_precedence_and_reasons(self) -> None:
        self.assertEqual(afm.outcome(row(success="True", reason="success")), "success")
        self.assertEqual(afm.outcome(row(damage="True", reason="tower_damage")), "damage")
        self.assertEqual(afm.outcome(row(reason="timeout")), "timeout")
        self.assertEqual(afm.outcome(row(reason="step_cap")), "timeout")
        self.assertEqual(afm.outcome(row(reason="terminated")), "other")
        # Damage after a reached success in a settling run is damage, not success.
        self.assertEqual(afm.outcome(row(success="False", damage="True", reason="settled")), "damage")

    def test_tripped_limits_are_strict_and_combinable(self) -> None:
        self.assertEqual(afm.tripped_limits(row(xy=0.030)), ("xy",))
        self.assertEqual(afm.tripped_limits(row(xy=0.030, z=0.016, rot=16.0)), ("xy", "z", "rot"))
        self.assertEqual(afm.tripped_limits(row(xy=0.025, z=0.015, rot=15.0)), ())

    def test_summarize_counts_medians_and_timeout_extraction(self) -> None:
        rows = [
            row(success="True", reason="success"),
            row(damage="True", reason="tower_damage", xy=0.030, z=0.001, rot=2.0),
            row(damage="True", reason="tower_damage", xy=0.050, z=0.020, rot=3.0),
            row(damage="True", reason="tower_damage", xy=0.028, z=0.001, rot=20.0),
            row(reason="timeout", extracted="True"),
            row(reason="step_cap", extracted="False"),
        ]
        summary = afm.summarize(rows)
        self.assertEqual(summary["n"], 6)
        self.assertEqual(summary["counts"], {"success": 1, "damage": 3, "timeout": 2, "other": 0})
        self.assertEqual(summary["damage_combos"], {("xy",): 1, ("xy", "z"): 1, ("xy", "rot"): 1})
        self.assertAlmostEqual(summary["damage_median"]["xy_mm"], 30.0)
        self.assertAlmostEqual(summary["damage_median"]["z_mm"], 1.0)
        self.assertAlmostEqual(summary["damage_median"]["rot_deg"], 3.0)
        self.assertEqual((summary["timeout_extracted"], summary["timeout_n"]), (1, 2))
        text = afm.format_summary("policy", summary)
        self.assertIn("damage= 50.0%", text)
        self.assertIn("limits xy+z", text)

    def test_no_damage_or_timeout_has_no_detail_lines(self) -> None:
        text = afm.format_summary("x", afm.summarize([row(success="True", reason="success")]))
        self.assertEqual(len(text.splitlines()), 1)

    def test_csv_loading_filtering_and_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "eval-1-episodes.csv"
            second = Path(directory) / "baseline-2-episodes.csv"
            for path, controller in ((first, "policy"), (second, "straight")):
                with path.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=COLUMNS + ["extra"])
                    writer.writeheader()
                    for layer, target in ((6, "b6_1"), (7, "b7_1"), (2, "b2_1")):
                        writer.writerow({**row(controller, target, layer), "extra": 1})
            rows = afm.load_rows([str(Path(directory) / "*-episodes.csv")])
            self.assertEqual(len(rows), 6)
            groups = afm.group_rows(rows, layers={6, 7, 8})
            self.assertEqual(sorted(groups), ["policy", "straight"])
            self.assertEqual(len(groups["policy"]), 2)
            groups = afm.group_rows(rows, targets={"b2_1"}, by_file=True)
            self.assertEqual(len(groups), 2)
            with self.assertRaises(ValueError):
                afm.load_rows([str(Path(directory) / "missing-*.csv")])
            bad = Path(directory) / "bad.csv"
            bad.write_text("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                afm.load_rows([str(bad)])


if __name__ == "__main__":
    unittest.main()
