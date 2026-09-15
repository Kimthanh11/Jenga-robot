from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "compare_conditions.py"
SPEC = importlib.util.spec_from_file_location("compare_conditions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_conditions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_conditions)


class CompareConditionsTest(unittest.TestCase):
    def test_paired_difference_uses_a_t_interval_over_targets(self) -> None:
        mean, low, high = compare_conditions.paired_mean_difference([0.1, 0.3])
        self.assertAlmostEqual(mean, 0.2)
        # Standard error 0.1 with one degree of freedom, t = 12.706.
        self.assertAlmostEqual(low, 0.2 - 1.2706, places=4)
        self.assertAlmostEqual(high, 0.2 + 1.2706, places=4)

    def test_t_quantile_falls_back_to_normal_for_many_targets(self) -> None:
        self.assertAlmostEqual(compare_conditions.t_quantile_975(19), 2.093)
        self.assertAlmostEqual(compare_conditions.t_quantile_975(100), 1.960)

    def test_exact_mcnemar_counts_only_discordant_pairs(self) -> None:
        self.assertEqual(compare_conditions.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(compare_conditions.exact_mcnemar(0, 5), 2 / 32)
        self.assertEqual(
            compare_conditions.exact_mcnemar(3, 7), compare_conditions.exact_mcnemar(7, 3)
        )

    def test_exact_mcnemar_handles_thousands_of_discordant_pairs(self) -> None:
        self.assertLess(compare_conditions.exact_mcnemar(195, 1596), 1e-100)
        self.assertAlmostEqual(compare_conditions.exact_mcnemar(1000, 1000), 1.0, places=6)

    def test_condition_label_can_select_a_controller(self) -> None:
        pattern = str(MODULE_PATH)
        label, controller, files = compare_conditions.parse_condition(f"base@tap={pattern}")
        self.assertEqual((label, controller, files), ("base", "tap", [pattern]))
        with self.assertRaisesRegex(ValueError, "matched no file"):
            compare_conditions.parse_condition("x=does/not/exist-*.csv")


if __name__ == "__main__":
    unittest.main()
