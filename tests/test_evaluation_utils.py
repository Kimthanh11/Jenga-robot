from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[1] / "mjlab_jenga" / "evaluation_utils.py"
SPEC = importlib.util.spec_from_file_location("evaluation_utils", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation_utils)

configure_evaluation_case = evaluation_utils.configure_evaluation_case
evaluation_seed = evaluation_utils.evaluation_seed
resolve_targets = evaluation_utils.resolve_targets
scenario_id = evaluation_utils.scenario_id
target_groups = evaluation_utils.target_groups
valid_missing_pattern_ids = evaluation_utils.valid_missing_pattern_ids
wilson_interval = evaluation_utils.wilson_interval


class EvaluationUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimpleNamespace(
            LAYERS=9,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(
                "b2_1",
                "b2_2",
                "b2_3",
                "b3_1",
                "b9_1",
                "b9_2",
                "b9_3",
            ),
            MISSING_BLOCK_PATTERNS=(
                (),
                ("b4_1",),
                ("b4_3",),
                ("b5_2",),
                ("b4_1", "b5_2"),
                ("b4_3", "b5_2"),
                ("b3_2", "b5_1"),
                ("b3_3", "b5_3"),
                ("b3_2", "b4_1", "b5_2"),
                ("b3_3", "b4_3", "b5_2"),
                ("b4_1", "b5_2", "b8_2"),
            ),
        )

    def test_target_groups_separate_training_and_heldout_blocks(self) -> None:
        groups = target_groups(self.cfg)
        self.assertEqual(len(groups["tower"]), 27)
        self.assertEqual(len(groups["legal"]), 24)
        self.assertEqual(groups["trained-legal"], ("b2_1", "b2_2", "b2_3", "b3_1"))
        self.assertEqual(len(groups["heldout-legal"]), 20)
        self.assertNotIn("b9_1", groups["legal"])
        self.assertNotIn("b2_1", groups["heldout-legal"])

    def test_all_is_rejected_as_ambiguous(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_targets("all", self.cfg)

    def test_explicit_targets_are_validated(self) -> None:
        name, targets = resolve_targets("b1_1,b8_3", self.cfg)
        self.assertEqual(name, "explicit")
        self.assertEqual(targets, ("b1_1", "b8_3"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_targets("b10_1", self.cfg)

    def test_missing_patterns_keep_target_present(self) -> None:
        pattern_ids = valid_missing_pattern_ids(self.cfg, "b5_2", 1)
        self.assertEqual(pattern_ids, (1, 2))
        self.assertTrue(
            all("b5_2" not in self.cfg.MISSING_BLOCK_PATTERNS[i] for i in pattern_ids)
        )

    def test_case_configuration_uses_exact_valid_patterns(self) -> None:
        pattern_ids = configure_evaluation_case(self.cfg, "b4_1", 2, pattern_offset=3)
        self.assertEqual(pattern_ids, (5, 6, 7))
        self.assertEqual(self.cfg.FORCED_MISSING_BLOCK_COUNT, 2)
        self.assertEqual(self.cfg.FORCED_MISSING_PATTERN_IDS, pattern_ids)
        self.assertEqual(self.cfg.FORCED_MISSING_PATTERN_OFFSET, 3)
        self.assertEqual(self.cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY, 1.0)
        self.assertEqual(self.cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY, 0.0)

    def test_scenario_identity_and_seed_are_stable(self) -> None:
        self.assertEqual(evaluation_seed(7, 2), evaluation_seed(7, 2))
        self.assertNotEqual(evaluation_seed(7, 2), evaluation_seed(7, 3))
        self.assertEqual(scenario_id("b2_1", 1, 7, 2, 4), "b2_1-m1-s7-b2-e4")

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
