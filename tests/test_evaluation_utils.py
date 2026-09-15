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
illegal_targets = evaluation_utils.illegal_targets
resolve_targets = evaluation_utils.resolve_targets
scenario_id = evaluation_utils.scenario_id
summarize_episode_rows = evaluation_utils.summarize_episode_rows
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

    def test_illegal_targets_are_exactly_the_top_layer(self) -> None:
        groups = target_groups(self.cfg)
        self.assertEqual(illegal_targets(groups["legal"], self.cfg), ())
        self.assertEqual(
            illegal_targets(groups["trained"], self.cfg),
            ("b9_1", "b9_2", "b9_3"),
        )
        self.assertEqual(len(illegal_targets(groups["tower"], self.cfg)), 3)

    def test_duplicate_targets_need_an_explicit_opt_in(self) -> None:
        # Evaluation would measure the repeated block twice under one scenario id;
        # training uses repetition to raise a target's sampling share.
        with self.assertRaisesRegex(ValueError, "duplicates"):
            resolve_targets("b2_1,b2_1", self.cfg)
        _, targets = resolve_targets("b2_1,b2_1", self.cfg, allow_duplicates=True)
        self.assertEqual(targets, ("b2_1", "b2_1"))

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

    def test_settling_summary_separates_success_at_the_moment_from_after(self) -> None:
        # Two attempts reach success; one of those towers fails while settling.
        outcomes = (
            {"reached_success": True, "safe_success": True, "damaged_after_success": False},
            {"reached_success": True, "safe_success": False, "damaged_after_success": True},
            {"reached_success": False, "safe_success": False, "damaged_after_success": False},
        )
        rows = [_episode(**outcome, settle_steps=100) for outcome in outcomes]
        summary = summarize_episode_rows(rows)
        self.assertAlmostEqual(summary["reached_success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["damaged_after_success_rate"], 1 / 3)
        self.assertEqual(summary["settle_steps"], 100)

    def test_regular_summary_has_no_settling_columns(self) -> None:
        summary = summarize_episode_rows([_episode(safe_success=True)])
        self.assertNotIn("reached_success_rate", summary)
        self.assertEqual(summary["success_rate"], 1.0)


def _episode(**overrides) -> dict:
    row = {
        "controller": "policy",
        "checkpoint": "model.pt",
        "commit": "abc",
        "target_set": "explicit",
        "target": "b2_1",
        "layer": 2,
        "is_trained": True,
        "is_legal": True,
        "missing_level": 0,
        "safe_success": False,
        "extracted": False,
        "tower_damage": False,
    }
    for key in (
        "progress_final", "progress_max", "steps", "tower_xy_final", "tower_xy_max",
        "tower_xy_recovery", "tower_z_final", "tower_z_max", "tower_z_recovery",
        "tower_rot_deg_final", "tower_rot_deg_max", "tower_rot_deg_recovery",
        "contact_rate", "contact_force_mean", "contact_force_max", "stuck_rate",
        "stop_rate", "retreat_rate",
    ):
        row[key] = 0.0
    row.update(overrides)
    return row


if __name__ == "__main__":
    unittest.main()
