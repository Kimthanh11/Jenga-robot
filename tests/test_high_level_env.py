import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from mjlab_jenga.high_level_env import (
    HighLevelJengaEnv,
    action_to_block,
    build_action_mask,
    build_block_names_by_action,
)


class HighLevelActionMappingTest(unittest.TestCase):
    def test_valid_action_returns_corresponding_block(self):
        block_names_by_action = ("b1_1", "b1_2", "b1_3")
        action_mask = (True, True, True)

        block_name = action_to_block(1, block_names_by_action, action_mask)

        self.assertEqual(block_name, "b1_2")

    def test_invalid_actions_are_rejected(self):
        block_names_by_action = ("b1_1", "b1_2", "b1_3")
        action_mask = (True, True, True)

        with self.assertRaises(ValueError):
            action_to_block(-1, block_names_by_action, action_mask)
        with self.assertRaises(ValueError):
            action_to_block(len(block_names_by_action), block_names_by_action, action_mask)

    def test_masked_action_is_rejected(self):
        block_names_by_action = ("b1_1", "b1_2", "b1_3")
        action_mask = (True, False, True)

        with self.assertRaises(ValueError):
            action_to_block(1, block_names_by_action, action_mask)

    def test_mask_length_must_match_action_names(self):
        with self.assertRaises(ValueError):
            action_to_block(0, ("b1_1", "b1_2"), (True,))

    def test_action_names_scale_with_tower_configuration(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )

        block_names_by_action = build_block_names_by_action(cfg)

        self.assertEqual(
            block_names_by_action,
            ("b1_1", "b1_2", "b1_3", "b2_1", "b2_2", "b2_3"),
        )

    def test_action_mask_requires_legal_and_remaining(self):
        block_names_by_action = ("b1_1", "b1_2", "b2_1")
        legal_blocks = {"b1_1", "b1_2"}
        remaining_blocks = {"b1_1", "b2_1"}

        action_mask = build_action_mask(
            block_names_by_action,
            legal_blocks,
            remaining_blocks,
        )

        self.assertEqual(action_mask, (True, False, False))

    def test_removed_block_is_masked_and_cannot_be_removed_twice(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        low_level_executor = Mock(return_value=("success", 500))
        env = HighLevelJengaEnv(cfg, low_level_executor, Mock(), Mock())
        action = env.block_names_by_action.index("b1_2")

        self.assertTrue(env.current_action_mask()[action])

        env.mark_block_removed("b1_2")

        self.assertFalse(env.current_action_mask()[action])
        with self.assertRaises(ValueError):
            env.block_name_for_action(action)
        with self.assertRaises(ValueError):
            env.mark_block_removed("b1_2")

    def test_block_attempt_translates_action_and_calls_low_level_executor(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        low_level_executor = Mock(return_value=("success", 500))
        env = HighLevelJengaEnv(cfg, low_level_executor, Mock(), Mock())
        action = env.block_names_by_action.index("b1_2")

        result = env.run_block_attempt(action)

        self.assertEqual(result, ("b1_2", "success", 500))
        low_level_executor.assert_called_once_with("b1_2")

    def test_success_continues_when_another_legal_block_remains(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        env = HighLevelJengaEnv(cfg, Mock(), Mock(), Mock())

        done = env.update_after_attempt("b1_1", "success")

        self.assertFalse(done)
        self.assertNotIn("b1_1", env.remaining_blocks)

    def test_success_ends_episode_after_last_legal_block(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        env = HighLevelJengaEnv(cfg, Mock(), Mock(), Mock())
        env.mark_block_removed("b1_1")
        env.mark_block_removed("b1_2")

        done = env.update_after_attempt("b1_3", "success")

        self.assertTrue(done)

    def test_tower_damage_ends_episode_without_removing_block(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        env = HighLevelJengaEnv(cfg, Mock(), Mock(), Mock())

        done = env.update_after_attempt("b1_1", "tower_damage")

        self.assertTrue(done)
        self.assertIn("b1_1", env.remaining_blocks)

    def test_nonterminal_failure_keeps_block_available(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        env = HighLevelJengaEnv(cfg, Mock(), Mock(), Mock())

        done = env.update_after_attempt("b1_1", "step_cap")

        self.assertFalse(done)
        self.assertIn("b1_1", env.remaining_blocks)

    def test_step_runs_consecutive_attempts_without_reset(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        low_level_executor = Mock(
            side_effect=[("success", 100), ("tower_damage", 50)]
        )
        observation_provider = Mock(
            side_effect=["tower_after_success", "damaged_tower"]
        )
        env = HighLevelJengaEnv(
            cfg,
            low_level_executor,
            observation_provider,
            Mock(),
        )
        first_action = env.block_names_by_action.index("b1_1")
        second_action = env.block_names_by_action.index("b1_2")

        first_observation, first_reward, first_done, first_info = env.step(
            first_action
        )
        second_observation, second_reward, second_done, second_info = env.step(
            second_action
        )

        self.assertEqual(first_observation, "tower_after_success")
        self.assertEqual(first_reward, 1.0)
        self.assertFalse(first_done)
        self.assertFalse(first_info["action_mask"][first_action])
        self.assertNotIn("b1_1", env.remaining_blocks)

        self.assertEqual(second_observation, "damaged_tower")
        self.assertEqual(second_reward, -2.0)
        self.assertTrue(second_done)
        self.assertEqual(second_info["reason"], "tower_damage")
        low_level_executor.assert_has_calls([call("b1_1"), call("b1_2")])

    def test_reset_restores_block_availability_and_returns_new_observation(self):
        cfg = SimpleNamespace(
            LAYERS=2,
            BLOCKS_PER_LAYER=3,
            RANDOM_TARGET_BLOCK_NAMES=(),
        )
        environment_resetter = Mock()
        observation_provider = Mock(return_value="fresh_tower")
        env = HighLevelJengaEnv(
            cfg,
            Mock(),
            observation_provider,
            environment_resetter,
        )
        env.mark_block_removed("b1_1")

        observation, info = env.reset()

        self.assertEqual(observation, "fresh_tower")
        self.assertIn("b1_1", env.remaining_blocks)
        self.assertTrue(info["action_mask"][0])
        environment_resetter.assert_called_once_with()
        observation_provider.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
