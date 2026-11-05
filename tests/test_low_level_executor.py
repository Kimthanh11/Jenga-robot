import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mjlab_jenga.low_level_executor import LearnedLowLevelExecutor, action_column


class LowLevelExecutorTest(unittest.TestCase):
    def test_action_column_sums_dimensions_before_named_term(self):
        env = SimpleNamespace(
            action_manager=SimpleNamespace(
                active_terms=["touch", "yaw", "push_stop_retreat"],
                action_term_dim=[2, 1, 1],
            )
        )

        self.assertEqual(action_column(env, "push_stop_retreat"), 3)

    def test_unknown_action_term_is_rejected(self):
        env = SimpleNamespace(
            action_manager=SimpleNamespace(
                active_terms=["touch"],
                action_term_dim=[2],
            )
        )

        with self.assertRaisesRegex(ValueError, "Unknown action term"):
            action_column(env, "missing")

    @patch("mjlab_jenga.low_level_executor.get_target_command")
    @patch("mjlab_jenga.low_level_executor.remove_block_from_tower")
    @patch("mjlab_jenga.low_level_executor.extract_block")
    def test_success_is_parked_and_rebaselined(
        self,
        extract_block_mock,
        remove_block_mock,
        get_target_command_mock,
    ):
        extract_block_mock.return_value = ("b1_1", "success", 123)
        command = Mock()
        get_target_command_mock.return_value = command
        env = SimpleNamespace(num_envs=1)
        executor = LearnedLowLevelExecutor(env, Mock(), Mock())

        result = executor("b1_1")

        self.assertEqual(result, ("success", 123))
        remove_block_mock.assert_called_once_with(env, "b1_1")
        command.rebaseline.assert_called_once_with()

    @patch("mjlab_jenga.low_level_executor.get_target_command")
    @patch("mjlab_jenga.low_level_executor.remove_block_from_tower")
    @patch("mjlab_jenga.low_level_executor.extract_block")
    def test_failed_attempt_keeps_physical_tower_unchanged(
        self,
        extract_block_mock,
        remove_block_mock,
        get_target_command_mock,
    ):
        extract_block_mock.return_value = ("b1_1", "step_cap", 2000)
        env = SimpleNamespace(num_envs=1)
        executor = LearnedLowLevelExecutor(env, Mock(), Mock())

        result = executor("b1_1")

        self.assertEqual(result, ("step_cap", 2000))
        remove_block_mock.assert_not_called()
        get_target_command_mock.assert_not_called()

    def test_parallel_environment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_envs=1"):
            LearnedLowLevelExecutor(SimpleNamespace(num_envs=2), Mock(), Mock())


if __name__ == "__main__":
    unittest.main()
