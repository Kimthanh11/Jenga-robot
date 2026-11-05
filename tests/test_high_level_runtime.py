import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mjlab_jenga.high_level_runtime import SerialHighLevelRuntime


class SerialHighLevelRuntimeTest(unittest.TestCase):
    @patch("mjlab_jenga.high_level_runtime.get_target_command")
    def test_observation_flattens_blocks_and_features(self, get_command_mock):
        command = Mock()
        command.absolute_tower_state_tensor.return_value = torch.zeros(1, 27, 14)
        get_command_mock.return_value = command
        runtime = SerialHighLevelRuntime(SimpleNamespace(num_envs=1))

        observation = runtime.observation()

        self.assertEqual(tuple(observation.shape), (1, 27 * 14))

    @patch("mjlab_jenga.high_level_runtime.get_target_command")
    def test_reset_resets_simulator_and_reference(self, get_command_mock):
        command = Mock()
        get_command_mock.return_value = command
        env = SimpleNamespace(num_envs=1, reset=Mock())
        runtime = SerialHighLevelRuntime(env)

        runtime.reset()

        env.reset.assert_called_once_with()
        command.restore_pristine_reference.assert_called_once_with()

    @patch("mjlab_jenga.high_level_runtime.get_target_command")
    def test_parallel_environment_is_rejected(self, get_command_mock):
        with self.assertRaisesRegex(ValueError, "num_envs=1"):
            SerialHighLevelRuntime(SimpleNamespace(num_envs=4))
        get_command_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
