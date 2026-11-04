import unittest
from unittest.mock import Mock

import torch

from mjlab_jenga.high_level_vecenv import HighLevelRslRlVecEnv


class HighLevelRslRlVecEnvTest(unittest.TestCase):
    def _high_level_env(self):
        env = Mock()
        env.legal_blocks = {"b1_1", "b1_2"}
        env.environment_resetter = Mock()
        env.current_action_mask.return_value = (True, False, True)
        env.reset.return_value = (torch.zeros(1, 6), {"action_mask": (True, False, True)})
        return env

    def test_observations_include_tower_and_mask(self):
        env = self._high_level_env()
        adapter = HighLevelRslRlVecEnv(env, "cpu")

        observations = adapter.get_observations()

        self.assertEqual(tuple(observations["tower_state"].shape), (1, 6))
        self.assertEqual(tuple(observations["action_mask"].shape), (1, 3))
        self.assertEqual(adapter.num_actions, 1)

    def test_terminal_step_auto_resets_for_runner(self):
        env = self._high_level_env()
        env.step.return_value = (
            torch.ones(1, 6),
            -2.0,
            True,
            {"reason": "tower_damage"},
        )
        adapter = HighLevelRslRlVecEnv(env, "cpu")

        _, rewards, dones, extras = adapter.step(torch.tensor([[1]]))

        env.step.assert_called_once_with(1)
        self.assertEqual(rewards.item(), -2.0)
        self.assertEqual(dones.item(), 1)
        self.assertIn("terminal_high_level", extras)
        self.assertEqual(env.reset.call_count, 2)


if __name__ == "__main__":
    unittest.main()
