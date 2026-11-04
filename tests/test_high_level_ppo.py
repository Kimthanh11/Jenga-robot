import unittest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from mjlab_jenga.high_level_ppo_cfg import high_level_ppo_train_dict


class _FakeHighLevelVecEnv:
    num_envs = 1
    num_actions = 1


class HighLevelPpoConstructionTest(unittest.TestCase):
    def _observations(self) -> TensorDict:
        return TensorDict(
            {
                "tower_state": torch.zeros(1, 27 * 14),
                "action_mask": torch.tensor(
                    [[False, False, True] + [False] * 24], dtype=torch.bool
                ),
            },
            batch_size=[1],
        )

    def test_rsl_ppo_constructs_and_samples_only_valid_action(self):
        observations = self._observations()
        cfg = high_level_ppo_train_dict(num_block_actions=27)

        algorithm = PPO.construct_algorithm(
            observations,
            _FakeHighLevelVecEnv(),
            cfg,
            device="cpu",
        )
        action = algorithm.act(observations)

        self.assertEqual(tuple(action.shape), (1, 1))
        self.assertEqual(action.item(), 2)

    def test_one_complete_rsl_ppo_update_runs(self):
        observations = self._observations()
        cfg = high_level_ppo_train_dict(num_block_actions=27)
        rollout_steps = cfg["num_steps_per_env"]
        algorithm = PPO.construct_algorithm(
            observations,
            _FakeHighLevelVecEnv(),
            cfg,
            device="cpu",
        )

        for _ in range(rollout_steps):
            algorithm.act(observations)
            algorithm.process_env_step(
                observations,
                rewards=torch.ones(1),
                dones=torch.zeros(1, dtype=torch.long),
                extras={},
            )
        algorithm.compute_returns(observations)

        losses = algorithm.update()

        self.assertIn("surrogate", losses)
        self.assertIn("value", losses)


if __name__ == "__main__":
    unittest.main()
