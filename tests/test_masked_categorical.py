import unittest

import torch
from tensordict import TensorDict

from mjlab_jenga.masked_categorical import (
    CategoricalDistribution,
    MaskedCategoricalModel,
)


class CategoricalDistributionTest(unittest.TestCase):
    def test_sample_and_log_prob_use_single_action_column(self):
        distribution = CategoricalDistribution(output_dim=1, num_categories=3)
        distribution.update(torch.tensor([[0.0, 1.0, 2.0]]))

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        self.assertEqual(tuple(action.shape), (1, 1))
        self.assertEqual(tuple(log_prob.shape), (1,))

    def test_deterministic_output_uses_largest_logit(self):
        distribution = CategoricalDistribution(output_dim=1, num_categories=3)
        output = distribution.deterministic_output(torch.tensor([[1.0, 5.0, 2.0]]))
        self.assertEqual(output.item(), 1)


class MaskedCategoricalModelTest(unittest.TestCase):
    def _model(self):
        obs = TensorDict(
            {
                "tower_state": torch.zeros(1, 4),
                "action_mask": torch.ones(1, 3, dtype=torch.bool),
            },
            batch_size=[1],
        )
        model = MaskedCategoricalModel(
            obs=obs,
            obs_groups={"actor": ["tower_state"]},
            obs_set="actor",
            output_dim=1,
            hidden_dims=(8,),
            distribution_cfg={
                "class_name": "mjlab_jenga.masked_categorical:CategoricalDistribution",
                "num_categories": 3,
            },
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return model

    def test_masked_action_is_never_sampled(self):
        model = self._model()
        obs = TensorDict(
            {
                "tower_state": torch.zeros(64, 4),
                "action_mask": torch.tensor([[False, True, False]]).repeat(64, 1),
            },
            batch_size=[64],
        )

        actions = model(obs, stochastic_output=True)

        self.assertTrue(torch.equal(actions, torch.ones_like(actions)))

    def test_all_invalid_mask_is_rejected(self):
        model = self._model()
        obs = TensorDict(
            {
                "tower_state": torch.zeros(1, 4),
                "action_mask": torch.zeros(1, 3, dtype=torch.bool),
            },
            batch_size=[1],
        )

        with self.assertRaisesRegex(ValueError, "at least one valid action"):
            model(obs, stochastic_output=True)


if __name__ == "__main__":
    unittest.main()
