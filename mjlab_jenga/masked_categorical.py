"""Masked categorical actor components compatible with the installed RSL-RL PPO."""

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Categorical

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import unpad_trajectories


class CategoricalDistribution(Distribution):
    """Represent one discrete action stored as a one-column tensor."""

    def __init__(self, output_dim: int, num_categories: int) -> None:
        if output_dim != 1:
            raise ValueError("CategoricalDistribution expects one scalar action")
        if num_categories <= 1:
            raise ValueError("num_categories must be greater than one")
        super().__init__(output_dim)
        self.num_categories = num_categories
        self._distribution: Categorical | None = None

    def update(self, mlp_output: torch.Tensor) -> None:
        self._distribution = Categorical(logits=mlp_output)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample().unsqueeze(-1)  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output.argmax(dim=-1, keepdim=True)

    def as_deterministic_output_module(self) -> nn.Module:
        return _CategoricalArgmax()

    @property
    def input_dim(self) -> int:
        return self.num_categories

    @property
    def mean(self) -> torch.Tensor:
        indices = torch.arange(
            self.num_categories,
            device=self._distribution.probs.device,  # type: ignore[union-attr]
            dtype=self._distribution.probs.dtype,  # type: ignore[union-attr]
        )
        return (self._distribution.probs * indices).sum(dim=-1, keepdim=True)  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.variance.sqrt().unsqueeze(-1)  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._distribution.logits,)  # type: ignore[union-attr]

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs.squeeze(-1).long())  # type: ignore[union-attr]

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_distribution = Categorical(logits=old_params[0])
        new_distribution = Categorical(logits=new_params[0])
        return torch.distributions.kl_divergence(old_distribution, new_distribution)


class MaskedCategoricalModel(MLPModel):
    """MLP actor that sets logits of currently illegal blocks to negative infinity."""

    def __init__(self, *args, action_mask_key: str = "action_mask", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_mask_key = action_mask_key

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if masks is not None and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)

        latent = self.get_latent(obs, masks=None, hidden_state=hidden_state)
        logits = self.mlp(latent)
        action_mask = obs[self.action_mask_key].bool()
        if action_mask.shape != logits.shape:
            raise ValueError(
                "Action-mask shape must match categorical logits: "
                f"{tuple(action_mask.shape)} != {tuple(logits.shape)}"
            )
        if not action_mask.any(dim=-1).all():
            raise ValueError("Every environment must have at least one valid action")

        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        if self.distribution is None:
            raise RuntimeError("MaskedCategoricalModel requires a distribution_cfg")
        if stochastic_output:
            self.distribution.update(masked_logits)
            return self.distribution.sample()
        return self.distribution.deterministic_output(masked_logits)


class _CategoricalArgmax(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1, keepdim=True)
