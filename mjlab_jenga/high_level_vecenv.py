"""RSL-RL VecEnv adapter for the serial high-level Jenga macro environment."""

import torch
from tensordict import TensorDict

from mjlab_jenga.high_level_env import HighLevelJengaEnv


class HighLevelRslRlVecEnv:
    """Expose one serial high-level game through RSL-RL's VecEnv protocol."""

    def __init__(self, env: HighLevelJengaEnv, device: str | torch.device) -> None:
        self.high_level_env = env
        self.device = torch.device(device)
        self.num_envs = 1
        self.num_actions = 1
        self.max_episode_length = len(env.legal_blocks)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.cfg = getattr(
            getattr(env.environment_resetter, "__self__", None),
            "env",
            None,
        )
        if self.cfg is not None:
            self.cfg = self.cfg.cfg

        observation, _ = self.high_level_env.reset()
        self._observation = self._as_observation_tensor(observation)

    def _as_observation_tensor(self, observation: object) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("The PPO runtime observation must be a torch.Tensor")
        if observation.ndim != 2 or observation.shape[0] != 1:
            raise ValueError(
                "Expected observation shape (1, features), got "
                f"{tuple(observation.shape)}"
            )
        return observation.to(self.device)

    def get_observations(self) -> TensorDict:
        action_mask = torch.tensor(
            self.high_level_env.current_action_mask(),
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        return TensorDict(
            {
                "tower_state": self._observation,
                "action_mask": action_mask,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        action = int(actions.reshape(-1)[0].item())
        observation, reward, done, info = self.high_level_env.step(action)
        self.episode_length_buf += 1

        extras = {"high_level": info}
        if done:
            observation, reset_info = self.high_level_env.reset()
            self.episode_length_buf.zero_()
            extras["terminal_high_level"] = info
            extras["reset_high_level"] = reset_info

        self._observation = self._as_observation_tensor(observation)
        rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
        dones = torch.tensor([done], dtype=torch.long, device=self.device)
        return self.get_observations(), rewards, dones, extras

    def reset(self) -> TensorDict:
        observation, _ = self.high_level_env.reset()
        self._observation = self._as_observation_tensor(observation)
        self.episode_length_buf.zero_()
        return self.get_observations()

    def seed(self, seed: int) -> None:
        physical_env = getattr(
            getattr(self.high_level_env.environment_resetter, "__self__", None),
            "env",
            None,
        )
        if physical_env is not None:
            physical_env.seed(seed)
