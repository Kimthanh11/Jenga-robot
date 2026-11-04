from collections.abc import Callable

from mjlab_jenga.evaluation_utils import target_groups

LowLevelExecutor = Callable[[str], tuple[str, int]]
ObservationProvider = Callable[[], object]
EnvironmentResetter = Callable[[], None]


def action_to_block(
    action: int,
    block_names_by_action: tuple[str, ...],
    action_mask: tuple[bool, ...],
) -> str:
    """Translate a valid high-level action index into its block name."""
    if len(action_mask) != len(block_names_by_action):
        raise ValueError("Action mask and block names must have the same length")

    if action >= len(block_names_by_action) or action < 0:
        raise ValueError("Index not valid in blocklist")

    if not action_mask[action]:
        raise ValueError("Selected action is currently not allowed")

    block_name = block_names_by_action[action]
    return block_name


def build_block_names_by_action(cfg) -> tuple[str, ...]:
    """Return stable action-to-block identities for the configured tower."""
    tower = target_groups(cfg)["tower"]
    return tower


def build_action_mask(
    block_names_by_action: tuple[str, ...],
    legal_blocks: set[str],
    remaining_blocks: set[str],
) -> tuple[bool, ...]:
    """Return one selectability flag per stable high-level action."""

    eligible_blocks = legal_blocks & remaining_blocks
    action_mask = tuple(name in eligible_blocks for name in block_names_by_action)
    return action_mask


class HighLevelJengaEnv:
    def __init__(
        self,
        cfg,
        low_level_executor: LowLevelExecutor,
        observation_provider: ObservationProvider,
        environment_resetter: EnvironmentResetter,
    ) -> None:
        self.block_names_by_action = build_block_names_by_action(cfg)

        groups = target_groups(cfg)
        self.legal_blocks = set(groups["legal"])
        self.remaining_blocks = set(self.block_names_by_action)
        self.low_level_executor = low_level_executor
        self.observation_provider = observation_provider
        self.environment_resetter = environment_resetter

    def current_action_mask(self) -> tuple[bool, ...]:
        """Return the mask aligned with ``block_names_by_action``."""
        return build_action_mask(
            self.block_names_by_action,
            self.legal_blocks,
            self.remaining_blocks,
        )

    def block_name_for_action(self, action: int) -> str:
        """Translate one valid integer action into its stable block identity."""
        action_mask = self.current_action_mask()
        block_name = action_to_block(
            action=action,
            block_names_by_action=self.block_names_by_action,
            action_mask=action_mask,
        )
        return block_name

    def mark_block_removed(self, block_name: str) -> None:
        """Remove one successfully extracted block from future actions."""
        if block_name in self.remaining_blocks:
            self.remaining_blocks.remove(block_name)
        else:
            raise ValueError("The block is not in the remaining block names anymore.")

    def reset_remaining_blocks(self) -> None:
        """Restore availability at the beginning of a new game."""
        self.remaining_blocks = set(self.block_names_by_action)

    def run_block_attempt(self, action: int) -> tuple[str, str, int]:
        """Execute one complete low-level attempt for the selected action."""
        block_name = self.block_name_for_action(action)
        reason, steps = self.low_level_executor(block_name)

        return block_name, reason, steps

    def update_after_attempt(self, block_name: str, reason: str) -> bool:
        """Update block availability and return whether the episode ended."""
        if reason == "success":
            self.mark_block_removed(block_name)
            return not any(self.current_action_mask())
        if reason in ("step_cap", "settle_failed"):
            return False
        if reason == "tower_damage":
            return True
        raise ValueError(f"Unknown attempt reason: {reason}")

    def reward_for_attempt(self, reason: str) -> float:
        """Map a low-level terminal reason to the pilot high-level reward."""
        if reason == "success":
            return 1.0
        if reason in ("step_cap", "settle_failed"):
            return -0.2
        if reason == "tower_damage":
            return -2.0
        raise ValueError(f"Unknown attempt reason: {reason}")

    def current_observation(self) -> object:
        """Return the current representation supplied by the active encoder."""
        return self.observation_provider()

    def step(self, action: int) -> tuple[object, float, bool, dict[str, object]]:
        """Execute one high-level action, which is one complete block attempt."""
        block_name, reason, steps = self.run_block_attempt(action)
        reward = self.reward_for_attempt(reason)
        done = self.update_after_attempt(block_name, reason)
        observation = self.current_observation()

        info = {
            "block_name": block_name,
            "reason": reason,
            "steps": steps,
            "action_mask": self.current_action_mask(),
        }

        return observation, reward, done, info

    def reset(self) -> tuple[object, dict[str, object]]:
        """Reset the physical game and the high-level availability state."""
        self.environment_resetter()
        self.reset_remaining_blocks()
        observation = self.current_observation()

        info = {
            "action_mask": self.current_action_mask(),
        }

        return observation, info
