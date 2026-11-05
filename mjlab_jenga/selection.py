from dataclasses import dataclass

def select_random_block(blocks, rng):
    block = rng.choice(blocks)
    return block


@dataclass
class GameState:
    remaining: list[str]
    removed: list[str]
    failed: list[str]
    attempts: list[dict[str, object]]
    tower_state: dict[str, object]
