import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import mjlab_jenga.jenga_mjenv_cfg as jenga_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab_jenga.evaluation_utils import target_groups
from mjlab_jenga.jenga_mjenv_cfg import (
    jenga_env_cfg,
    jenga_ppo_runner_cfg,
    remove_block_from_tower,
    tower_state_observation,
)
from mjlab_jenga.low_level_executor import extract_block, get_target_command
from mjlab_jenga.selection import GameState, select_random_block


def _block_physics_snapshot(
    env: ManagerBasedRlEnv,
    block_names: list[str],
) -> dict[str, dict[str, object]]:
    """
    Captures privileged information: Mass and friction

    """
    result = {} #saves friciton and mass
    for name in block_names:
        asset = env.scene[name] #gets the blocks out of the scene

        #get internal IDs:
        #geom_ids for Geoemtrie, hence for friciton; body_ids for the mass
        geom_ids = asset.data.indexing.geom_ids
        body_ids = asset.data.indexing.body_ids

        #read the actual values out of the simulation
        result[name] = {
            "friction": env.sim.model.geom_friction[0, geom_ids]
            .detach()
            .cpu()
            .tolist(),
            "mass": env.sim.model.body_mass[0, body_ids].detach().cpu().tolist(),
        }
    return result


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Persist completed games without exposing a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def gameloop(
    blocklist,
    env: ManagerBasedRlEnv,
    policy,
    wrapped_env: RslRlVecEnvWrapper,
    rng,
    selection_strategy,
    settle_steps: int,
    max_steps_per_block: int,
    max_attempts_per_block: int,
    max_moves_per_game: int | None,
):
    target_command = get_target_command(env)
    target_command.restore_pristine_reference()
    blocklist_copy = blocklist.copy()
    gamestate = GameState(
        remaining=blocklist_copy,
        removed=[],
        failed=[],
        attempts=[],
        tower_state=target_command.absolute_tower_state(),
    )
    attempt_counts = {block: 0 for block in blocklist_copy}

    while True:
        if max_moves_per_game is not None and len(gamestate.attempts) >= max_moves_per_game:
            break
        eligible_blocks = [
            block
            for block in gamestate.remaining
            if attempt_counts[block] < max_attempts_per_block
        ]
        if not eligible_blocks:
            break

        state_before = target_command.absolute_tower_state()
        gamestate.tower_state = state_before

        # Select the strategy to pick a block, e.g. random or learned.
        target_block = selection_strategy(eligible_blocks, rng)

        # Run one complete extraction attempt for the selected block.
        target_block, reason, steps = extract_block(
            env=env,
            target_block=target_block,
            wrapped_env=wrapped_env,
            policy=policy,
            settle_steps=settle_steps,
            max_steps_per_block=max_steps_per_block,
        )
        attempt_counts[target_block] += 1 # counts the attempts per block

        # we had a small settling period after pushing, to account for swinging
        state_after_settle = target_command.absolute_tower_state()
        low_level_state_after_settle = (
            tower_state_observation(env)[0].detach().cpu().tolist()
        )
        if reason == "success":
            remove_block_from_tower(env, target_block)
            target_command.rebaseline() # rebase to account for shiftings as the new starting pos
            gamestate.remaining.remove(target_block)
            gamestate.removed.append(target_block)
        elif reason == "tower_damage":
            pass
        elif reason in ("step_cap", "settle_failed"):
            if attempt_counts[target_block] >= max_attempts_per_block:
                gamestate.failed.append(target_block)
        else:
            raise RuntimeError(f"Unknown extraction result: {reason}")

        state_next = target_command.absolute_tower_state()
        low_level_state_next = tower_state_observation(env)[0].detach().cpu().tolist()
        gamestate.tower_state = state_next
        gamestate.attempts.append(
            {
                "attempt": len(gamestate.attempts) + 1,
                "eligible_blocks": list(eligible_blocks),
                "state_before": state_before,
                "block": target_block,
                "reason": reason,
                "steps": steps,
                "state_after_settle": state_after_settle,
                "low_level_tower_state_after_settle": low_level_state_after_settle,
                "state_next": state_next,
                "low_level_tower_state_next": low_level_state_next,
            }
        )
        if reason == "tower_damage":
            break

    return gamestate


def main(
    device: str,
    checkpoint: str,
    settle_steps: int,
    seed: int,
    max_steps_per_block: int,
    max_attempts_per_block: int,
    max_moves_per_game: int | None,
    games: int,
    output: Path,
):
    jenga_cfg.apply_low_level_stage("target")
    # Match the final training task represented by the selected checkpoint.
    jenga_cfg.SUCCESS_CURRICULUM_START = jenga_cfg.SUCCESS_CURRICULUM_END
    jenga_cfg.YAW_CURRICULUM_START = 0.0
    jenga_cfg.YAW_CURRICULUM_END = 0.0
    # A baseline game starts with a complete tower. Missing blocks are introduced only
    # by successful extractions during this game.
    jenga_cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
    jenga_cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
    env_cfg = jenga_env_cfg(play=True)
    env_cfg.auto_reset = False
    env_cfg.scene.num_envs = 1
    # Evaluate these conditions directly to keep playing after an extraction.
    env_cfg.terminations.pop("success", None)
    env_cfg.terminations.pop("tower_damage", None)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.seed(seed)
    agent_cfg = jenga_ppo_runner_cfg()
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
        wrapped_env,
        asdict(agent_cfg),
        device=device,
    )
    runner.load(path=checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    legal_blocks = list(target_groups(jenga_cfg)["legal"])
    command = get_target_command(env)
    game_records = []
    payload = {
        "schema_version": 1,
        "seed": seed,
        "checkpoint": checkpoint,
        "settle_steps": settle_steps,
        "max_steps_per_block": max_steps_per_block,
        "max_attempts_per_block": max_attempts_per_block,
        "max_moves_per_game": max_moves_per_game,
        "games": games,
        "legal_blocks": legal_blocks,
        "game_records": game_records,
    }

    try:
        for game_index in range(games):
            game_seed = seed + game_index
            env.reset(seed=game_seed)
            command.restore_pristine_reference()
            initial_state = command.absolute_tower_state()
            block_names = initial_state["block_names"]
            assert isinstance(block_names, list)
            physics = _block_physics_snapshot(env, block_names)
            rng = random.Random(game_seed)
            gamestate = gameloop(
                blocklist=legal_blocks,
                env=env,
                policy=policy,
                wrapped_env=wrapped_env,
                rng=rng,
                selection_strategy=select_random_block,
                settle_steps=settle_steps,
                max_steps_per_block=max_steps_per_block,
                max_attempts_per_block=max_attempts_per_block,
                max_moves_per_game=max_moves_per_game,
            )
            game_records.append(
                {
                    "game": game_index + 1,
                    "seed": game_seed,
                    "initial_state": initial_state,
                    "privileged_block_physics": physics,
                    "result": asdict(gamestate),
                }
            )
            _write_json_atomic(output, payload)

            sliding = [values["friction"][0][0] for values in physics.values()]
            print(
                f"Game {game_index + 1}: sampled sliding friction range "
                f"{min(sliding):.5f} .. {max(sliding):.5f}; "
                f"extracted={len(gamestate.removed)}; attempts={len(gamestate.attempts)}",
                flush=True,
            )
    finally:
        env.close()

    print(f"Wrote evaluation data: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate random Jenga block selection.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps-per-block", type=int, default=2000)
    parser.add_argument("--max-attempts-per-block", type=int, default=1)
    parser.add_argument(
        "--max-moves-per-game",
        type=int,
        default=0,
        help="Stop after this many attempts; zero means no game-level cap.",
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=100,
        help="Control steps after extraction with the push stopped (100 is about 1 s).",
    )
    args = parser.parse_args()
    if args.settle_steps < 0:
        parser.error("--settle-steps must not be negative")
    if args.max_steps_per_block <= 0:
        parser.error("--max-steps-per-block must be positive")
    if args.max_attempts_per_block <= 0:
        parser.error("--max-attempts-per-block must be positive")
    if args.max_moves_per_game < 0:
        parser.error("--max-moves-per-game must not be negative")
    if args.games <= 0:
        parser.error("--games must be positive")
    main(
        device=args.device,
        checkpoint=args.checkpoint,
        settle_steps=args.settle_steps,
        seed=args.seed,
        max_steps_per_block=args.max_steps_per_block,
        max_attempts_per_block=args.max_attempts_per_block,
        max_moves_per_game=(
            None if args.max_moves_per_game == 0 else args.max_moves_per_game
        ),
        games=args.games,
        output=args.output,
    )
