"""Play a sequence of extractions on one tower, to see how far the skill composes.

Standalone experiment. Imports jenga_mjenv_cfg and drives it from outside; nothing in
the training or evaluation path is touched, and deleting this file changes nothing.

Every measurement so far evaluates a single extraction from a tower that is reset
beforehand, with at most three blocks pre-removed from a fixed set of patterns. A game
is different: the tower state after five moves follows from the moves themselves, and
is almost never one of those patterns. This script asks the question that per-target
success rates cannot answer -- how many blocks come out in a row before the tower
falls.

Procedure per game: reset once, then repeatedly pick the most promising remaining
target, run the policy until the attempt terminates, and on success remove the block
from the tower and continue. A game ends when the tower is damaged, when every
remaining target has been tried without success, or at the extraction cap.

Two modelling choices worth stating, both switchable:

* Damage baseline. tower_damage compares against `_start_pos`, the nominal spawn pose,
  and its 25 mm threshold was calibrated for one extraction. Across ten moves the tower
  drifts past that under any policy, so by default the baseline is re-established after
  each successful extraction and every attempt is judged on its own. --cumulative keeps
  the original baseline instead, which is stricter and arguably closer to a real game.
* Target order. Fixed, from the measured per-target difficulty rather than learned:
  the top layer first, the loaded centre block of layer 2 last. Block selection is not
  part of this project, so a defensible heuristic stands in for it.

One environment per game, because `_start_pos` is shared across environments and a
per-game baseline would otherwise leak between them.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Measured single-extraction difficulty, easiest first: the top layer carries no load,
# then the outer blocks of the lower layers, and b2_2 last -- confined on both sides
# with seven layers above it, and the only target that ever damages the tower.
DEFAULT_ORDER = ("b9_1", "b9_3", "b9_2", "b2_1", "b2_3", "b3_1", "b2_2")


def _maybe_force_cpu() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        import torch

        if torch.cuda.is_available():
            return
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract blocks one after another from a single tower."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--max-extractions",
        type=int,
        default=12,
        help="Cap per game. Seven targets are viable, so a game normally ends before "
        "this; the cap only bounds the runtime if something goes wrong.",
    )
    parser.add_argument(
        "--attempt-steps",
        type=int,
        default=1200,
        help="Step budget per attempt. Beyond roughly 900 steps an attempt has "
        "stalled; the longest successful extraction measured takes 910.",
    )
    parser.add_argument(
        "--order",
        default=",".join(DEFAULT_ORDER),
        help="Target order, easiest first.",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Keep the original damage baseline for the whole game instead of "
        "re-establishing it after each extraction.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Allow a target that timed out to be attempted again later, once other "
        "blocks have been removed and the load around it has changed.",
    )
    parser.add_argument("--abort", action="store_true", help="Checkpoint of the abort variant.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _maybe_force_cpu()

    import torch
    from dataclasses import asdict

    import mjlab_jenga.jenga_mjenv_cfg as cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    task = cfg
    if args.abort:
        import mjlab_jenga.jenga_abort_cfg as abort_cfg

        task = abort_cfg

    order = [t.strip() for t in args.order.split(",") if t.strip()]

    cfg.apply_low_level_stage("target")
    cfg.YAW_CURRICULUM_START = 0.0
    cfg.YAW_CURRICULUM_END = 0.0
    # Resets must leave the tower intact; this script removes blocks itself.
    cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
    # Any block has to be markable as removed, not only the eight that appear in the
    # hand-written patterns. Marking matters beyond bookkeeping: the damage criterion
    # masks by presence, so a parked block that is not marked absent registers as a
    # 1.5 m displacement and ends the game immediately.
    #
    # Set before the environment is built -- the mask is sized from this and the park
    # offsets are indexed by position in it.
    #
    # One block stays out. TargetBlockCommand builds its selectable set by excluding
    # everything listed here and refuses to start if nothing is left, on the reasonable
    # assumption that a block cannot be both a training target and removable. This
    # script never samples a target, it forces every one, so the selectable set is only
    # needed to get past that check. b1_2 is a layer-1 centre block: never a target,
    # and never removed here.
    placeholder = "b1_2"
    cfg.MISSING_BLOCK_CANDIDATES = tuple(
        name
        for layer in range(1, cfg.LAYERS + 1)
        for slot in range(1, cfg.BLOCKS_PER_LAYER + 1)
        if (name := f"b{layer}_{slot}") != placeholder
    )
    if placeholder in order:
        raise SystemExit(f"{placeholder} is reserved as the selectable placeholder")

    env_cfg = task.jenga_env_cfg()
    # selectable_target_names is a dataclass field default, bound to
    # RANDOM_TARGET_BLOCK_NAMES when the module is imported, so it has to be set on the
    # instance rather than through the module global.
    env_cfg.commands["target_block"].selectable_target_names = (placeholder,)
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False
    env_cfg.observations["actor"].enable_corruption = False
    # Attempts are bounded by --attempt-steps, not by the episode length, which would
    # otherwise fire a time_out termination in the middle of a game.
    env_cfg.episode_length_s = 1.0e10
    # The success and damage terminations are removed and evaluated here instead. With
    # auto_reset off, a fired termination makes the environment demand a reset before
    # the next step, and that reset would rebuild the tower -- the one thing a game
    # must not do between moves. The same two functions are called directly below, so
    # nothing is lost but the environment's own reaction to them.
    env_cfg.terminations.pop("success", None)
    env_cfg.terminations.pop("tower_damage", None)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    agent_cfg = task.jenga_ppo_runner_cfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        args.checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    cmd = env.command_manager.get_term("target_block")
    names = list(cmd._all_names)
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    park_base = torch.tensor(
        cfg.MISSING_BLOCK_PARK_OFFSET, device=env.device, dtype=torch.float32
    )

    def force_target(name: str) -> None:
        """Select a target and send the hook to its home pose, without a tower reset."""
        idx = names.index(name)
        cmd._force_target_per_env = torch.full(
            (env.num_envs,), idx, dtype=torch.long, device=env.device
        )
        cmd._resample_command(env_ids)
        actual = names[int(cmd.selected_block_idx[0].item())]
        if actual != name:
            raise RuntimeError(f"target not applied: asked {name}, got {actual}")

    def remove_block(name: str) -> None:
        """Park the block far away and mark it absent for the observation terms."""
        candidate_idx = cfg.MISSING_BLOCK_CANDIDATES.index(name)
        mask = cfg._ensure_missing_block_state(env)
        mask[env_ids, candidate_idx] = True
        asset = env.scene[name]
        root_state = asset.data.default_root_state[env_ids].clone()
        offset = park_base.clone()
        offset[0] += cfg.MISSING_BLOCK_PARK_SPACING * candidate_idx
        root_state[:, :3] += offset
        root_state[:, 7:] = 0.0
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)

    # _start_pos is written once in the command term's constructor and is not
    # restored by env.reset(), so a game that moves the baseline leaves it moved for
    # every game after it. Keep the original to put back at the start of each game.
    pristine_start_pos = cmd._start_pos.clone()

    def rebaseline() -> None:
        """Make the current tower the reference for progress and damage."""
        current = torch.stack(
            [b.data.body_link_pos_w[:, 0, :] for b in cmd._blocks], dim=0
        )
        cmd._start_pos[:] = current[:, 0, :]

    rows: list[dict] = []
    for game in range(1, args.games + 1):
        env.reset()
        # Restore the nominal spawn reference; the reset rebuilds the tower but not
        # this. Without it, every game after the first measures a fresh tower against
        # the dismantled one left by its predecessor and is damaged on the first move.
        cmd._start_pos[:] = pristine_start_pos
        removed: list[str] = []
        failed: list[str] = []
        reason = "no_target"

        while len(removed) < args.max_extractions:
            remaining = [
                t for t in order
                if t not in removed and (args.retry_failed or t not in failed)
            ]
            if not remaining:
                reason = "no_target"
                break

            target = remaining[0]
            force_target(target)
            obs = wrapped.get_observations()

            outcome = "timeout"
            for _ in range(args.attempt_steps):
                with torch.no_grad():
                    action = policy(obs)
                obs, _, _, _ = wrapped.step(action)
                if bool(cfg.tower_damage(env)[0].item()):
                    outcome = "damage"
                    break
                if bool(cfg.success_block_extract(env)[0].item()):
                    outcome = "success"
                    break

            if args.verbose:
                print(f"  game {game}: {target:6} -> {outcome}", flush=True)

            if outcome == "damage":
                reason = "damage"
                break
            if outcome == "success":
                remove_block(target)
                if not args.cumulative:
                    rebaseline()
                removed.append(target)
            else:
                failed.append(target)
                if args.retry_failed and set(failed) >= set(remaining):
                    reason = "all_stalled"
                    break

        row = {
            "game": game,
            "extracted": len(removed),
            "end_reason": reason,
            "order_removed": " ".join(removed),
            "stalled": " ".join(failed),
        }
        rows.append(row)
        print(
            f"game {game:3}: {len(removed)} blocks, ended on {reason}"
            f"  [{' '.join(removed)}]",
            flush=True,
        )

    env.close()

    counts = [r["extracted"] for r in rows]
    print()
    print(f"games:               {len(counts)}")
    print(f"blocks per game:     mean {sum(counts)/len(counts):.2f}, "
          f"min {min(counts)}, max {max(counts)}")
    for reason in ("damage", "no_target", "all_stalled"):
        n = sum(1 for r in rows if r["end_reason"] == reason)
        if n:
            print(f"  ended on {reason:12} {n:3} of {len(rows)}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
