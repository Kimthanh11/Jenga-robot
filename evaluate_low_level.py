from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_jenga.jenga_mjenv_cfg as cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


def _parse_targets(value: str) -> list[str]:
    if value == "all":
        return list(cfg.RANDOM_TARGET_BLOCK_NAMES)
    return [item.strip() for item in value.split(",") if item.strip()]


def _set_eval_curriculum(missing_level: int) -> None:
    cfg.FORCED_MISSING_BLOCK_COUNT = missing_level
    cfg.MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = -1
    cfg.MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 1
    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 1.0 if missing_level > 0 else 0.0
    cfg.RANDOM_TARGET_BLOCK_BEGIN_STEP = 10**12
    cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 10**12


def _evaluate_case(
    checkpoint: Path,
    target: str,
    missing_level: int,
    episodes: int,
    num_envs: int,
    max_steps: int,
    device: str,
) -> dict[str, float | int | str]:
    _set_eval_curriculum(missing_level)

    env_cfg = cfg.jenga_env_cfg(play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.auto_reset = False
    env_cfg.commands["target_block"].force_target_name = target

    agent_cfg = cfg.jenga_ppo_runner_cfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    completed = 0
    success_count = 0
    tower_large_count = 0
    progress_sum = 0.0
    progress_max = 0.0
    length_sum = 0.0

    obs = wrapped.get_observations()
    steps = 0
    with torch.inference_mode():
        while completed < episodes and steps < max_steps:
            action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
            steps += 1

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() == 0:
                continue

            remaining = episodes - completed
            collect_ids = done_ids[:remaining]
            progress = cfg.block_progress(env)[collect_ids].detach()
            success = cfg.success_block_extract(env)[collect_ids].detach()
            tower_large = cfg.tower_large_perturbation(env)[collect_ids].detach()
            lengths = env.episode_length_buf[collect_ids].detach()

            completed += int(collect_ids.numel())
            success_count += int(success.sum().item())
            tower_large_count += int(tower_large.sum().item())
            progress_sum += float(progress.sum().item())
            progress_max = max(progress_max, float(progress.max().item()))
            length_sum += float(lengths.float().sum().item())

            env.reset(env_ids=done_ids)
            obs = wrapped.get_observations()

    env.close()

    if completed == 0:
        return {
            "target": target,
            "missing_level": missing_level,
            "episodes": 0,
            "success_rate": 0.0,
            "tower_large_rate": 0.0,
            "progress_mean": 0.0,
            "progress_max": 0.0,
            "episode_length_mean": 0.0,
        }

    return {
        "target": target,
        "missing_level": missing_level,
        "episodes": completed,
        "success_rate": success_count / completed,
        "tower_large_rate": tower_large_count / completed,
        "progress_mean": progress_sum / completed,
        "progress_max": progress_max,
        "episode_length_mean": length_sum / completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Jenga low-level policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--missing-levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    targets = _parse_targets(args.targets)
    rows = []

    for missing_level in args.missing_levels:
        for target in targets:
            row = _evaluate_case(
                checkpoint=checkpoint,
                target=target,
                missing_level=missing_level,
                episodes=args.episodes,
                num_envs=args.num_envs,
                max_steps=args.max_steps,
                device=args.device,
            )
            rows.append(row)
            print(
                f"{target:>5} missing={missing_level} "
                f"success={row['success_rate']:.3f} "
                f"progress={row['progress_mean']:.4f} "
                f"tower_large={row['tower_large_rate']:.3f} "
                f"len={row['episode_length_mean']:.1f}",
                flush=True,
            )

    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
