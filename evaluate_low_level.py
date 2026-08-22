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
    cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = (
        1.0 if missing_level > 0 else 0.0
    )
    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 1.0 if missing_level > 0 else 0.0
    cfg.RANDOM_TARGET_BLOCK_BEGIN_STEP = 10**12
    cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 0.0


# Checkpoints trained before the distribution fix carry `std_param` (std_type
# "scalar"); the fixed config uses `log_std_param` (std_type "log"), so they no longer
# load with strict=True. Evaluation runs the deterministic policy, where std is
# irrelevant -- this switch just restores the old parameter name for loading.
LEGACY_DISTRIBUTION_CFG = {
    "class_name": "GaussianDistribution",
    "init_std": 0.8,
    "std_type": "scalar",
}


def _evaluate_case(
    checkpoint: Path,
    target: str,
    missing_level: int,
    episodes: int,
    num_envs: int,
    max_steps: int,
    device: str,
    legacy_distribution: bool = False,
    impratio: float | None = None,
    cone: str | None = None,
) -> dict[str, float | int | str]:
    _set_eval_curriculum(missing_level)

    env_cfg = cfg.jenga_env_cfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.auto_reset = False
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.commands["target_block"].force_target_name = target
    # Changing contact physics invalidates comparisons against runs made under the old
    # settings, so a checkpoint has to be re-evaluated under the new ones to compare.
    if impratio is not None:
        env_cfg.sim.mujoco.impratio = impratio
    if cone is not None:
        env_cfg.sim.mujoco.cone = cone

    agent_cfg = cfg.jenga_ppo_runner_cfg()
    if legacy_distribution:
        agent_cfg.actor.distribution_cfg = dict(LEGACY_DISTRIBUTION_CFG)
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
    extraction_count = 0
    tower_damage_count = 0
    progress_sum = 0.0
    progress_max = 0.0
    length_sum = 0.0
    contact_rate_sum = 0.0
    force_mean_sum = 0.0
    stuck_rate_sum = 0.0
    stop_rate_sum = 0.0
    retreat_rate_sum = 0.0

    episode_steps = torch.zeros(num_envs, device=device)
    episode_contact_steps = torch.zeros(num_envs, device=device)
    episode_force_sum = torch.zeros(num_envs, device=device)
    episode_stuck_steps = torch.zeros(num_envs, device=device)
    episode_stop_steps = torch.zeros(num_envs, device=device)
    episode_retreat_steps = torch.zeros(num_envs, device=device)
    episode_progress_max = torch.zeros(num_envs, device=device)

    obs = wrapped.get_observations()
    steps = 0
    with torch.inference_mode():
        while completed < episodes and steps < max_steps:
            action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
            steps += 1

            current_progress = cfg.block_progress(env)
            episode_steps += 1
            episode_contact_steps += cfg.hook_contact_found(env)
            episode_force_sum += cfg.hook_contact_force_norm(env)
            episode_stuck_steps += cfg.stuck_contact_signal(env)
            episode_stop_steps += cfg.stop_action_fraction(env)
            episode_retreat_steps += cfg.retreat_action_fraction(env)
            episode_progress_max = torch.maximum(episode_progress_max, current_progress)

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() == 0:
                continue

            remaining = episodes - completed
            collect_ids = done_ids[:remaining]
            progress = cfg.block_progress(env)[collect_ids].detach()
            extraction = cfg.target_extraction_reached(env)[collect_ids].detach()
            success = cfg.success_block_extract(env)[collect_ids].detach()
            tower_damage = cfg.tower_damage_signal(env)[collect_ids].detach()
            lengths = env.episode_length_buf[collect_ids].detach()

            completed += int(collect_ids.numel())
            extraction_count += int(extraction.sum().item())
            success_count += int(success.sum().item())
            tower_damage_count += int(tower_damage.sum().item())
            progress_sum += float(progress.sum().item())
            progress_max = max(
                progress_max,
                float(episode_progress_max[collect_ids].max().item()),
            )
            length_sum += float(lengths.float().sum().item())
            denominators = episode_steps[collect_ids].clamp_min(1.0)
            contact_rate_sum += float(
                (episode_contact_steps[collect_ids] / denominators).sum().item()
            )
            force_mean_sum += float(
                (episode_force_sum[collect_ids] / denominators).sum().item()
            )
            stuck_rate_sum += float(
                (episode_stuck_steps[collect_ids] / denominators).sum().item()
            )
            stop_rate_sum += float(
                (episode_stop_steps[collect_ids] / denominators).sum().item()
            )
            retreat_rate_sum += float(
                (episode_retreat_steps[collect_ids] / denominators).sum().item()
            )

            env.reset(env_ids=done_ids)
            episode_steps[done_ids] = 0.0
            episode_contact_steps[done_ids] = 0.0
            episode_force_sum[done_ids] = 0.0
            episode_stuck_steps[done_ids] = 0.0
            episode_stop_steps[done_ids] = 0.0
            episode_retreat_steps[done_ids] = 0.0
            episode_progress_max[done_ids] = 0.0
            obs = wrapped.get_observations()

    env.close()

    if completed == 0:
        return {
            "target": target,
            "missing_level": missing_level,
            "episodes": 0,
            "extraction_rate": 0.0,
            "success_rate": 0.0,
            "tower_damage_rate": 0.0,
            "progress_mean": 0.0,
            "progress_max": 0.0,
            "episode_length_mean": 0.0,
            "contact_rate": 0.0,
            "contact_force_mean": 0.0,
            "stuck_rate": 0.0,
            "stop_rate": 0.0,
            "retreat_rate": 0.0,
        }

    return {
        "target": target,
        "missing_level": missing_level,
        "episodes": completed,
        "extraction_rate": extraction_count / completed,
        "success_rate": success_count / completed,
        "tower_damage_rate": tower_damage_count / completed,
        "progress_mean": progress_sum / completed,
        "progress_max": progress_max,
        "episode_length_mean": length_sum / completed,
        "contact_rate": contact_rate_sum / completed,
        "contact_force_mean": force_mean_sum / completed,
        "stuck_rate": stuck_rate_sum / completed,
        "stop_rate": stop_rate_sum / completed,
        "retreat_rate": retreat_rate_sum / completed,
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
    parser.add_argument(
        "--legacy-distribution",
        action="store_true",
        help="Load a checkpoint trained before the bounded-std fix.",
    )
    parser.add_argument("--impratio", type=float, default=None)
    parser.add_argument("--cone", default=None)
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
                legacy_distribution=args.legacy_distribution,
                impratio=args.impratio,
                cone=args.cone,
            )
            rows.append(row)
            print(
                f"{target:>5} missing={missing_level} "
                f"extracted={row['extraction_rate']:.3f} "
                f"success={row['success_rate']:.3f} "
                f"progress={row['progress_mean']:.4f} "
                f"tower_damage={row['tower_damage_rate']:.3f} "
                f"contact={row['contact_rate']:.3f} "
                f"stuck={row['stuck_rate']:.3f} "
                f"retreat={row['retreat_rate']:.3f} "
                f"len={row['episode_length_mean']:.1f}",
                flush=True,
            )

    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
