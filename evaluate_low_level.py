"""Reproducible per-target evaluation of a trained Jenga policy.

Every original vector environment contributes exactly one episode.  This avoids the
fast-episode bias of repeatedly resetting successful environments while slow failures
are still running.  The per-episode CSV is the primary result; the summary CSV is a
convenience view with confidence intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_jenga.jenga_mjenv_cfg as cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab_jenga.evaluation_utils import (
    append_rows,
    block_layer,
    configure_evaluation_case,
    evaluation_seed,
    git_commit,
    parse_int_csv,
    pattern_label,
    resolve_targets,
    scenario_id,
    summarize_episode_rows,
    target_groups,
    validate_evaluation_reset,
)


task = cfg

LEGACY_DISTRIBUTION_CFG = {
    "class_name": "GaussianDistribution",
    "init_std": 0.8,
    "std_type": "scalar",
}


def _terminal_reason(env: ManagerBasedRlEnv, env_id: int, abort_enabled: bool) -> str:
    if bool(cfg.success_block_extract(env)[env_id].item()):
        return "success"
    if bool(cfg.tower_damage_signal(env)[env_id].item()):
        return "tower_damage"
    if abort_enabled:
        raw_abort = env.action_manager.get_term("abort").raw_action[env_id, 0]
        if bool((raw_abort > task.abort_threshold(env)).item()):
            return "abort"
    if bool(env.reset_time_outs[env_id].item()):
        return "timeout"
    return "terminated"


def _episode_row(
    *,
    env: ManagerBasedRlEnv,
    env_id: int,
    target_set: str,
    target: str,
    missing_level: int,
    checkpoint: Path,
    commit: str,
    base_seed: int,
    reset_seed: int,
    batch_index: int,
    episode_steps: torch.Tensor,
    progress_max: torch.Tensor,
    tower_xy_max: torch.Tensor,
    tower_z_max: torch.Tensor,
    tower_rot_max: torch.Tensor,
    contact_steps: torch.Tensor,
    force_sum: torch.Tensor,
    force_max: torch.Tensor,
    stuck_steps: torch.Tensor,
    stop_steps: torch.Tensor,
    retreat_steps: torch.Tensor,
    episode_step_cap: int,
    freeze_yaw: bool,
    abort_enabled: bool,
) -> dict:
    pattern_id = int(env._jenga_missing_pattern_id[env_id].item())
    steps = max(int(episode_steps[env_id].item()), 1)
    progress = float(cfg.block_progress(env)[env_id].item())
    tower_xy = float(cfg.tower_max_block_horizontal_shift(env)[env_id].item())
    tower_z = float(cfg.tower_max_block_vertical_shift(env)[env_id].item())
    tower_rot = float(cfg.tower_max_block_rotation(env)[env_id].item())
    groups = target_groups(cfg)
    return {
        "controller": "policy",
        "checkpoint": str(checkpoint),
        "commit": commit,
        "target_set": target_set,
        "target": target,
        "layer": block_layer(target),
        "is_trained": target in groups["trained"],
        "is_legal": target in groups["legal"],
        "missing_level": missing_level,
        "missing_pattern_id": pattern_id,
        "missing_pattern": pattern_label(cfg, pattern_id),
        "base_seed": base_seed,
        "reset_seed": reset_seed,
        "batch_index": batch_index,
        "env_index": env_id,
        "scenario_id": scenario_id(
            target, missing_level, base_seed, batch_index, env_id
        ),
        "yaw_mode": "frozen" if freeze_yaw else "configured",
        "episode_step_cap": episode_step_cap,
        "success_distance": float(cfg.success_done_distance(env).item()),
        "density_randomization": cfg.RESET_DENSITY_RANDOMIZATION,
        "friction_sliding_min": cfg.RESET_FRICTION_SLIDING_RANGE[0],
        "friction_sliding_max": cfg.RESET_FRICTION_SLIDING_RANGE[1],
        "sim_impratio": env.cfg.sim.mujoco.impratio,
        "sim_cone": str(env.cfg.sim.mujoco.cone),
        "reason": _terminal_reason(env, env_id, abort_enabled),
        "steps": steps,
        "extracted": bool(cfg.target_extraction_reached(env)[env_id].item()),
        "safe_success": bool(cfg.success_block_extract(env)[env_id].item()),
        "tower_damage": bool(cfg.tower_damage_signal(env)[env_id].item()),
        "progress_final": progress,
        "progress_max": float(progress_max[env_id].item()),
        "tower_xy_final": tower_xy,
        "tower_xy_max": float(tower_xy_max[env_id].item()),
        "tower_xy_recovery": float(tower_xy_max[env_id].item()) - tower_xy,
        "tower_z_final": tower_z,
        "tower_z_max": float(tower_z_max[env_id].item()),
        "tower_z_recovery": float(tower_z_max[env_id].item()) - tower_z,
        "tower_rot_deg_final": tower_rot * 57.295779513,
        "tower_rot_deg_max": float(tower_rot_max[env_id].item()) * 57.295779513,
        "tower_rot_deg_recovery": (
            float(tower_rot_max[env_id].item()) - tower_rot
        ) * 57.295779513,
        "contact_rate": float(contact_steps[env_id].item()) / steps,
        "contact_force_mean": float(force_sum[env_id].item()) / steps,
        "contact_force_max": float(force_max[env_id].item()),
        "stuck_rate": float(stuck_steps[env_id].item()) / steps,
        "stop_rate": float(stop_steps[env_id].item()) / steps,
        "retreat_rate": float(retreat_steps[env_id].item()) / steps,
    }


def _run_policy_batch(
    *,
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy,
    active_count: int,
    target_set: str,
    target: str,
    missing_level: int,
    checkpoint: Path,
    commit: str,
    base_seed: int,
    batch_index: int,
    max_steps: int,
    freeze_yaw: bool,
    abort_enabled: bool,
) -> list[dict]:
    cfg.FORCED_MISSING_PATTERN_OFFSET = batch_index * env.num_envs
    reset_seed = evaluation_seed(base_seed, batch_index)
    env.reset(seed=reset_seed)
    validate_evaluation_reset(env, cfg, target, active_count)
    obs = wrapped.get_observations()

    active = torch.arange(env.num_envs, device=env.device) < active_count
    finished = ~active
    episode_steps = torch.zeros(env.num_envs, device=env.device)
    progress_max = torch.full((env.num_envs,), -torch.inf, device=env.device)
    tower_xy_max = torch.zeros(env.num_envs, device=env.device)
    tower_z_max = torch.zeros(env.num_envs, device=env.device)
    tower_rot_max = torch.zeros(env.num_envs, device=env.device)
    contact_steps = torch.zeros(env.num_envs, device=env.device)
    force_sum = torch.zeros(env.num_envs, device=env.device)
    force_max = torch.zeros(env.num_envs, device=env.device)
    stuck_steps = torch.zeros(env.num_envs, device=env.device)
    stop_steps = torch.zeros(env.num_envs, device=env.device)
    retreat_steps = torch.zeros(env.num_envs, device=env.device)
    rows: list[dict] = []

    # mjlab mutates simulator and manager tensors during step/reset.  Inference mode
    # marks newly created tensors as immutable outside its scope, which breaks a later
    # manager reset.  no_grad disables autograd without imposing that restriction.
    with torch.no_grad():
        for _ in range(max_steps):
            live = active & ~finished
            if not bool(live.any().item()):
                break

            actions = policy(obs).clone()
            actions[~live] = 0.0
            obs, _, dones, _ = wrapped.step(actions)

            progress = cfg.block_progress(env)
            tower_xy = cfg.tower_max_block_horizontal_shift(env)
            tower_z = cfg.tower_max_block_vertical_shift(env)
            tower_rot = cfg.tower_max_block_rotation(env)
            contact = cfg.hook_contact_found(env)
            force = cfg.hook_contact_force_norm(env)

            episode_steps[live] += 1
            progress_max[live] = torch.maximum(progress_max[live], progress[live])
            tower_xy_max[live] = torch.maximum(tower_xy_max[live], tower_xy[live])
            tower_z_max[live] = torch.maximum(tower_z_max[live], tower_z[live])
            tower_rot_max[live] = torch.maximum(tower_rot_max[live], tower_rot[live])
            contact_steps[live] += contact[live]
            force_sum[live] += force[live]
            force_max[live] = torch.maximum(force_max[live], force[live])
            stuck_steps[live] += cfg.stuck_contact_signal(env)[live]
            stop_steps[live] += cfg.stop_action_fraction(env)[live]
            retreat_steps[live] += cfg.retreat_action_fraction(env)[live]

            first_done = torch.nonzero(dones & live, as_tuple=False).squeeze(-1)
            for env_id in first_done.tolist():
                rows.append(
                    _episode_row(
                        env=env,
                        env_id=env_id,
                        target_set=target_set,
                        target=target,
                        missing_level=missing_level,
                        checkpoint=checkpoint,
                        commit=commit,
                        base_seed=base_seed,
                        reset_seed=reset_seed,
                        batch_index=batch_index,
                        episode_steps=episode_steps,
                        progress_max=progress_max,
                        tower_xy_max=tower_xy_max,
                        tower_z_max=tower_z_max,
                        tower_rot_max=tower_rot_max,
                        contact_steps=contact_steps,
                        force_sum=force_sum,
                        force_max=force_max,
                        stuck_steps=stuck_steps,
                        stop_steps=stop_steps,
                        retreat_steps=retreat_steps,
                        episode_step_cap=max_steps,
                        freeze_yaw=freeze_yaw,
                        abort_enabled=abort_enabled,
                    )
                )
            finished[first_done] = True

            # Manual-reset mode requires every terminated environment to be reset,
            # including inactive or already recorded ones. They remain ignored.
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                env.reset(env_ids=done_ids)
                obs = wrapped.get_observations()

    unfinished = torch.nonzero(active & ~finished, as_tuple=False).squeeze(-1)
    for env_id in unfinished.tolist():
        row = _episode_row(
            env=env,
            env_id=env_id,
            target_set=target_set,
            target=target,
            missing_level=missing_level,
            checkpoint=checkpoint,
            commit=commit,
            base_seed=base_seed,
            reset_seed=reset_seed,
            batch_index=batch_index,
            episode_steps=episode_steps,
            progress_max=progress_max,
            tower_xy_max=tower_xy_max,
            tower_z_max=tower_z_max,
            tower_rot_max=tower_rot_max,
            contact_steps=contact_steps,
            force_sum=force_sum,
            force_max=force_max,
            stuck_steps=stuck_steps,
            stop_steps=stop_steps,
            retreat_steps=retreat_steps,
            episode_step_cap=max_steps,
            freeze_yaw=freeze_yaw,
            abort_enabled=abort_enabled,
        )
        row["reason"] = "step_cap"
        rows.append(row)
        finished[env_id] = True

    if len(rows) != active_count:
        raise RuntimeError(
            f"Expected {active_count} first episodes, recorded {len(rows)}."
        )
    return rows


def _evaluate_case(
    *,
    checkpoint: Path,
    target_set: str,
    target: str,
    missing_level: int,
    episodes_per_seed: int,
    num_envs: int,
    seeds: tuple[int, ...],
    max_steps: int,
    device: str,
    legacy_distribution: bool,
    impratio: float | None,
    cone: str | None,
    freeze_yaw: bool,
    abort_enabled: bool,
) -> list[dict]:
    configure_evaluation_case(cfg, target, missing_level)
    vector_size = min(num_envs, episodes_per_seed)
    env_cfg = task.jenga_env_cfg()
    env_cfg.scene.num_envs = vector_size
    env_cfg.auto_reset = False
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.commands["target_block"].force_target_name = target
    if impratio is not None:
        env_cfg.sim.mujoco.impratio = impratio
    if cone is not None:
        env_cfg.sim.mujoco.cone = cone

    agent_cfg = task.jenga_ppo_runner_cfg()
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
    commit = git_commit()
    rows: list[dict] = []

    try:
        batch_index = 0
        for base_seed in seeds:
            remaining = episodes_per_seed
            while remaining > 0:
                active_count = min(vector_size, remaining)
                rows.extend(
                    _run_policy_batch(
                        env=env,
                        wrapped=wrapped,
                        policy=policy,
                        active_count=active_count,
                        target_set=target_set,
                        target=target,
                        missing_level=missing_level,
                        checkpoint=checkpoint,
                        commit=commit,
                        base_seed=base_seed,
                        batch_index=batch_index,
                        max_steps=max_steps,
                        freeze_yaw=freeze_yaw,
                        abort_enabled=abort_enabled,
                    )
                )
                remaining -= active_count
                batch_index += 1
    finally:
        env.close()

    expected = episodes_per_seed * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} episodes, got {len(rows)}.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", default="trained")
    parser.add_argument(
        "--target-index",
        type=int,
        help="Evaluate only this zero-based member of the resolved target set while "
        "retaining the set label. Used by target-sharded SLURM evaluations.",
    )
    parser.add_argument("--missing-levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--episodes",
        "--episodes-per-seed",
        dest="episodes_per_seed",
        type=int,
        default=50,
    )
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes-csv")
    parser.add_argument("--summary-csv")
    parser.add_argument("--csv", help="Deprecated alias for --summary-csv.")
    parser.add_argument("--legacy-distribution", action="store_true")
    parser.add_argument("--yaw-limit", type=float)
    yaw_group = parser.add_mutually_exclusive_group()
    yaw_group.add_argument("--freeze-yaw", dest="freeze_yaw", action="store_true")
    yaw_group.add_argument("--enable-yaw", dest="freeze_yaw", action="store_false")
    parser.set_defaults(freeze_yaw=True)
    parser.add_argument("--abort", action="store_true")
    parser.add_argument("--impratio", type=float)
    parser.add_argument("--cone")
    args = parser.parse_args()

    if args.episodes_per_seed <= 0 or args.num_envs <= 0 or args.max_steps <= 0:
        parser.error("episodes, num-envs, and max-steps must be positive")
    if any(level < 0 or level > 3 for level in args.missing_levels):
        parser.error("missing levels must be between 0 and 3")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    if args.abort:
        global task
        import mjlab_jenga.jenga_abort_cfg as abort_cfg

        task = abort_cfg

    cfg.apply_low_level_stage("target")
    # Report evaluation always uses the final extraction distance, independent of the
    # training counter stored in the checkpoint.
    cfg.SUCCESS_CURRICULUM_START = cfg.SUCCESS_CURRICULUM_END
    if args.yaw_limit is not None:
        cfg.YAW_TARGET_LIMIT = args.yaw_limit
    if args.freeze_yaw:
        cfg.YAW_CURRICULUM_START = 0.0
        cfg.YAW_CURRICULUM_END = 0.0

    try:
        target_set, targets = resolve_targets(args.targets, cfg)
        seeds = parse_int_csv(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.target_index is not None:
        if args.target_index < 0 or args.target_index >= len(targets):
            parser.error(
                f"target-index {args.target_index} is outside {target_set} "
                f"with {len(targets)} targets"
            )
        targets = (targets[args.target_index],)

    summary_path = args.summary_csv or args.csv
    print(
        f"targets={target_set} selected={targets} seeds={seeds} "
        f"episodes_per_seed={args.episodes_per_seed} max_steps={args.max_steps} "
        f"yaw={'frozen' if args.freeze_yaw else 'configured'}",
        flush=True,
    )

    for missing_level in args.missing_levels:
        for target in targets:
            rows = _evaluate_case(
                checkpoint=checkpoint,
                target_set=target_set,
                target=target,
                missing_level=missing_level,
                episodes_per_seed=args.episodes_per_seed,
                num_envs=args.num_envs,
                seeds=seeds,
                max_steps=args.max_steps,
                device=args.device,
                legacy_distribution=args.legacy_distribution,
                impratio=args.impratio,
                cone=args.cone,
                freeze_yaw=args.freeze_yaw,
                abort_enabled=args.abort,
            )
            summary = summarize_episode_rows(rows)
            append_rows(args.episodes_csv, rows)
            append_rows(summary_path, [summary])
            print(
                f"{target:>5} missing={missing_level} n={summary['episodes']} "
                f"extracted={summary['extraction_rate']:.3f} "
                f"success={summary['success_rate']:.3f} "
                f"damage={summary['tower_damage_rate']:.3f} "
                f"progress={summary['progress_max_mean']:.4f} "
                f"steps={summary['episode_length_mean']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
