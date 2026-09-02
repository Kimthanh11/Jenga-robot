"""Evaluate scripted controllers under the policy evaluation protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import mjlab_jenga.jenga_mjenv_cfg as cfg
from mjlab.envs import ManagerBasedRlEnv
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


CONTROLLERS = ("settle", "straight", "pulsed", "tap")


def _actions(
    controller: str,
    *,
    num_envs: int,
    action_dim: int,
    live: torch.Tensor,
    contact_seen: torch.Tensor,
    contact_age: torch.Tensor,
    push_steps: int,
    pause_steps: int,
    retreat_steps: int,
    contact_x: float,
    contact_z: float,
    device: str,
) -> torch.Tensor:
    action = torch.zeros((num_envs, action_dim), device=device)
    action[live, 1] = contact_x
    action[live, 2] = contact_z
    if controller == "settle":
        return action
    if controller == "straight":
        action[live, 0] = -1.0
        return action

    # Reach the face continuously. Once contact has been observed, use a fixed
    # schedule; no block pose, progress or tower-state feedback enters the baseline.
    approaching = live & ~contact_seen
    action[approaching, 0] = -1.0
    after_contact = live & contact_seen
    if controller == "pulsed":
        cycle = max(push_steps + pause_steps, 1)
        phase = torch.remainder(contact_age, cycle)
        action[after_contact & (phase < push_steps), 0] = -1.0
        return action
    if controller == "tap":
        cycle = max(push_steps + retreat_steps + pause_steps, 1)
        phase = torch.remainder(contact_age, cycle)
        pushing = after_contact & (phase < push_steps)
        retreating = after_contact & (phase >= push_steps) & (
            phase < push_steps + retreat_steps
        )
        action[pushing, 0] = -1.0
        action[retreating, 0] = 1.0
        return action
    raise ValueError(f"Unknown controller: {controller}")


def _reason(env: ManagerBasedRlEnv, env_id: int) -> str:
    if bool(cfg.success_block_extract(env)[env_id].item()):
        return "success"
    if bool(cfg.tower_damage_signal(env)[env_id].item()):
        return "tower_damage"
    if bool(env.reset_time_outs[env_id].item()):
        return "timeout"
    return "terminated"


def _episode_row(
    *,
    env: ManagerBasedRlEnv,
    env_id: int,
    controller: str,
    target_set: str,
    target: str,
    missing_level: int,
    commit: str,
    base_seed: int,
    reset_seed: int,
    batch_index: int,
    steps: torch.Tensor,
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
    push_steps: int,
    pause_steps: int,
    retreat_phase_steps: int,
    contact_x: float,
    contact_z: float,
) -> dict:
    pattern_id = int(env._jenga_missing_pattern_id[env_id].item())
    episode_steps = max(int(steps[env_id].item()), 1)
    progress = float(cfg.block_progress(env)[env_id].item())
    tower_xy = float(cfg.tower_max_block_horizontal_shift(env)[env_id].item())
    tower_z = float(cfg.tower_max_block_vertical_shift(env)[env_id].item())
    tower_rot = float(cfg.tower_max_block_rotation(env)[env_id].item())
    groups = target_groups(cfg)
    return {
        "controller": controller,
        "checkpoint": "",
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
        "yaw_mode": "frozen",
        "episode_step_cap": episode_step_cap,
        "success_distance": float(cfg.success_done_distance(env).item()),
        "density_randomization": cfg.RESET_DENSITY_RANDOMIZATION,
        "friction_sliding_min": cfg.RESET_FRICTION_SLIDING_RANGE[0],
        "friction_sliding_max": cfg.RESET_FRICTION_SLIDING_RANGE[1],
        "sim_impratio": env.cfg.sim.mujoco.impratio,
        "sim_cone": str(env.cfg.sim.mujoco.cone),
        "push_steps": push_steps,
        "pause_steps": pause_steps,
        "retreat_steps": retreat_phase_steps,
        "contact_x": contact_x,
        "contact_z": contact_z,
        "control_semantics": (
            "open_loop" if controller in {"settle", "straight"} else "contact_triggered"
        ),
        "reason": _reason(env, env_id),
        "steps": episode_steps,
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
        "contact_rate": float(contact_steps[env_id].item()) / episode_steps,
        "contact_force_mean": float(force_sum[env_id].item()) / episode_steps,
        "contact_force_max": float(force_max[env_id].item()),
        "stuck_rate": float(stuck_steps[env_id].item()) / episode_steps,
        "stop_rate": float(stop_steps[env_id].item()) / episode_steps,
        "retreat_rate": float(retreat_steps[env_id].item()) / episode_steps,
    }


def _run_batch(
    *,
    env: ManagerBasedRlEnv,
    controller: str,
    active_count: int,
    target_set: str,
    target: str,
    missing_level: int,
    commit: str,
    base_seed: int,
    batch_index: int,
    max_steps: int,
    push_steps: int,
    pause_steps: int,
    retreat_steps: int,
    contact_x: float,
    contact_z: float,
) -> list[dict]:
    cfg.FORCED_MISSING_PATTERN_OFFSET = batch_index * env.num_envs
    reset_seed = evaluation_seed(base_seed, batch_index)
    env.reset(seed=reset_seed)
    validate_evaluation_reset(env, cfg, target, active_count)

    active = torch.arange(env.num_envs, device=env.device) < active_count
    finished = ~active
    contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    contact_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    steps = torch.zeros(env.num_envs, device=env.device)
    progress_max = torch.full((env.num_envs,), -torch.inf, device=env.device)
    tower_xy_max = torch.zeros(env.num_envs, device=env.device)
    tower_z_max = torch.zeros(env.num_envs, device=env.device)
    tower_rot_max = torch.zeros(env.num_envs, device=env.device)
    contact_steps = torch.zeros(env.num_envs, device=env.device)
    force_sum = torch.zeros(env.num_envs, device=env.device)
    force_max = torch.zeros(env.num_envs, device=env.device)
    stuck_steps = torch.zeros(env.num_envs, device=env.device)
    stop_steps = torch.zeros(env.num_envs, device=env.device)
    retreat_action_steps = torch.zeros(env.num_envs, device=env.device)
    rows: list[dict] = []

    for _ in range(max_steps):
        live = active & ~finished
        if not bool(live.any().item()):
            break
        action = _actions(
            controller,
            num_envs=env.num_envs,
            action_dim=env.action_manager.total_action_dim,
            live=live,
            contact_seen=contact_seen,
            contact_age=contact_age,
            push_steps=push_steps,
            pause_steps=pause_steps,
            retreat_steps=retreat_steps,
            contact_x=contact_x,
            contact_z=contact_z,
            device=env.device,
        )
        _, _, terminated, truncated, _ = env.step(action)
        dones = terminated | truncated

        progress = cfg.block_progress(env)
        tower_xy = cfg.tower_max_block_horizontal_shift(env)
        tower_z = cfg.tower_max_block_vertical_shift(env)
        tower_rot = cfg.tower_max_block_rotation(env)
        contact = cfg.hook_contact_found(env)
        force = cfg.hook_contact_force_norm(env)

        steps[live] += 1
        progress_max[live] = torch.maximum(progress_max[live], progress[live])
        tower_xy_max[live] = torch.maximum(tower_xy_max[live], tower_xy[live])
        tower_z_max[live] = torch.maximum(tower_z_max[live], tower_z[live])
        tower_rot_max[live] = torch.maximum(tower_rot_max[live], tower_rot[live])
        contact_steps[live] += contact[live]
        force_sum[live] += force[live]
        force_max[live] = torch.maximum(force_max[live], force[live])
        stuck_steps[live] += cfg.stuck_contact_signal(env)[live]
        stop_steps[live] += cfg.stop_action_fraction(env)[live]
        retreat_action_steps[live] += cfg.retreat_action_fraction(env)[live]

        was_seen = contact_seen.clone()
        contact_seen[live] |= contact[live] > 0.0
        contact_age[live & was_seen] += 1

        first_done = torch.nonzero(dones & live, as_tuple=False).squeeze(-1)
        for env_id in first_done.tolist():
            rows.append(
                _episode_row(
                    env=env,
                    env_id=env_id,
                    controller=controller,
                    target_set=target_set,
                    target=target,
                    missing_level=missing_level,
                    commit=commit,
                    base_seed=base_seed,
                    reset_seed=reset_seed,
                    batch_index=batch_index,
                    steps=steps,
                    progress_max=progress_max,
                    tower_xy_max=tower_xy_max,
                    tower_z_max=tower_z_max,
                    tower_rot_max=tower_rot_max,
                    contact_steps=contact_steps,
                    force_sum=force_sum,
                    force_max=force_max,
                    stuck_steps=stuck_steps,
                    stop_steps=stop_steps,
                    retreat_steps=retreat_action_steps,
                    episode_step_cap=max_steps,
                    push_steps=push_steps,
                    pause_steps=pause_steps,
                    retreat_phase_steps=retreat_steps,
                    contact_x=contact_x,
                    contact_z=contact_z,
                )
            )
        finished[first_done] = True

        done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            env.reset(env_ids=done_ids)

    unfinished = torch.nonzero(active & ~finished, as_tuple=False).squeeze(-1)
    for env_id in unfinished.tolist():
        row = _episode_row(
            env=env,
            env_id=env_id,
            controller=controller,
            target_set=target_set,
            target=target,
            missing_level=missing_level,
            commit=commit,
            base_seed=base_seed,
            reset_seed=reset_seed,
            batch_index=batch_index,
            steps=steps,
            progress_max=progress_max,
            tower_xy_max=tower_xy_max,
            tower_z_max=tower_z_max,
            tower_rot_max=tower_rot_max,
            contact_steps=contact_steps,
            force_sum=force_sum,
            force_max=force_max,
            stuck_steps=stuck_steps,
            stop_steps=stop_steps,
            retreat_steps=retreat_action_steps,
            episode_step_cap=max_steps,
            push_steps=push_steps,
            pause_steps=pause_steps,
            retreat_phase_steps=retreat_steps,
            contact_x=contact_x,
            contact_z=contact_z,
        )
        row["reason"] = "step_cap"
        rows.append(row)

    if len(rows) != active_count:
        raise RuntimeError(f"Expected {active_count} episodes, got {len(rows)}.")
    return rows


def _evaluate_case(
    *,
    controller: str,
    target_set: str,
    target: str,
    missing_level: int,
    episodes_per_seed: int,
    num_envs: int,
    seeds: tuple[int, ...],
    max_steps: int,
    device: str,
    push_steps: int,
    pause_steps: int,
    retreat_steps: int,
    contact_x: float,
    contact_z: float,
) -> list[dict]:
    configure_evaluation_case(cfg, target, missing_level)
    vector_size = min(num_envs, episodes_per_seed)
    env_cfg = cfg.jenga_env_cfg()
    env_cfg.scene.num_envs = vector_size
    env_cfg.auto_reset = False
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.commands["target_block"].force_target_name = target
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    commit = git_commit()
    rows: list[dict] = []
    try:
        batch_index = 0
        for base_seed in seeds:
            remaining = episodes_per_seed
            while remaining > 0:
                active_count = min(vector_size, remaining)
                rows.extend(
                    _run_batch(
                        env=env,
                        controller=controller,
                        active_count=active_count,
                        target_set=target_set,
                        target=target,
                        missing_level=missing_level,
                        commit=commit,
                        base_seed=base_seed,
                        batch_index=batch_index,
                        max_steps=max_steps,
                        push_steps=push_steps,
                        pause_steps=pause_steps,
                        retreat_steps=retreat_steps,
                        contact_x=contact_x,
                        contact_z=contact_z,
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
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--targets", default="trained")
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
    parser.add_argument("--push-steps", type=int, default=20)
    parser.add_argument("--pause-steps", type=int, default=10)
    parser.add_argument("--retreat-steps", type=int, default=5)
    parser.add_argument("--contact-x", type=float, default=0.0)
    parser.add_argument("--contact-z", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes-csv")
    parser.add_argument("--summary-csv")
    args = parser.parse_args()

    controllers = tuple(
        item.strip().lower() for item in args.controllers.split(",") if item.strip()
    )
    unknown = sorted(set(controllers) - set(CONTROLLERS))
    if unknown:
        parser.error(f"unknown controllers: {unknown}; choose from {CONTROLLERS}")
    if len(set(controllers)) != len(controllers):
        parser.error("controllers contain duplicates")
    if args.episodes_per_seed <= 0 or args.num_envs <= 0 or args.max_steps <= 0:
        parser.error("episodes, num-envs, and max-steps must be positive")
    if min(args.push_steps, args.pause_steps, args.retreat_steps) < 0:
        parser.error("controller phase lengths must be non-negative")
    if not -1.0 <= args.contact_x <= 1.0 or not -1.0 <= args.contact_z <= 1.0:
        parser.error("contact-x and contact-z must be in [-1, 1]")
    if any(level < 0 or level > 3 for level in args.missing_levels):
        parser.error("missing levels must be between 0 and 3")

    cfg.apply_low_level_stage("target")
    cfg.SUCCESS_CURRICULUM_START = cfg.SUCCESS_CURRICULUM_END
    cfg.YAW_CURRICULUM_START = 0.0
    cfg.YAW_CURRICULUM_END = 0.0
    try:
        target_set, targets = resolve_targets(args.targets, cfg)
        seeds = parse_int_csv(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"controllers={controllers} targets={target_set} ({len(targets)}) "
        f"seeds={seeds} episodes_per_seed={args.episodes_per_seed} "
        f"max_steps={args.max_steps}",
        flush=True,
    )
    for controller in controllers:
        for missing_level in args.missing_levels:
            for target in targets:
                rows = _evaluate_case(
                    controller=controller,
                    target_set=target_set,
                    target=target,
                    missing_level=missing_level,
                    episodes_per_seed=args.episodes_per_seed,
                    num_envs=args.num_envs,
                    seeds=seeds,
                    max_steps=args.max_steps,
                    device=args.device,
                    push_steps=args.push_steps,
                    pause_steps=args.pause_steps,
                    retreat_steps=args.retreat_steps,
                    contact_x=args.contact_x,
                    contact_z=args.contact_z,
                )
                summary = summarize_episode_rows(rows)
                append_rows(args.episodes_csv, rows)
                append_rows(args.summary_csv, [summary])
                print(
                    f"{controller:>8} {target:>5} missing={missing_level} "
                    f"n={summary['episodes']} success={summary['success_rate']:.3f} "
                    f"damage={summary['tower_damage_rate']:.3f} "
                    f"progress={summary['progress_max_mean']:.4f} "
                    f"steps={summary['episode_length_mean']:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
