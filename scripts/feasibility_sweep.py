"""Measure scripted extraction feasibility across targets and settings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _append_row(csv_path, row) -> None:
    """Append a result immediately so partial Slurm runs remain usable."""
    if csv_path is None:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()), lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted extraction feasibility sweep.")
    parser.add_argument("--targets", default="trained")
    parser.add_argument(
        "--contact-z",
        default="0.0",
        help="Comma-separated vertical contact points on the push face, in [-1, 1].",
    )
    parser.add_argument(
        "--contact-x",
        default="0.0",
        help="Comma-separated lateral contact points on the push face, in [-1, 1].",
    )
    parser.add_argument("--frictions", default="", help="Empty means sample per reset.")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--stall-steps", type=int, default=200)
    parser.add_argument("--stall-eps", type=float, default=0.0002)
    parser.add_argument("--lock-yaw", action="store_true")
    parser.add_argument(
        "--action-x",
        type=float,
        default=-1.0,
        help="Push action. Use 0 for a null-action settle test: the tower_max_* "
        "columns then show how much of the damage budget the reset transient eats.",
    )
    parser.add_argument(
        "--vertical-slack",
        type=float,
        default=0.0,
        help="Metres added to both vertical tower limits, as a proxy for a "
        "pre-settled tower. The stability criteria are evaluated against the nominal "
        "spawn pose, but the tower settles about 5.4 mm below it within the first "
        "steps of an episode, consuming part of the allowance before the policy acts.",
    )
    parser.add_argument(
        "--impratio",
        type=float,
        default=None,
        help="Override the MuJoCo impratio of the configuration. Mean drag ratio "
        "over the calibration targets: 0.664 at 1.0 (the MuJoCo default), 0.302 at "
        "10, 0.251 at 30. MuJoCo recommends 10-100 for friction-critical contact.",
    )
    parser.add_argument("--cone", default=None, help="pyramidal or elliptic.")
    parser.add_argument(
        "--success-fraction",
        type=float,
        default=0.75,
        help="Fraction of the block length that counts as extracted, held constant. "
        "0.75 = 112.5 mm, the full task.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import mjlab_jenga.jenga_mjenv_cfg as cfg
    from mjlab_jenga.evaluation_utils import resolve_targets

    try:
        _, resolved_targets = resolve_targets(args.targets, cfg)
    except ValueError as exc:
        parser.error(str(exc))
    targets = list(resolved_targets)
    contact_points = [
        (cx, cz)
        for cz in _parse_floats(args.contact_z)
        for cx in _parse_floats(args.contact_x)
    ]
    frictions: list[float | None] = _parse_floats(args.frictions) or [None]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg.apply_low_level_stage("fixed")
    cfg.YAW_CURRICULUM_START = 0.0
    cfg.YAW_CURRICULUM_END = 0.0
    # Feasibility is measured at the final extraction distance.
    cfg.SUCCESS_CURRICULUM_START = args.success_fraction
    cfg.SUCCESS_CURRICULUM_END = args.success_fraction
    if args.lock_yaw:
        cfg.YAW_TARGET_LIMIT = 0.0
    if args.vertical_slack:
        cfg.TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT += args.vertical_slack
        cfg.TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT += args.vertical_slack

    num_envs = len(targets) * len(contact_points)
    env_cfg = cfg.jenga_env_cfg(play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.auto_reset = False
    env_cfg.commands["target_block"].force_target_names = tuple(targets)
    if args.impratio is not None:
        env_cfg.sim.mujoco.impratio = args.impratio
    if args.cone is not None:
        env_cfg.sim.mujoco.cone = args.cone

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    rows: list[dict] = []
    try:
        hook = env.scene["hook"]
        slide_ids, _ = hook.find_joints(("hook_slide",), preserve_order=True)
        slide_dof = int(slide_ids[0])

        # Targets cycle fastest; contact points cycle across target batches.
        action = torch.zeros((num_envs, env.action_manager.total_action_dim), device=env.device)
        action[:, 0] = args.action_x
        for env_idx in range(num_envs):
            cx, cz = contact_points[env_idx // len(targets)]
            action[env_idx, 1] = cx
            action[env_idx, 2] = cz

        for friction in frictions:
            if friction is not None:
                cfg.RESET_FRICTION_SLIDING_RANGE = (friction, friction)
            for seed in seeds:
                rows.extend(
                    _run_pass(
                        env=env,
                        cfg=cfg,
                        torch=torch,
                        action=action,
                        targets=targets,
                        contact_points=contact_points,
                        slide_dof=slide_dof,
                        friction=friction,
                        seed=seed,
                        args=args,
                    )
                )
    finally:
        env.close()

    if args.csv is not None:
        print(f"Wrote {Path(args.csv)} ({len(rows)} rows)", flush=True)


def _run_pass(*, env, cfg, torch, action, targets, contact_points, slide_dof,
              friction, seed, args) -> list[dict]:
    num_envs = action.shape[0]
    dev = env.device
    env.reset(seed=seed)

    # Verify that forced target assignment reached the command term.
    cmd = env.command_manager.get_term("target_block")
    actual = [cmd._all_names[i] for i in cmd.selected_block_idx.tolist()]
    expected = [targets[i % len(targets)] for i in range(num_envs)]
    if actual != expected:
        raise RuntimeError(
            "Per-env target assignment did not take effect.\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )

    finished = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    reason = ["step_cap"] * num_envs
    best_progress = torch.full((num_envs,), float("-inf"), device=dev)
    stall_ref = torch.full((num_envs,), float("-inf"), device=dev)
    best_step = torch.zeros(num_envs, dtype=torch.long, device=dev)
    steps_to_contact = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
    contact_steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
    commanded_travel = torch.zeros(num_envs, device=dev)
    max_contact_force = torch.zeros(num_envs, device=dev)
    max_actuator_force = torch.zeros(num_envs, device=dev)
    final_extracted = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    final_success = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    final_damage = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    final_steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
    # Record the individual damage criterion for diagnosis.
    max_tower_xy = torch.zeros(num_envs, device=dev)
    max_tower_z = torch.zeros(num_envs, device=dev)
    max_tower_rot = torch.zeros(num_envs, device=dev)

    success_distance = float(cfg.success_done_distance(env).item())
    step_dt = float(env.step_dt)

    for step in range(1, args.max_steps + 1):
        _, _, terminated, truncated, _ = env.step(action)
        live = ~finished

        progress = cfg.block_progress(env)
        contact = cfg.hook_contact_found(env) > 0.0
        contact_force = cfg.hook_contact_force_norm(env)
        hook_force = env.scene["hook"].data.qfrc_actuator[:, slide_dof].abs()

        newly_touching = contact & (steps_to_contact < 0) & live
        steps_to_contact = torch.where(
            newly_touching, torch.full_like(steps_to_contact, step), steps_to_contact
        )
        contact_steps += (contact & live).long()
        after_contact = (steps_to_contact >= 0) & live
        commanded_travel += torch.where(
            after_contact, cfg.push_velocity_target(env).abs() * step_dt,
            torch.zeros_like(commanded_travel),
        )
        max_contact_force = torch.where(
            live, torch.maximum(max_contact_force, contact_force), max_contact_force
        )
        max_actuator_force = torch.where(
            live, torch.maximum(max_actuator_force, hook_force), max_actuator_force
        )
        max_tower_xy = torch.where(
            live, torch.maximum(max_tower_xy, cfg.tower_max_block_horizontal_shift(env)),
            max_tower_xy)
        max_tower_z = torch.where(
            live, torch.maximum(max_tower_z, cfg.tower_max_block_vertical_shift(env)),
            max_tower_z)
        max_tower_rot = torch.where(
            live, torch.maximum(max_tower_rot, cfg.tower_max_block_rotation(env)),
            max_tower_rot)

        improved = (progress > stall_ref + args.stall_eps) & live
        stall_ref = torch.where(improved, progress, stall_ref)
        best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
        best_progress = torch.where(
            live, torch.maximum(best_progress, progress), best_progress
        )

        extracted = cfg.target_extraction_reached(env)
        success = cfg.success_block_extract(env)
        damaged = cfg.tower_damage(env)
        done = (terminated | truncated) & live
        stalled = after_contact & ((step - best_step) >= args.stall_steps) & ~done
        closing = done | stalled

        if bool(closing.any().item()):
            for env_idx in closing.nonzero(as_tuple=False).squeeze(-1).tolist():
                if bool(done[env_idx].item()):
                    reason[env_idx] = (
                        "damage" if bool(damaged[env_idx].item())
                        else "success" if bool(success[env_idx].item())
                        else "timeout"
                    )
                else:
                    reason[env_idx] = "stalled"
            final_extracted = torch.where(closing, extracted, final_extracted)
            final_success = torch.where(closing, success, final_success)
            final_damage = torch.where(closing, damaged, final_damage)
            final_steps = torch.where(
                closing, torch.full_like(final_steps, step), final_steps
            )
            finished |= closing

        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            # Manual reset is required when auto_reset is disabled.
            env.reset(env_ids=done_ids)

        if bool(finished.all().item()):
            break

    final_steps = torch.where(
        final_steps == 0, torch.full_like(final_steps, args.max_steps), final_steps
    )

    rows = []
    for env_idx in range(num_envs):
        travel = float(commanded_travel[env_idx].item())
        progress_m = float(best_progress[env_idx].item())
        cx, cz = contact_points[env_idx // len(targets)]
        rows.append({
            "target": targets[env_idx % len(targets)],
            "layer": int(targets[env_idx % len(targets)][1:].split("_")[0]),
            "contact_x": cx,
            "contact_z": cz,
            "friction": "sampled" if friction is None else friction,
            "seed": seed,
            "lock_yaw": args.lock_yaw,
            "reason": reason[env_idx],
            "steps": int(final_steps[env_idx].item()),
            "steps_to_contact": int(steps_to_contact[env_idx].item()),
            "contact_steps": int(contact_steps[env_idx].item()),
            "max_progress": round(progress_m, 6),
            "success_distance": round(success_distance, 6),
            "commanded_travel": round(travel, 6),
            "slip_ratio": round(progress_m / travel, 4) if travel > 1e-9 else "",
            "max_actuator_force": round(float(max_actuator_force[env_idx].item()), 4),
            "max_contact_force": round(float(max_contact_force[env_idx].item()), 4),
            "tower_max_xy": round(float(max_tower_xy[env_idx].item()), 5),
            "tower_max_z": round(float(max_tower_z[env_idx].item()), 5),
            "tower_max_rot_deg": round(
                float(max_tower_rot[env_idx].item()) * 180.0 / 3.141592653589793, 2),
            "damage_mode": _damage_mode(cfg, max_tower_xy[env_idx],
                                        max_tower_z[env_idx], max_tower_rot[env_idx]),
            "extracted": bool(final_extracted[env_idx].item()),
            "tower_damage": bool(final_damage[env_idx].item()),
            "safe_success": bool(final_success[env_idx].item()),
        })
        _append_row(args.csv, rows[-1])
        row = rows[-1]
        print(
            f"{row['target']:>5} L{row['layer']} cz={cz:+.2f} "
            f"mu={row['friction']} seed={seed} "
            f"reason={row['reason']:<8} progress={row['max_progress']:.5f} "
            f"act_f={row['max_actuator_force']:.3f} "
            f"contact_f={row['max_contact_force']:.3f} "
            f"slip={row['slip_ratio']} success={row['safe_success']}",
            flush=True,
        )
    return rows


def _damage_mode(cfg, xy, z, rot) -> str:
    """Which damage limit was exceeded, if any."""
    modes = []
    if float(xy.item()) > cfg.TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT:
        modes.append("slide")
    if float(z.item()) > cfg.TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT:
        modes.append("sink")
    if float(rot.item()) > cfg.TOWER_DAMAGE_MAX_BLOCK_ROTATION:
        modes.append("tilt")
    return "+".join(modes) if modes else "none"


if __name__ == "__main__":
    main()
