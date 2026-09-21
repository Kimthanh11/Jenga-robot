"""Evaluate scripted controllers under the policy evaluation protocol.

Controllers: settle (no push), straight (constant maximum push), pulsed and tap
(contact-triggered schedules), forcelim (maximum push scaled down when the hook contact
force exceeds --force-limit) and stall (maximum push; when the contact force exceeds
--stall-force while the target progress grew by less than --stall-eps over the last
--stall-window steps, retreat for --stall-retreat-steps and then hold still). forcelim and
stall use only the signals the policy also observes (contact force, target progress).
--settle-steps N keeps simulating N steps after success with the push stopped and judges
the tower again, as evaluate_low_level.py --settle-steps does for the policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Repo root on sys.path so that `python paper_eval/scripts/X.py` finds mjlab_jenga_paper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mjlab_jenga_paper.jenga_mjenv_cfg as cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab_jenga_paper.evaluation_utils import (
    append_rows,
    block_layer,
    configure_evaluation_case,
    evaluation_seed,
    git_commit,
    parse_float_range,
    parse_int_csv,
    pattern_label,
    resolve_targets,
    scenario_id,
    summarize_episode_rows,
    target_groups,
    validate_evaluation_reset,
)


CONTROLLERS = ("settle", "straight", "pulsed", "tap", "forcelim", "stall")


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
    force: torch.Tensor,
    stalled: torch.Tensor,
    stall_age: torch.Tensor,
    force_limit: float,
    stall_retreat_steps: int,
) -> torch.Tensor:
    action = torch.zeros((num_envs, action_dim), device=device)
    action[live, 1] = contact_x
    action[live, 2] = contact_z
    if controller == "settle":
        return action
    if controller == "straight":
        action[live, 0] = -1.0
        return action
    if controller == "forcelim":
        # Push at maximum speed; above the limit the command shrinks like limit / force
        # (a force twice the limit halves it). force is the value of the previous step.
        scale = torch.clamp(force_limit / force.clamp(min=1e-6), max=1.0)
        action[live, 0] = -scale[live]
        return action
    if controller == "stall":
        # Push until a stall is declared, then retreat fully for stall_retreat_steps and
        # hold still (abandon the target; the episode ends by timeout).
        action[live & ~stalled, 0] = -1.0
        action[live & stalled & (stall_age < stall_retreat_steps), 0] = 1.0
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


def _stall_update(
    *,
    stalled: torch.Tensor,
    hist: torch.Tensor,
    hist_count: torch.Tensor,
    live: torch.Tensor,
    force: torch.Tensor,
    stall_force: float,
    stall_eps: float,
    stall_window: int,
) -> None:
    """Declare a stall: force above stall_force and progress up by < stall_eps in the window.

    hist is a ring buffer of the target progress recorded on every step since first contact
    (window + 1 slots). With hist_count entries recorded, entry hist_count - 1 is the newest
    and entry hist_count - 1 - window sits in slot hist_count % (window + 1), so a window of
    stall_window steps needs stall_window + 1 entries. Once stalled, always stalled.
    """
    size = stall_window + 1
    ready = live & ~stalled & (hist_count >= size)
    ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
        return
    newest = hist[ids, (hist_count[ids] - 1) % size]
    oldest = hist[ids, hist_count[ids] % size]
    hit = (force[ids] > stall_force) & ((newest - oldest) < stall_eps)
    stalled[ids[hit]] = True


def _stall_record(
    *,
    hist: torch.Tensor,
    hist_count: torch.Tensor,
    mask: torch.Tensor,
    progress: torch.Tensor,
) -> None:
    ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
        return
    hist[ids, hist_count[ids] % hist.shape[1]] = progress[ids]
    hist_count[ids] += 1


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
    extra: dict | None = None,
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
            "open_loop"
            if controller in {"settle", "straight"}
            else "force_feedback"
            if controller in {"forcelim", "stall"}
            else "contact_triggered"
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
        **(extra or {}),
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
    settle_steps: int = 0,
    force_limit: float = 5.0,
    stall_force: float = 2.0,
    stall_eps: float = 0.001,
    stall_window: int = 20,
    stall_retreat_steps: int = 100,
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
    # Feedback controllers: previous-step contact force, and the stall detector's state.
    last_force = torch.zeros(env.num_envs, device=env.device)
    stalled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    stall_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    stall_hist = torch.zeros(env.num_envs, stall_window + 1, device=env.device)
    stall_hist_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Only used with a settling period, where success no longer ends the episode: the
    # step at which success was first reached (-1 before). From the next step on the push
    # command is zero while the contact point stays, until the tower is judged again
    # settle_steps later.
    success_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    parameter_columns = {
        "force_limit": force_limit,
        "stall_force": stall_force,
        "stall_eps": stall_eps,
        "stall_window": stall_window,
        "stall_retreat_steps": stall_retreat_steps,
    }
    rows: list[dict] = []

    def settle_extra(env_id: int, settled: bool) -> dict:
        extra = dict(parameter_columns)
        if settle_steps <= 0:
            return extra
        reached = int(success_step[env_id].item()) >= 0
        extra.update(
            {
                "settle_steps": settle_steps,
                "success_step": int(success_step[env_id].item()),
                "reached_success": reached,
                "safe_after_settle": settled
                and bool(cfg.success_block_extract(env)[env_id].item()),
                "damaged_after_success": reached
                and bool(cfg.tower_damage_signal(env)[env_id].item()),
            }
        )
        return extra

    for _ in range(max_steps + settle_steps):
        live = active & ~finished
        if not bool(live.any().item()):
            break
        if controller == "stall":
            _stall_update(
                stalled=stalled,
                hist=stall_hist,
                hist_count=stall_hist_count,
                live=live,
                force=last_force,
                stall_force=stall_force,
                stall_eps=stall_eps,
                stall_window=stall_window,
            )
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
            force=last_force,
            stalled=stalled,
            stall_age=stall_age,
            force_limit=force_limit,
            stall_retreat_steps=stall_retreat_steps,
        )
        if settle_steps > 0:
            # Stop pushing after success but keep the contact point (columns 1 and 2).
            action[success_step >= 0, 0] = 0.0
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

        last_force = force
        if controller == "stall":
            # Count only from first contact on; the age of a declared stall grows after
            # the action of this step was chosen.
            _stall_record(
                hist=stall_hist,
                hist_count=stall_hist_count,
                mask=live & contact_seen,
                progress=progress,
            )
            stall_age[live & stalled] += 1

        ended = dones
        settled = torch.zeros_like(live)
        expired = torch.zeros_like(live)
        if settle_steps > 0:
            # A success after the regular step cap would not have counted without the
            # extension, so it does not count here either.
            reached_now = (
                cfg.success_block_extract(env)
                & live
                & (success_step < 0)
                & (steps <= max_steps)
            )
            success_step[reached_now] = steps[reached_now].long()
            elapsed = steps.long() - success_step
            settled = live & (success_step >= 0) & (elapsed >= settle_steps) & ~ended
            # The regular time limit, which the extended episode no longer enforces.
            expired = live & (success_step < 0) & (steps >= max_steps) & ~ended

        first_done = torch.nonzero(
            (ended | settled | expired) & live, as_tuple=False
        ).squeeze(-1)
        for env_id in first_done.tolist():
            is_settled = bool(settled[env_id].item())
            extra = settle_extra(env_id, is_settled)
            if is_settled:
                extra["reason"] = "settled"
            elif bool(expired[env_id].item()):
                extra["reason"] = "timeout"
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
                    extra=extra,
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
            extra=settle_extra(env_id, settled=False),
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
    settle_steps: int = 0,
    force_limit: float = 5.0,
    stall_force: float = 2.0,
    stall_eps: float = 0.001,
    stall_window: int = 20,
    stall_retreat_steps: int = 100,
) -> list[dict]:
    configure_evaluation_case(cfg, target, missing_level)
    vector_size = min(num_envs, episodes_per_seed)
    env_cfg = cfg.jenga_env_cfg()
    env_cfg.scene.num_envs = vector_size
    env_cfg.auto_reset = False
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.commands["target_block"].force_target_name = target
    if settle_steps > 0:
        # Success must not end the episode, or there is nothing left to observe. The
        # loop detects success itself and judges the tower again after settle_steps.
        # Damage stays a termination, so a collapse during settling is still recorded.
        env_cfg.terminations.pop("success", None)
        # Room for a success on the last regular step to finish settling. The loop
        # enforces the regular limit itself, so the environment's own limit must never
        # fire first; one spare step covers rounding in the step count.
        step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
        env_cfg.episode_length_s += (settle_steps + 1) * step_dt
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
                        settle_steps=settle_steps,
                        force_limit=force_limit,
                        stall_force=stall_force,
                        stall_eps=stall_eps,
                        stall_window=stall_window,
                        stall_retreat_steps=stall_retreat_steps,
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
    parser.add_argument(
        "--controllers",
        default="settle,straight,pulsed,tap",
        help=f"Comma-separated, from {','.join(CONTROLLERS)}. The default is the original "
        "four; forcelim and stall must be requested.",
    )
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
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Control steps (100 Hz, so 200 = 2 s) to keep simulating after success "
        "before judging the tower again. The push stops at success and the contact "
        "point is held. success_rate then means safe after settling; "
        "reached_success_rate is the rate at the moment of success. 0 keeps the regular "
        "protocol.",
    )
    parser.add_argument(
        "--force-limit",
        type=float,
        default=5.0,
        help="forcelim: contact force in N above which the push command is scaled by "
        "limit/force. Default 5.0, a free parameter with no counterpart in the task "
        "configuration.",
    )
    parser.add_argument(
        "--stall-force",
        type=float,
        default=2.0,
        help="stall: contact force in N that must be exceeded for a stall (default 2.0, "
        "the threshold of the stuck-contact reward term).",
    )
    parser.add_argument(
        "--stall-eps",
        type=float,
        default=0.001,
        help="stall: minimum target progress in m over the window; less counts as no "
        "progress. Default 0.001. This is a displacement over the window, not the reward "
        "term's 0.002 m/s speed test, and at the default it is the looser of the two.",
    )
    parser.add_argument(
        "--stall-window",
        type=int,
        default=20,
        help="stall: window in control steps, counted from first contact (default 20, "
        "the grace period of the stuck-contact reward term).",
    )
    parser.add_argument(
        "--stall-retreat-steps",
        type=int,
        default=100,
        help="stall: steps of full retreat after a stall is declared; the controller "
        "then holds still. Default 100 (1 s).",
    )
    parser.add_argument(
        "--friction-range",
        help="LO,HI: sliding-friction range sampled per block at every reset, replacing "
        "the configured range. Set before the environment is created; the value is "
        "written to friction_sliding_min/max.",
    )
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
    if args.settle_steps < 0:
        parser.error("settle-steps must not be negative")
    if args.force_limit <= 0 or args.stall_force < 0 or args.stall_eps < 0:
        parser.error("force-limit must be positive; stall-force and stall-eps non-negative")
    if args.stall_window <= 0 or args.stall_retreat_steps < 0:
        parser.error("stall-window must be positive; stall-retreat-steps non-negative")
    friction_range = None
    if args.friction_range is not None:
        try:
            friction_range = parse_float_range(args.friction_range)
        except ValueError as exc:
            parser.error(str(exc))

    cfg.apply_low_level_stage("target")
    cfg.SUCCESS_CURRICULUM_START = cfg.SUCCESS_CURRICULUM_END
    cfg.YAW_CURRICULUM_START = 0.0
    cfg.YAW_CURRICULUM_END = 0.0
    if friction_range is not None:
        # randomize_block_physics reads this module constant at every reset, so setting
        # it before the environment is created is enough.
        cfg.RESET_FRICTION_SLIDING_RANGE = friction_range
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
                    settle_steps=args.settle_steps,
                    force_limit=args.force_limit,
                    stall_force=args.stall_force,
                    stall_eps=args.stall_eps,
                    stall_window=args.stall_window,
                    stall_retreat_steps=args.stall_retreat_steps,
                )
                summary = summarize_episode_rows(rows)
                append_rows(args.episodes_csv, rows)
                append_rows(args.summary_csv, [summary])
                print(
                    f"{controller:>8} {target:>5} missing={missing_level} "
                    f"n={summary['episodes']} success={summary['success_rate']:.3f} "
                    f"damage={summary['tower_damage_rate']:.3f} "
                    f"progress={summary['progress_max_mean']:.4f} "
                    f"steps={summary['episode_length_mean']:.1f}"
                    + (
                        f" reached={summary['reached_success_rate']:.3f} "
                        f"damaged_after={summary['damaged_after_success_rate']:.3f}"
                        if args.settle_steps > 0
                        else ""
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
