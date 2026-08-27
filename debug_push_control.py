"""Open-loop extraction test for a single Jenga target block.

Drives the hook with a constant push command and runs until the episode terminates,
reporting which terminal condition was reached. This serves as a feasibility oracle:
a block that a straight scripted push cannot extract is unlikely to be a viable
reinforcement learning target, and establishing that costs seconds rather than GPU
hours.

Terminal conditions, reported as `reason`:

    success     the target reached the success distance with the tower stable
    damage      a non-target block exceeded a tower_damage threshold
    stalled     no significant progress for STALL_PATIENCE steps
    step_cap    the hard step limit was reached without a terminal condition

The distinction between `stalled` and `step_cap` matters: without it, "the block is
stuck" cannot be told apart from "the test budget was too small". Extraction requires
112.5 mm of travel, which at 100 Hz and 0.03 m/s is 375 contact steps plus roughly 73
steps to close the 20 mm approach gap. Targets with high slip need considerably more,
so a fixed budget is not a valid test.

Both the actuator force on hook_slide and the contact sensor force are recorded. The
two separate an actuator saturated at its stall force from a tip that is losing
contact; the contact force alone cannot distinguish them.
"""

from __future__ import annotations

import argparse
import math
import os


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


def _configure_debug_stage(cfg, friction: float | None, lock_yaw: bool) -> None:
    """Isolate the target block: no missing blocks, no yaw, optionally fixed friction."""
    cfg.apply_low_level_stage("fixed")
    cfg.YAW_CURRICULUM_START = 0.0
    cfg.YAW_CURRICULUM_END = 0.0
    if lock_yaw:
        # CurriculumYawAction sets target = MEASURED position + delta, so with delta 0
        # the servo error is always zero and the joint free-wheels under contact load.
        # Collapsing the clamp window pins the target to the per-env home yaw instead.
        cfg.YAW_TARGET_LIMIT = 0.0
    if friction is not None:
        cfg.RESET_FRICTION_SLIDING_RANGE = (friction, friction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether a target block can be extracted by a full push."
    )
    parser.add_argument("--target", default="b6_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Hard cap. A slip-free extraction needs ~450 steps; at the slip seen on "
        "b6_1 it needs ~950.",
    )
    parser.add_argument(
        "--stall-steps",
        type=int,
        default=150,
        help="Abort if best progress has not improved over this many steps.",
    )
    parser.add_argument(
        "--stall-eps",
        type=float,
        default=0.0002,
        help="Progress improvement (m) that counts as 'not stalled'.",
    )
    parser.add_argument(
        "--contact-x",
        type=float,
        default=0.0,
        help="Lateral contact point on the push face, in [-1, 1].",
    )
    parser.add_argument(
        "--contact-z",
        type=float,
        default=0.0,
        help="Vertical contact point on the push face, in [-1, 1].",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Pin the sliding friction instead of sampling it per reset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lock-yaw",
        action="store_true",
        help="Pin the yaw target to the home yaw instead of letting it track the "
        "measured joint position.",
    )
    parser.add_argument(
        "--trace",
        type=int,
        default=0,
        help="Print a per-step row every N steps. Shows whether the tip holds its "
        "contact point on the face or is pushed off it by the soft y/z servos.",
    )
    args = parser.parse_args()

    _maybe_force_cpu()

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import mjlab_jenga.jenga_mjenv_cfg as cfg

    _configure_debug_stage(cfg, args.friction, args.lock_yaw)
    env_cfg = cfg.jenga_env_cfg(play=True)
    env_cfg.scene.num_envs = 1
    # Keep the terminal state readable instead of resetting it away underneath us.
    env_cfg.auto_reset = False
    env_cfg.commands["target_block"].force_target_name = args.target

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    try:
        env.reset(seed=args.seed)

        hook = env.scene["hook"]
        slide_ids, _ = hook.find_joints(("hook_slide",), preserve_order=True)
        slide_dof = int(slide_ids[0])
        slide_ctrl = _resolve_slide_actuator(hook)

        action = torch.zeros(env.action_space.shape, device=env.device)
        action[:, 0] = -1.0                 # full push
        action[:, 1] = args.contact_x
        action[:, 2] = args.contact_z
        action[:, 3] = 0.0                  # yaw frozen

        success_distance = float(cfg.success_done_distance(env).item())
        step_dt = float(env.step_dt)

        best_progress = float("-inf")
        stall_ref = float("-inf")
        best_step = 0
        steps_to_contact = -1
        contact_steps = 0
        commanded_travel = 0.0
        max_contact_force = 0.0
        max_slide_actuator_force = 0.0
        max_ctrl_actuator_force = 0.0
        max_hook_speed = 0.0
        reason = "step_cap"
        steps = 0

        for step in range(1, args.max_steps + 1):
            steps = step
            _, _, terminated, truncated, _ = env.step(action)

            progress = float(cfg.block_progress(env).item())
            contact = float(cfg.hook_contact_found(env).item()) > 0.0
            contact_force = float(cfg.hook_contact_force_norm(env).item())
            slide_force = float(hook.data.qfrc_actuator[0, slide_dof].item())
            hook_speed = abs(float(hook.data.joint_vel[0, slide_dof].item()))
            if slide_ctrl is not None:
                max_ctrl_actuator_force = max(
                    max_ctrl_actuator_force,
                    abs(float(hook.data.actuator_force[0, slide_ctrl].item())),
                )

            if contact:
                if steps_to_contact < 0:
                    steps_to_contact = step
                contact_steps += 1
            if steps_to_contact >= 0:
                commanded_travel += abs(float(cfg.push_velocity_target(env).item())) * step_dt

            max_contact_force = max(max_contact_force, contact_force)
            max_slide_actuator_force = max(max_slide_actuator_force, abs(slide_force))
            max_hook_speed = max(max_hook_speed, hook_speed)

            if args.trace and step % args.trace == 0:
                joints = cfg.hook_joint_pos_ordered(env)[0]        # slide, y, z, yaw
                touch = env.action_manager.get_term("block_local_touch")
                target_yz = touch._processed_targets[0]
                tip_block = cfg.hook_tip_pos_in_block_frame(env)[0]
                # block_progress is drift-corrected against the same-layer neighbours.
                # Log the raw displacement too, so "block did not move" can be told
                # apart from "block and its reference moved together".
                cmd = env.command_manager.get_term("target_block")
                extraction = cmd.selected_extraction_w()[0]
                target_w = cmd.selected_target_pos_w()[0]
                ref_w = cmd.selected_ref_pos_w()[0]
                start_w = cmd._start_pos[cmd.selected_block_idx][0]
                raw_progress = float(torch.dot(target_w - start_w, extraction).item())
                ref_shift = float(
                    torch.dot(ref_w - cmd._start_pos[cmd._neighbor_idx[
                        cmd.selected_block_idx]][0].mean(dim=0), extraction).item()
                )
                print(
                    "TRACE",
                    f"step={step}",
                    f"progress={progress:.5f}",
                    f"raw_progress={raw_progress:.5f}",
                    f"ref_shift={ref_shift:.5f}",
                    f"tip_w=({cfg.hook_tip_pos(env)[0, 0].item():.5f},"
                    f"{cfg.hook_tip_pos(env)[0, 1].item():.5f},"
                    f"{cfg.hook_tip_pos(env)[0, 2].item():.5f})",
                    f"block_w=({target_w[0].item():.5f},{target_w[1].item():.5f},"
                    f"{target_w[2].item():.5f})",
                    f"slide={joints[0].item():.5f}",
                    f"y={joints[1].item():.5f}/{target_yz[0].item():.5f}",
                    f"z={joints[2].item():.5f}/{target_yz[1].item():.5f}",
                    f"yaw_deg={math.degrees(joints[3].item()):.3f}",
                    f"yaw_target_deg={math.degrees(env.action_manager.get_term('yaw')._processed_targets[0, 0].item()):.3f}",
                    f"yaw_home_deg={math.degrees(cmd.selected_hook_home()[0, 3].item()):.3f}",
                    f"tip_in_block=({tip_block[0].item():.5f},"
                    f"{tip_block[1].item():.5f},{tip_block[2].item():.5f})",
                    f"contact_f={contact_force:.3f}",
                    f"act_f={slide_force:.3f}",
                    f"hook_v={hook_speed:.5f}",
                    flush=True,
                )

            # Two separate quantities: `best_progress` is the true maximum, while
            # `stall_ref` is the level of the last SIGNIFICANT improvement. Folding them
            # into one lets sub-epsilon creep raise the bar every step, so the epsilon
            # never triggers and the stall fires on schedule regardless of motion.
            if progress > stall_ref + args.stall_eps:
                stall_ref = progress
                best_step = step
            best_progress = max(best_progress, progress)

            if bool(terminated.any().item()):
                reason = "damage" if bool(cfg.tower_damage(env).any().item()) else "success"
                break
            if bool(truncated.any().item()):
                reason = "timeout"
                break
            if steps_to_contact >= 0 and step - best_step >= args.stall_steps:
                reason = "stalled"
                break

        extracted = bool(cfg.target_extraction_reached(env).any().item())
        safe_success = bool(cfg.success_block_extract(env).any().item())
        damaged = bool(cfg.tower_damage(env).any().item())
        slip = best_progress / commanded_travel if commanded_travel > 1e-9 else float("nan")

        # passed now means "safely extracted", not "some contact happened".
        passed = safe_success

        print(
            "DEBUG_PUSH_CONTROL",
            f"target={args.target}",
            f"seed={args.seed}",
            f"friction={args.friction if args.friction is not None else 'sampled'}",
            f"lock_yaw={args.lock_yaw}",
            f"contact_point=({args.contact_x:.2f},{args.contact_z:.2f})",
            f"reason={reason}",
            f"steps={steps}",
            f"steps_to_contact={steps_to_contact}",
            f"contact_steps={contact_steps}",
            f"max_progress={best_progress:.5f}",
            f"success_distance={success_distance:.5f}",
            f"commanded_travel={commanded_travel:.5f}",
            f"slip_ratio={slip:.3f}",
            f"max_slide_actuator_force={max_slide_actuator_force:.3f}",
            f"max_ctrl_actuator_force={max_ctrl_actuator_force:.3f}",
            f"max_contact_force={max_contact_force:.3f}",
            f"max_hook_speed={max_hook_speed:.5f}",
            f"extracted={extracted}",
            f"tower_damage={damaged}",
            f"passed={passed}",
            flush=True,
        )
        if not passed:
            raise SystemExit(1)
    finally:
        env.close()


def _resolve_slide_actuator(hook) -> int | None:
    """Index of the hook_slide actuator in actuator_force, or None if unresolvable."""
    try:
        names = tuple(hook.actuator_names)
    except Exception:
        return None
    for wanted in ("hook_slide", "hook_x_vel"):
        if wanted in names:
            return names.index(wanted)
    return 0 if names else None


if __name__ == "__main__":
    main()
