from __future__ import annotations

import argparse
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


def _configure_debug_stage(cfg) -> None:
    cfg.apply_low_level_stage("fixed")
    cfg.YAW_CURRICULUM_START = 0.0
    cfg.YAW_CURRICULUM_END = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check push, stop, retreat, and hook contact without training."
    )
    parser.add_argument("--target", default="b6_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--push-steps", type=int, default=180)
    parser.add_argument("--stop-steps", type=int, default=12)
    parser.add_argument("--retreat-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _maybe_force_cpu()

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import mjlab_jenga.jenga_mjenv_cfg as cfg

    _configure_debug_stage(cfg)
    env_cfg = cfg.jenga_env_cfg(play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["target_block"].force_target_name = args.target

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    try:
        env.reset(seed=args.seed)
        action = torch.zeros(env.action_space.shape, device=env.device)
        max_force = 0.0
        max_progress = float("-inf")
        contact_steps = 0
        total_reward = 0.0

        action[:, 0] = -1.0
        for _ in range(args.push_steps):
            _, reward, _, _, _ = env.step(action)
            total_reward += float(reward.sum().item())
            force = cfg.hook_contact_force_norm(env)
            progress = cfg.block_progress(env)
            max_force = max(max_force, float(force.max().item()))
            max_progress = max(max_progress, float(progress.max().item()))
            contact_steps += int(cfg.hook_contact_found(env).sum().item())
        push_velocity = float(cfg.push_velocity_target(env).item())

        action.zero_()
        for _ in range(args.stop_steps):
            _, reward, _, _, _ = env.step(action)
            total_reward += float(reward.sum().item())
        stop_velocity = float(cfg.push_velocity_target(env).item())

        action[:, 0] = 1.0
        for _ in range(args.retreat_steps):
            _, reward, _, _, _ = env.step(action)
            total_reward += float(reward.sum().item())
        retreat_velocity = float(cfg.push_velocity_target(env).item())

        push_ok = push_velocity < -0.02
        stop_ok = abs(stop_velocity) < 1.0e-6
        retreat_ok = retreat_velocity > 0.02
        contact_ok = contact_steps > 0 and max_force > 0.0
        passed = push_ok and stop_ok and retreat_ok and contact_ok
        print(
            "DEBUG_PUSH_CONTROL",
            f"target={args.target}",
            f"push_velocity={push_velocity:.5f}",
            f"stop_velocity={stop_velocity:.5f}",
            f"retreat_velocity={retreat_velocity:.5f}",
            f"contact_steps={contact_steps}",
            f"max_force={max_force:.3f}",
            f"max_progress={max_progress:.5f}",
            f"total_reward={total_reward:.4f}",
            f"passed={passed}",
            flush=True,
        )
        if not passed:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
