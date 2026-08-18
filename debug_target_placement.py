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


def _fmt(values) -> str:
    return "(" + ",".join(f"{float(value):.5f}" for value in values) + ")"


def _configure_debug_stage(cfg) -> None:
    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
    cfg.RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
    cfg.RANDOM_TARGET_BLOCK_START_PROBABILITY = 1.0
    cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
    cfg.RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 10**12
    cfg.YAW_CURRICULUM_START = cfg.YAW_CURRICULUM_END


def _expected_tip_block(cfg, cmd, env):
    import torch

    selected = cmd.selected_block_idx
    features = cmd._target_features[selected]
    layer = torch.round(features[:, 0] * max(cfg.LAYERS - 1, 1) + 1).long()
    face_y = cmd.selected_contact_face_y()

    expected = torch.zeros(env.num_envs, 3, device=env.device)
    expected[:, 1] = face_y + torch.sign(face_y) * cfg.HOOK_APPROACH_GAP
    expected[:, 2] = torch.where(
        layer == 1,
        torch.full_like(face_y, cfg.HOOK_BOTTOM_LAYER_Z_LIFT),
        torch.zeros_like(face_y),
    )
    return expected


def _inspect_target(target_name: str, args) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import mjlab_jenga.jenga_mjenv_cfg as cfg

    _configure_debug_stage(cfg)
    env_cfg = cfg.jenga_env_cfg(play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["target_block"].force_target_name = target_name
    env_cfg.commands["target_block"].debug_target_reset = args.debug_reset

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    try:
        env.reset(seed=args.seed)
        for _ in range(args.steps):
            zero_action = torch.zeros(
                env.action_space.shape,
                device=env.device,
                dtype=torch.float32,
            )
            env.step(zero_action)

        cmd = env.command_manager.get_term("target_block")
        joints = cfg.hook_joint_pos_ordered(env)
        target_home = cmd.selected_hook_home()
        tip_block = cfg.hook_tip_pos_in_block_frame(env)
        tip_task = cfg.hook_tip_pos_in_task_frame(env)
        expected = _expected_tip_block(cfg, cmd, env)
        error = tip_block - expected
        target_idx = int(cmd.selected_block_idx[0].item())

        print(
            "DEBUG_TARGET_GEOM",
            f"target={cmd._all_names[target_idx]}",
            f"envs={env.num_envs}",
            f"steps={args.steps}",
            f"tip_block_mean={_fmt(tip_block.mean(dim=0).detach().cpu())}",
            f"expected_tip_block_mean={_fmt(expected.mean(dim=0).detach().cpu())}",
            f"err_mean={_fmt(error.mean(dim=0).detach().cpu())}",
            f"err_abs_max={_fmt(error.abs().amax(dim=0).detach().cpu())}",
            f"tip_task_mean={_fmt(tip_task.mean(dim=0).detach().cpu())}",
            f"joints_mean={_fmt(joints.mean(dim=0).detach().cpu())}",
            f"home_mean={_fmt(target_home.mean(dim=0).detach().cpu())}",
            flush=True,
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print numeric hook-tip placement errors for forced target blocks."
    )
    parser.add_argument("targets", nargs="+", help="Block names, for example b2_1 b1_1.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Zero checks the reset pose. Larger values also include controller motion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-reset",
        action="store_true",
        help="Also print DEBUG_TARGET_RESET lines during env reset.",
    )
    args = parser.parse_args()

    _maybe_force_cpu()
    for target_name in args.targets:
        _inspect_target(target_name, args)


if __name__ == "__main__":
    main()
