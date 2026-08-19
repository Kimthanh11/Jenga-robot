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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a checkpoint with one forced random target block."
    )
    parser.add_argument("--checkpoint", default="model_last.pt")
    parser.add_argument(
        "--agent",
        choices=("zero", "random", "trained"),
        default="zero",
        help="Use zero to inspect reset/teleport positions without a policy.",
    )
    parser.add_argument("--viewer", default="native")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--target",
        default=None,
        help="Optional block name to force, for example b1_1 or b2_3.",
    )
    parser.add_argument(
        "--debug-target",
        action="store_true",
        help="Print target reset geometry for the first few envs.",
    )
    parser.add_argument(
        "--force-yaw",
        action="store_true",
        help="Enable the full yaw curriculum during play.",
    )
    args = parser.parse_args()

    if args.agent == "trained" and not args.checkpoint:
        parser.error("--checkpoint is required when --agent trained is used")

    _maybe_force_cpu()

    # Importing the package registers the normal training task. We then patch only
    # this process and register a separate play-only task for teleport inspection.
    import mjlab_jenga.jenga_mjenv_cfg as cfg
    from mjlab.scripts.play import PlayConfig, run_play
    from mjlab.tasks.registry import register_mjlab_task

    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
    cfg.RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
    cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
    cfg.RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 10**12
    if args.force_yaw:
        cfg.YAW_CURRICULUM_START = cfg.YAW_CURRICULUM_END

    env_cfg = cfg.jenga_env_cfg()
    play_env_cfg = cfg.jenga_env_cfg(play=True)
    env_cfg.commands["target_block"].debug_target_reset = args.debug_target
    play_env_cfg.commands["target_block"].debug_target_reset = args.debug_target
    if args.target is not None:
        env_cfg.commands["target_block"].force_target_name = args.target
        play_env_cfg.commands["target_block"].force_target_name = args.target

    task_id = "Mjlab-Jenga-ForcedRandomTargetPlay"
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=cfg.jenga_ppo_runner_cfg(),
    )

    run_play(
        task_id,
        PlayConfig(
            agent=args.agent,
            checkpoint_file=args.checkpoint if args.agent == "trained" else None,
            viewer=args.viewer,
            num_envs=args.num_envs,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
