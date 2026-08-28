"""Replay a checkpoint or a scripted agent in an interactive viewer.

Each reset samples a new target block unless --target pins one. The --agent choice
selects between the trained policy, random actions, and zero actions; the last is useful
for inspecting reset geometry and the settling transient without a policy involved.

The yaw configuration must match the one the checkpoint was trained under. Neither flag
given runs the yaw curriculum at its start value, which is neither the frozen nor the
full setting, so a policy trained with --freeze-yaw has to be replayed with it as well.
The resolved yaw setting is printed at startup.

Use --viewer viser on a headless machine; the native viewer requires a display. With
--video the run is additionally recorded to an mp4 under the run's log directory.
"""

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
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render to an mp4 instead of an interactive window. Playback is bound by "
        "the physics, not the display, so this is the usable route on a machine "
        "without a CUDA GPU: render on the cluster and download the file.",
    )
    parser.add_argument("--video-length", type=int, default=1000)
    parser.add_argument("--camera", default=None)
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
    parser.add_argument(
        "--missing",
        type=int,
        default=0,
        help="Number of blocks removed before the episode starts. The default of 0 "
        "leaves the tower intact, which is not the condition a policy trained on the "
        "missing-block stages was evaluated under.",
    )
    parser.add_argument(
        "--freeze-yaw",
        action="store_true",
        help="Hold the hook at its home yaw, matching train_low_level_stage.py "
        "--freeze-yaw. Without this the play session still runs the curriculum at its "
        "start value of 0.10, so a policy trained with the yaw frozen would be watched "
        "under conditions it never saw.",
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

    cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
    cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0
    if args.missing > 0:
        cfg.FORCED_MISSING_BLOCK_COUNT = args.missing
        cfg.MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = -1
        cfg.MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 1
        cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 1.0
        cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 1.0
    cfg.RANDOM_TARGET_BLOCK_BEGIN_STEP = -1
    cfg.RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
    cfg.RANDOM_TARGET_BLOCK_START_PROBABILITY = 1.0
    cfg.RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
    cfg.RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 0.0
    cfg.RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 0.0
    if args.force_yaw and args.freeze_yaw:
        parser.error("--force-yaw and --freeze-yaw are mutually exclusive")
    if args.force_yaw:
        cfg.YAW_CURRICULUM_START = cfg.YAW_CURRICULUM_END
    if args.freeze_yaw:
        cfg.YAW_CURRICULUM_START = 0.0
        cfg.YAW_CURRICULUM_END = 0.0
    print(
        f"yaw: curriculum=({cfg.YAW_CURRICULUM_START}, {cfg.YAW_CURRICULUM_END}) "
        f"limit={cfg.YAW_TARGET_LIMIT} rad | missing blocks: {args.missing}",
        flush=True,
    )

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
            video=args.video,
            video_length=args.video_length,
            camera=args.camera,
        ),
    )


if __name__ == "__main__":
    main()
