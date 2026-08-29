"""Train one low-level extraction stage with PPO.

Stages are selected by name and configure target sampling and missing-block
randomization via cfg.apply_low_level_stage(): "fixed" and "target" train on the
complete tower, "missing1" through "missing3" progressively introduce towers with
blocks already removed.

Command-line overrides exist for the settings that are varied between runs -- entropy
coefficient, curriculum length, target set, yaw limit and yaw freezing -- so that a
comparison does not require editing the configuration module. The resolved command line
is echoed by the accompanying sbatch script.
"""

from __future__ import annotations

import argparse

import mjlab_jenga.jenga_mjenv_cfg as cfg
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import register_mjlab_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a staged Jenga low-level task.")
    parser.add_argument(
        "stage",
        choices=("fixed", "target", "missing1", "missing2", "missing3", "full"),
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--load-run", default=None)
    parser.add_argument("--load-checkpoint", default=None)
    # Overrides for A/B runs. Defaults stay in the config so a plain invocation is
    # always the committed setting; anything set here is echoed into the run name so
    # two concurrent runs stay distinguishable in the log directory.
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Once the task is saturated the entropy bonus is the only gradient left, "
        "so std drifts up to its cap. Lower this to test whether that hurts.",
    )
    parser.add_argument(
        "--success-curriculum-steps",
        type=int,
        default=None,
        help="Steps over which the success distance ramps from 2 cm to 11.25 cm.",
    )
    parser.add_argument(
        "--freeze-yaw",
        action="store_true",
        help="Zero the yaw curriculum so the yaw target stays at the per-env home. "
        "The action dimension is kept, so the network shape stays comparable -- this "
        "isolates whether yaw MOTION helps, not whether the input helps.",
    )
    parser.add_argument(
        "--base-relative-shift",
        action="store_true",
        help="Measure horizontal block displacement against the tower's base rather "
        "than the nominal spawn pose, so a rigid slide of the whole tower is not "
        "counted as damage.",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target blocks, overriding RANDOM_TARGET_BLOCK_NAMES. "
        "Repeating a name raises its sampling share, which is how a target the policy "
        "has written off can be given a concentrated signal without dropping the rest.",
    )
    parser.add_argument(
        "--yaw-limit",
        type=float,
        default=None,
        help="Override YAW_TARGET_LIMIT (rad). Beyond 0.56 rad (32.1 deg) the 80 mm "
        "hook shaft no longer fits into the 51 mm slot the target block vacates.",
    )
    parser.add_argument(
        "--abort",
        action="store_true",
        help="Train the abort variant: adds a fifth action that ends the episode when "
        "held over a threshold, so a block that cannot be extracted safely can be "
        "abandoned. Warm-start from a base checkpoint widened by "
        "widen_checkpoint_for_abort.py.",
    )
    parser.add_argument("--run-suffix", default=None)
    args = parser.parse_args()

    if (args.load_run is None) != (args.load_checkpoint is None):
        parser.error("--load-run and --load-checkpoint must be given together")

    cfg.apply_low_level_stage(args.stage)
    if args.success_curriculum_steps is not None:
        cfg.SUCCESS_CURRICULUM_STEPS = args.success_curriculum_steps
    if args.freeze_yaw:
        cfg.YAW_CURRICULUM_START = 0.0
        cfg.YAW_CURRICULUM_END = 0.0
    if args.yaw_limit is not None:
        cfg.YAW_TARGET_LIMIT = args.yaw_limit
    cfg.TOWER_SHIFT_RELATIVE_TO_BASE = args.base_relative_shift
    if args.targets:
        cfg.RANDOM_TARGET_BLOCK_NAMES = tuple(
            t.strip() for t in args.targets.split(",") if t.strip()
        )
    # The abort module builds on cfg and reads its globals, so every override above
    # still applies; only the environment and runner factories come from elsewhere.
    task = cfg
    if args.abort:
        import mjlab_jenga.jenga_abort_cfg as abort_cfg

        task = abort_cfg

    env_cfg = task.jenga_env_cfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = task.jenga_ppo_runner_cfg()
    agent_cfg.max_iterations = args.iterations
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef

    run_name = f"low_level_{args.stage}"
    if args.entropy_coef is not None:
        run_name += f"_ent{args.entropy_coef:g}"
    if args.success_curriculum_steps is not None:
        run_name += f"_cur{args.success_curriculum_steps}"
    if args.abort:
        run_name += "_abort"
    if args.freeze_yaw:
        run_name += "_noyaw"
    if args.yaw_limit is not None:
        run_name += f"_yaw{args.yaw_limit:g}"
    if args.base_relative_shift:
        run_name += "_baseshift"
    if args.targets:
        run_name += "_tgt%d" % len(cfg.RANDOM_TARGET_BLOCK_NAMES)
    if args.run_suffix:
        run_name += f"_{args.run_suffix}"
    agent_cfg.run_name = run_name

    print(
        f"stage={args.stage} run_name={run_name} "
        f"entropy_coef={agent_cfg.algorithm.entropy_coef} "
        f"success_curriculum_steps={cfg.SUCCESS_CURRICULUM_STEPS} "
        f"yaw_curriculum=({cfg.YAW_CURRICULUM_START},{cfg.YAW_CURRICULUM_END}) "
        f"base_relative_shift={cfg.TOWER_SHIFT_RELATIVE_TO_BASE} "
        f"targets={cfg.RANDOM_TARGET_BLOCK_NAMES} "
        f"yaw_target_limit={cfg.YAW_TARGET_LIMIT} "
        f"num_envs={args.num_envs} iterations={args.iterations}",
        flush=True,
    )
    if args.load_run is not None:
        agent_cfg.resume = True
        agent_cfg.load_run = args.load_run
        agent_cfg.load_checkpoint = args.load_checkpoint

    task_id = f"Mjlab-Jenga-LowLevel-{args.stage}"
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=task.jenga_env_cfg(play=True),
        rl_cfg=agent_cfg,
    )

    launch_training(task_id, TrainConfig(env=env_cfg, agent=agent_cfg))


if __name__ == "__main__":
    main()
