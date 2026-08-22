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
    parser.add_argument("--run-suffix", default=None)
    args = parser.parse_args()

    if (args.load_run is None) != (args.load_checkpoint is None):
        parser.error("--load-run and --load-checkpoint must be given together")

    cfg.apply_low_level_stage(args.stage)
    if args.success_curriculum_steps is not None:
        cfg.SUCCESS_CURRICULUM_STEPS = args.success_curriculum_steps
    env_cfg = cfg.jenga_env_cfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = cfg.jenga_ppo_runner_cfg()
    agent_cfg.max_iterations = args.iterations
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef

    run_name = f"low_level_{args.stage}"
    if args.entropy_coef is not None:
        run_name += f"_ent{args.entropy_coef:g}"
    if args.success_curriculum_steps is not None:
        run_name += f"_cur{args.success_curriculum_steps}"
    if args.run_suffix:
        run_name += f"_{args.run_suffix}"
    agent_cfg.run_name = run_name

    print(
        f"stage={args.stage} run_name={run_name} "
        f"entropy_coef={agent_cfg.algorithm.entropy_coef} "
        f"success_curriculum_steps={cfg.SUCCESS_CURRICULUM_STEPS} "
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
        play_env_cfg=cfg.jenga_env_cfg(play=True),
        rl_cfg=agent_cfg,
    )

    launch_training(task_id, TrainConfig(env=env_cfg, agent=agent_cfg))


if __name__ == "__main__":
    main()
