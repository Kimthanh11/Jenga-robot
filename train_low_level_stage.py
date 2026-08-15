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
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=768)
    args = parser.parse_args()

    cfg.apply_low_level_stage(args.stage)
    env_cfg = cfg.jenga_env_cfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = cfg.jenga_ppo_runner_cfg()
    agent_cfg.max_iterations = args.iterations
    agent_cfg.run_name = f"low_level_{args.stage}"

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
