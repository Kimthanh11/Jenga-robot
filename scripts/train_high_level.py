"""Train the serial masked-categorical high-level Jenga policy."""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mjlab_jenga.jenga_mjenv_cfg as jenga_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from mjlab_jenga.high_level_ppo_cfg import (
    high_level_ppo_runner_cfg,
    high_level_ppo_train_dict,
)
from mjlab_jenga.high_level_runtime import build_serial_high_level_env
from mjlab_jenga.high_level_vecenv import HighLevelRslRlVecEnv
from mjlab_jenga.jenga_mjenv_cfg import jenga_env_cfg, jenga_ppo_runner_cfg


def main(
    *,
    checkpoint: Path,
    device: str,
    seed: int,
    iterations: int,
    settle_steps: int,
    max_steps_per_block: int,
    log_root: Path,
) -> None:
    """Load the frozen low-level actor and optimize the high-level selector."""
    jenga_cfg.apply_low_level_stage("target")
    jenga_cfg.SUCCESS_CURRICULUM_START = jenga_cfg.SUCCESS_CURRICULUM_END
    jenga_cfg.YAW_CURRICULUM_START = 0.0
    jenga_cfg.YAW_CURRICULUM_END = 0.0
    jenga_cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
    jenga_cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0

    env_cfg = jenga_env_cfg(play=True)
    env_cfg.auto_reset = False
    env_cfg.scene.num_envs = 1
    # Evaluate these conditions manually so successful moves can share one tower.
    env_cfg.terminations.pop("success", None)
    env_cfg.terminations.pop("tower_damage", None)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.seed(seed)
    low_level_cfg = jenga_ppo_runner_cfg()
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=low_level_cfg.clip_actions)
    low_level_train_cfg = asdict(low_level_cfg)
    # RSL-RL 4.x's logger expects this optional extension key to exist even when
    # Random Network Distillation is disabled.
    low_level_train_cfg["algorithm"].setdefault("rnd_cfg", None)
    low_level_runner = MjlabOnPolicyRunner(
        wrapped_env,
        low_level_train_cfg,
        device=device,
    )
    low_level_runner.load(
        path=str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    low_level_policy = low_level_runner.get_inference_policy(device=device)

    high_level_env = build_serial_high_level_env(
        cfg=jenga_cfg,
        env=env,
        wrapped_env=wrapped_env,
        policy=low_level_policy,
        settle_steps=settle_steps,
        max_steps_per_block=max_steps_per_block,
    )
    high_level_vecenv = HighLevelRslRlVecEnv(high_level_env, device=device)
    high_level_vecenv.seed(seed)
    num_block_actions = len(high_level_env.block_names_by_action)
    high_level_cfg = high_level_ppo_runner_cfg(num_block_actions)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / high_level_cfg.experiment_name / timestamp
    runner = OnPolicyRunner(
        high_level_vecenv,
        high_level_ppo_train_dict(num_block_actions),
        log_dir=str(log_dir),
        device=device,
    )

    try:
        runner.learn(num_learning_iterations=iterations)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the serial high-level block-selection PPO pilot."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--max-steps-per-block", type=int, default=2000)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.settle_steps < 0:
        parser.error("--settle-steps must not be negative")
    if args.max_steps_per_block <= 0:
        parser.error("--max-steps-per-block must be positive")

    main(
        checkpoint=args.checkpoint,
        device=args.device,
        seed=args.seed,
        iterations=args.iterations,
        settle_steps=args.settle_steps,
        max_steps_per_block=args.max_steps_per_block,
        log_root=args.log_root,
    )
