from mjlab.tasks.registry import register_mjlab_task

from mjlab_jenga.jenga_mjenv_cfg import jenga_env_cfg, jenga_ppo_runner_cfg
from mjlab_jenga import jenga_incomplete as _incomplete
from mjlab_jenga import jenga_navigate_pull as _navpull
register_mjlab_task(
    task_id="Mjlab-Jenga",
    env_cfg=jenga_env_cfg(),
    play_env_cfg=jenga_env_cfg(play=True),
    rl_cfg=jenga_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Mjlab-Jenga-Incomplete",
    env_cfg=_incomplete.jenga_env_cfg(),
    play_env_cfg=_incomplete.jenga_env_cfg(play=True),
    rl_cfg=_incomplete.jenga_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Mjlab-Jenga-Navpull",
    env_cfg=_navpull.jenga_env_cfg(),
    play_env_cfg=_navpull.jenga_env_cfg(play=True),
    rl_cfg=_navpull.jenga_ppo_runner_cfg(),
)

# Random-block worker: each episode randomly picks a block, teleports the hook in
# front of it, policy learns to push it out. Built on Boris's main config + a
# JengaPushCommand. Launch: train.py / play.py Mjlab-Jenga-RandomBlock
# from mjlab_jenga import jenga_random_block_cfg as _randblock  # noqa: E402

# register_mjlab_task(
#     task_id="Mjlab-Jenga-RandomBlock",
#     env_cfg=_randblock.jenga_env_cfg(),
#     play_env_cfg=_randblock.jenga_env_cfg(play=True),
#     rl_cfg=_randblock.jenga_ppo_runner_cfg(),
# )

# Phase 1 baseline (Fazeli-style, fixed b6_1) registered here on the cluster for
# comparison runs. jenga_phase1_cfg.py is present locally (untracked upstream).
from mjlab_jenga import jenga_phase1_cfg as _phase1  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-Phase1",
    env_cfg=_phase1.jenga_env_cfg(),
    play_env_cfg=_phase1.jenga_env_cfg(play=True),
    rl_cfg=_phase1.jenga_ppo_runner_cfg(),
)
