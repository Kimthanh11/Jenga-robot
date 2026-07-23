from mjlab.tasks.registry import register_mjlab_task

from mjlab_jenga.jenga_mjenv_cfg import jenga_env_cfg, jenga_ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Jenga",
    env_cfg=jenga_env_cfg(),
    play_env_cfg=jenga_env_cfg(play=True),
    rl_cfg=jenga_ppo_runner_cfg(),
)

# Phase 1 (Fazeli-style reward, obs=11, 3-action hook). Self-contained frozen
# snapshot of commit 22431a7 so its checkpoints replay regardless of the current
# jenga_mjenv_cfg. Launch: play.py Mjlab-Jenga-Phase1
from mjlab_jenga import jenga_phase1_cfg as _phase1  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-Phase1",
    env_cfg=_phase1.jenga_env_cfg(),
    play_env_cfg=_phase1.jenga_env_cfg(play=True),
    rl_cfg=_phase1.jenga_ppo_runner_cfg(),
)

# Random-block worker: each episode randomly picks a block, teleports the hook in
# front of it, policy learns to push it out. Built on Boris's main config + a
# JengaPushCommand. Launch: train.py / play.py Mjlab-Jenga-RandomBlock
from mjlab_jenga import jenga_random_block_cfg as _randblock  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-RandomBlock",
    env_cfg=_randblock.jenga_env_cfg(),
    play_env_cfg=_randblock.jenga_env_cfg(play=True),
    rl_cfg=_randblock.jenga_ppo_runner_cfg(),
)

# Thanh's incomplete tower (b4_1, b4_3, b5_2 removed), fixed target b6_1. Same task id
# as on the cluster so play_latest_incomplete.py keeps working.
from mjlab_jenga import jenga_incomplete as _incomplete  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-Incomplete",
    env_cfg=_incomplete.jenga_env_cfg(),
    play_env_cfg=_incomplete.jenga_env_cfg(play=True),
    rl_cfg=_incomplete.jenga_ppo_runner_cfg(),
)

# Merge: incomplete tower + random block assignment. Each episode picks a random
# selectable block of the incomplete tower and pushes it out.
# Launch: train.py / play.py Mjlab-Jenga-IncompleteRandom
from mjlab_jenga import jenga_incomplete_random_cfg as _incrand  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-IncompleteRandom",
    env_cfg=_incrand.jenga_env_cfg(),
    play_env_cfg=_incrand.jenga_env_cfg(play=True),
    rl_cfg=_incrand.jenga_ppo_runner_cfg(),
)
