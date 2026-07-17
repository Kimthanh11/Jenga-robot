from mjlab.tasks.registry import register_mjlab_task

from mjlab_jenga.jenga_mjenv_cfg import jenga_env_cfg, jenga_ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Jenga",
    env_cfg=jenga_env_cfg(),
    play_env_cfg=jenga_env_cfg(play=True),
    rl_cfg=jenga_ppo_runner_cfg(),
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
