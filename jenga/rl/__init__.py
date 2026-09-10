from mjlab.tasks.registry import register_mjlab_task

from rl.env_cfgs import jenga_env_cfg
from rl.rl_cfg import jenga_ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Jenga",
    env_cfg=jenga_env_cfg(),
    play_env_cfg=jenga_env_cfg(play=True),
    rl_cfg=jenga_ppo_runner_cfg(),
)

