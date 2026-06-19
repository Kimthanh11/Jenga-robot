from mjlab.tasks.registry import register_mjlab_task

from mjlab_jenga.jenga_mjenv_cfg import jenga_env_cfg, jenga_ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Jenga",
    env_cfg=jenga_env_cfg(),
    play_env_cfg=jenga_env_cfg(play=True),
    rl_cfg=jenga_ppo_runner_cfg(),
)
