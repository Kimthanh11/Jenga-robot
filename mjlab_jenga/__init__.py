from mjlab.tasks.registry import register_mjlab_task

from mjlab_jenga.jenga_mjenv_cfg import jenga_env_cfg, jenga_ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Jenga",
    env_cfg=jenga_env_cfg(),
    play_env_cfg=jenga_env_cfg(play=True),
    rl_cfg=jenga_ppo_runner_cfg(),
)


# Timur integration: incomplete tower + random target block.
#
# Keep this as a separate task so the stable fixed-block Mjlab-Jenga checkpoints
# keep their exact environment/action semantics.
from mjlab_jenga import jenga_incomplete_random_cfg as _incomplete_random  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-IncompleteRandom",
    env_cfg=_incomplete_random.jenga_env_cfg(),
    play_env_cfg=_incomplete_random.jenga_env_cfg(play=True),
    rl_cfg=_incomplete_random.jenga_ppo_runner_cfg(),
)


# V2 adds Timur's task-frame/home-relative touch and yaw actions on top of the
# random-block command.
from mjlab_jenga import jenga_incomplete_random_v2_cfg as _incomplete_random_v2  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-Jenga-IncompleteRandomV2",
    env_cfg=_incomplete_random_v2.jenga_env_cfg(),
    play_env_cfg=_incomplete_random_v2.jenga_env_cfg(play=True),
    rl_cfg=_incomplete_random_v2.jenga_ppo_runner_cfg(),
)
