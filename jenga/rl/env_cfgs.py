from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  time_out,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.envs.mdp.rewards import action_rate_l2
from mjlab.viewer import ViewerConfig

from constants import *
import scene
from mdp.observations import *
from mdp.actions import *
from mdp.events import *
from mdp.rewards import *
from mdp.metrics import *
from mdp.commands import *
from mdp.terminations import *

def _make_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "pusher_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _HOOK_ALL_CFG}
        ),
        "pusher_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _HOOK_ALL_CFG}
        ),
        "pusher_target_home_error": ObservationTermCfg(
            func=hook_joint_pos_relative_to_target_home,
        ),
        "target_selection": ObservationTermCfg(
            func=target_selection_features,
        ),
        "hook_tip_task_position": ObservationTermCfg(
            func=hook_tip_pos_in_task_frame,
        ),
        "target_task_movement": ObservationTermCfg(
            func=target_block_movement_in_task_frame,
        ),
        "target_task_velocity": ObservationTermCfg(
            func=target_block_vel_in_task_frame,
        ),
        "hook_contact": ObservationTermCfg(
            func=hook_contact_observation,
        ),
        "support_presence": ObservationTermCfg(
            func=target_support_presence,
        ),
        "tower_state": ObservationTermCfg(
            func=tower_state_observation,
        ),
        "last_action": ObservationTermCfg(
            func=last_action,
        ),
    }

    critic_terms = {
        **actor_terms,
        "block_all_pos": ObservationTermCfg(
            func=all_block_pos,
        ),
    }


    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms),
    }


    actions : dict[str, ActionTermCfg] = {
        "push_stop_retreat": PushStopRetreatActionCfg(
            entity_name="hook",
        ),
        "block_local_touch": BlockLocalHookYZActionCfg(
            entity_name="hook",
            scale=(CONTACT_X_LIMIT, CONTACT_Z_LIMIT),
            asset_cfg=_TARGET_BLOCK_CFG,
        ),
        "yaw" : CurriculumYawActionCfg(
            entity_name="hook",
            scale=YAW_ACTION_SCALE,
        ),
    }


    events = {
        "randomize_block_physics": EventTermCfg(
            func=randomize_block_physics,
            mode="reset",
        ),
        "randomize_missing_blocks": EventTermCfg(
            func=randomize_missing_blocks,
            mode="reset",
        ),
    }


    rewards = {
        "normalized_new_progress": RewardTermCfg(
            func=NormalizedDeltaBlockProgressReward(),
            weight=PROGRESS_REWARD_WEIGHT,
        ),
        "action_rate": RewardTermCfg(
            func=action_rate_l2,
            weight=ACTION_RATE_REWARD_WEIGHT,
        ),
        "action_magnitude": RewardTermCfg(
            func=action_magnitude_l2,
            weight=ACTION_MAGNITUDE_REWARD_WEIGHT,
        ),
        "sustained_stuck": RewardTermCfg(
            func=SustainedStuckPenalty(),
            weight=STUCK_REWARD_WEIGHT,
        ),
        "successful_extract": RewardTermCfg(
            func=success_block_reward,
            weight=SUCCESS_REWARD_WEIGHT,
        ),
        "tower_instability": RewardTermCfg(
            func=NewTowerInstabilityPenalty(),
            weight=TOWER_INSTABILITY_REWARD_WEIGHT,
        ),
        "tower_damage": RewardTermCfg(
            func=tower_damage_signal,
            weight=TOWER_DAMAGE_REWARD_WEIGHT,
        ),
        "timeout": RewardTermCfg(
            func=time_out,
            weight=TIMEOUT_REWARD_WEIGHT,
        ),
        "debug_reward_signals": RewardTermCfg(
            func=debug_reward_signals,
            weight=1e-12,
        ),
    }

    metrics = {
        "block_progress_last": MetricsTermCfg(
            func=block_progress,
            reduce="last",
        ),
        "extraction_reached_last": MetricsTermCfg(
            func=target_extraction_reached,
            reduce="last",
        ),
        "normalized_new_progress_mean": MetricsTermCfg(
            func=NormalizedDeltaBlockProgressReward(),
            reduce="mean",
        ),
        "success_last": MetricsTermCfg(
            func=success_block_reward,
            reduce="last",
        ),
        "tower_com_shift_last": MetricsTermCfg(
            func=tower_com_shift,
            reduce="last",
        ),
        "tower_instability_mean": MetricsTermCfg(
            func=tower_instability_fraction,
            reduce="mean",
        ),
        "tower_damage_mean": MetricsTermCfg(
            func=tower_damage_signal,
            reduce="mean",
        ),
        "timeout_last": MetricsTermCfg(
            func=time_out,
            reduce="last",
        ),
        "tower_max_block_horizontal_shift_last": MetricsTermCfg(
            func=tower_max_block_horizontal_shift,
            reduce="last",
        ),
        "tower_max_block_vertical_shift_last": MetricsTermCfg(
            func=tower_max_block_vertical_shift,
            reduce="last",
        ),
        "tower_max_block_rotation_last": MetricsTermCfg(
            func=tower_max_block_rotation,
            reduce="last",
        ),
        "action_norm_mean": MetricsTermCfg(
            func=action_norm,
            reduce="mean",
        ),
        "hook_contact_force_mean": MetricsTermCfg(
            func=hook_contact_force_norm,
            reduce="mean",
        ),
        "hook_contact_found_mean": MetricsTermCfg(
            func=hook_contact_found,
            reduce="mean",
        ),
        "stuck_contact_mean": MetricsTermCfg(
            func=stuck_contact_signal,
            reduce="mean",
        ),
        "stop_action_mean": MetricsTermCfg(
            func=stop_action_fraction,
            reduce="mean",
        ),
        "retreat_action_mean": MetricsTermCfg(
            func=retreat_action_fraction,
            reduce="mean",
        ),
        "hook_x_position_last": MetricsTermCfg(
            func=hook_x_position,
            params={"asset_cfg": _HOOK1_CFG},
            reduce="last",
        ),
    }

    terminations = {
        "success": TerminationTermCfg(func=success_block_extract),
        "tower_damage": TerminationTermCfg(func=tower_damage),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    commands = {
        "target_block": TargetBlockCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
        ),
    }


    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=build_entities(),
            sensors=(
                ContactSensorCfg(
                    name=HOOK_CONTACT_SENSOR_NAME,
                    primary=ContactMatch(
                        mode="subtree",
                        pattern="hook_tool",
                        entity="hook",
                    ),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                    history_length=5,
                ),
            ),
            num_envs=512,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        scale_rewards_by_dt=False,
        metrics=metrics,
        terminations=terminations,
        commands=commands,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=1.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            nconmax=4096,
            njmax=4096,
            # impratio is the stiffness of friction constraints relative to
            # normal ones. At MuJoCo's default of 1.0 the two are equally compliant,
            # so contacts creep tangentially well below the friction limit and the
            # tower shears as a unit: pushing b6_1 displaces b8_1, two layers above,
            # by 89% as far, and twists the top of the tower by 8.45 deg. Since
            # tower_damage triggers at 25 mm for any non-target block, it fires at
            # roughly 28 mm of target progress against a success distance of
            # 112.5 mm, leaving 12 of 14 targets unextractable.
            #
            # Mean drag ratio over b6_1, b6_3, b3_1 and b7_1, pyramidal cone:
            #
            #        mu     impratio=1   impratio=10   impratio=30
            #      0.20          0.566         0.323         0.270
            #      0.28          0.615         0.272         0.196
            #      0.48          0.896         0.285         0.194
            #
            # The sign of the friction dependence flips. At impratio=1 higher friction
            # increases drag, because the quantity measured is compliance scaling with
            # contact load. Once that compliance is removed, higher friction decreases
            # drag, as neighbouring blocks hold one another. The configured friction
            # range is therefore appropriate and impratio is the parameter that
            # required correction. A value of 30 lies inside MuJoCo's recommended
            # range of 10-100 for friction-critical contact and raises the number of
            # extractable targets from 2 to 7 of 14.
            #
            # The elliptic cone is unusable here: it drives the reset settling
            # transient past the damage limit, with tower_damage firing at step 6
            # while the target is still stationary.
            mujoco=MujocoCfg(timestep=0.002, impratio=30.0),
        ),
        decimation=5,
        episode_length_s=20.0,
    )

def jenga_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg()

    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False

    return cfg