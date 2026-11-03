""" This is to generate snapshots / a dataset of a specific towerstate tested with multiple block extractions"""
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_jenga.jenga_mjenv_cfg import TargetBlockCommand


@dataclass
class JengaSimState:
    time: torch.Tensor
    #joint and block positions; qvel describes how the move
    qpos: torch.Tensor
    qvel: torch.Tensor

    act: torch.Tensor
    qacc_warmstart: torch.Tensor
    ctrl: torch.Tensor # commands for the simulated actuators
    qfrc_applied: torch.Tensor #generalized force
    xfrc_applied: torch.Tensor
    eq_active: torch.Tensor
    mocap_pos: torch.Tensor
    mocap_quat: torch.Tensor
    start_pos: torch.Tensor
    missing_block_mask: torch.Tensor
    play_removed_block_mask: torch.Tensor | None

    action: torch.Tensor
    prev_action: torch.Tensor
    prev_prev_action: torch.Tensor
    action_term_state: dict[str, dict[str, torch.Tensor]]


def _target_command(env: ManagerBasedRlEnv) -> TargetBlockCommand:
    command = env.command_manager.get_term("target_block")
    if not isinstance(command, TargetBlockCommand):
        raise TypeError("target_block has an unexpected command type")
    return command


def _capture_action_term_state(
    env: ManagerBasedRlEnv,
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        result[term_name] = {
            name: value.clone()
            for name, value in vars(term).items()
            if isinstance(value, torch.Tensor)
        }
    return result


def _restore_action_term_state(
    env: ManagerBasedRlEnv,
    state: dict[str, dict[str, torch.Tensor]],
) -> None:
    for term_name, attributes in state.items():
        term = env.action_manager.get_term(term_name)
        for name, saved_value in attributes.items():
            current_value = getattr(term, name)
            if not isinstance(current_value, torch.Tensor):
                raise TypeError(
                    f"Action term '{term_name}.{name}' is no longer a tensor"
                )
            if current_value.shape != saved_value.shape:
                raise ValueError(
                    f"Action term state shape mismatch for '{term_name}.{name}': "
                    f"expected {tuple(current_value.shape)}, "
                    f"got {tuple(saved_value.shape)}"
                )
            current_value.copy_(saved_value)


def capture_sim_state(env: ManagerBasedRlEnv) -> JengaSimState:
    command = _target_command(env)

    start_pos = command.reference_positions()
    qpos = env.sim.data.qpos.clone()
    qvel = env.sim.data.qvel.clone()

    # get the masks, play masks might not exist initially
    missing_block_mask = env._jenga_missing_block_mask.clone()
    play_removed_block_mask = getattr(
        env,
        "_jenga_play_removed_block_mask",
        None,
    )
    if play_removed_block_mask is not None:
        play_removed_block_mask = play_removed_block_mask.clone()

    action = env.action_manager.action.clone()
    prev_action = env.action_manager.prev_action.clone()
    prev_prev_action = env.action_manager.prev_prev_action.clone()

    return JengaSimState(
        time=env.sim.data.time.clone(),
        qpos=qpos,
        qvel=qvel,
        act=env.sim.data.act.clone(),
        qacc_warmstart=env.sim.data.qacc_warmstart.clone(),
        ctrl=env.sim.data.ctrl.clone(),
        qfrc_applied=env.sim.data.qfrc_applied.clone(),
        xfrc_applied=env.sim.data.xfrc_applied.clone(),
        eq_active=env.sim.data.eq_active.clone(),
        mocap_pos=env.sim.data.mocap_pos.clone(),
        mocap_quat=env.sim.data.mocap_quat.clone(),
        start_pos=start_pos,
        missing_block_mask=missing_block_mask,
        play_removed_block_mask=play_removed_block_mask,
        action=action,
        prev_action=prev_action,
        prev_prev_action=prev_prev_action,
        action_term_state=_capture_action_term_state(env),
    )


def restore_sim_state(env: ManagerBasedRlEnv, state: JengaSimState) -> None:
    command = _target_command(env)

    env.sim.data.time.copy_(state.time)
    env.sim.data.qpos.copy_(state.qpos)
    env.sim.data.qvel.copy_(state.qvel)
    env.sim.data.act.copy_(state.act)
    env.sim.data.qacc_warmstart.copy_(state.qacc_warmstart)
    env.sim.data.ctrl.copy_(state.ctrl)
    env.sim.data.qfrc_applied.copy_(state.qfrc_applied)
    env.sim.data.xfrc_applied.copy_(state.xfrc_applied)
    env.sim.data.eq_active.copy_(state.eq_active)
    env.sim.data.mocap_pos.copy_(state.mocap_pos)
    env.sim.data.mocap_quat.copy_(state.mocap_quat)
    env._jenga_missing_block_mask.copy_(state.missing_block_mask)

    env.action_manager.action.copy_(state.action)
    env.action_manager.prev_action.copy_(state.prev_action)
    env.action_manager.prev_prev_action.copy_(state.prev_prev_action)
    _restore_action_term_state(env, state.action_term_state)

    current_play_removed_mask = getattr(
        env,
        "_jenga_play_removed_block_mask",
        None,
    )

    if state.play_removed_block_mask is None:
        if current_play_removed_mask is not None:
            current_play_removed_mask.zero_()
    elif current_play_removed_mask is None:
        env._jenga_play_removed_block_mask = state.play_removed_block_mask.clone()
    else:
        current_play_removed_mask.copy_(state.play_removed_block_mask)

    env.sim.forward()
    command.restore_reference_positions(state.start_pos)
    # MJLab caches the last observation. It belongs to the discarded branch and
    # must be recomputed from the restored simulation on the next request.
    env.observation_manager._obs_buffer = None
