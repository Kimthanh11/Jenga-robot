import importlib.util
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("mjlab", "torch"))
POSITION_REPLAY_ATOL = 1.0e-6
VELOCITY_REPLAY_ATOL = 2.0e-4
if HAS_RUNTIME:
    import torch

    import mjlab_jenga.jenga_mjenv_cfg as jenga_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_jenga.jenga_mjenv_cfg import (
        TargetBlockCommand,
        jenga_env_cfg,
    )
    from mjlab_jenga.sim_state import capture_sim_state, restore_sim_state


@unittest.skipUnless(HAS_RUNTIME, "requires torch and mjlab")
class SimStateIntegrationTest(unittest.TestCase):
    def test_round_trip_restores_real_jenga_environment(self):
        jenga_cfg.apply_low_level_stage("target")
        jenga_cfg.MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.0
        jenga_cfg.MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.0

        cfg = jenga_env_cfg(play=True)
        cfg.auto_reset = False
        cfg.scene.num_envs = 1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env = ManagerBasedRlEnv(cfg=cfg, device=device)

        try:
            env.reset(seed=42)
            command = env.command_manager.get_term("target_block")
            self.assertIsInstance(command, TargetBlockCommand)
            state = capture_sim_state(env)

            test_actions = torch.tensor(
                [[0.25, 0.1, -0.1, 0.0]],
                device=env.device,
            )
            for _ in range(3):
                env.step(test_actions)
            expected_qpos_after_replay = env.sim.data.qpos.clone()
            expected_qvel_after_replay = env.sim.data.qvel.clone()

            env.sim.data.qpos.add_(0.25)
            env.sim.data.qvel.sub_(0.5)
            env._jenga_missing_block_mask.logical_not_()
            play_mask = getattr(env, "_jenga_play_removed_block_mask", None)
            if play_mask is not None:
                play_mask.logical_not_()
            env.action_manager.action.add_(1.0)
            env.action_manager.prev_action.add_(2.0)
            env.action_manager.prev_prev_action.add_(3.0)
            command.restore_reference_positions(state.start_pos + 0.01)
            self.assertIsNotNone(env.observation_manager._obs_buffer)

            restore_sim_state(env, state)

            simulation_fields = (
                "time",
                "qpos",
                "qvel",
                "act",
                "qacc_warmstart",
                "ctrl",
                "qfrc_applied",
                "xfrc_applied",
                "eq_active",
                "mocap_pos",
                "mocap_quat",
            )
            for field in simulation_fields:
                with self.subTest(restored_simulation_field=field):
                    actual = getattr(env.sim.data, field)
                    expected = getattr(state, field)
                    self.assertTrue(
                        torch.equal(actual, expected),
                        msg=(
                            f"{field} differs immediately after restore; "
                            f"max_abs_diff={self._max_abs_diff(actual, expected)}"
                        ),
                    )
            self.assertTrue(
                torch.equal(
                    env._jenga_missing_block_mask,
                    state.missing_block_mask,
                )
            )
            restored_play_mask = getattr(
                env,
                "_jenga_play_removed_block_mask",
                None,
            )
            if state.play_removed_block_mask is None:
                self.assertTrue(
                    restored_play_mask is None
                    or not restored_play_mask.any().item()
                )
            else:
                self.assertIsNotNone(restored_play_mask)
                self.assertTrue(
                    torch.equal(restored_play_mask, state.play_removed_block_mask)
                )
            self.assertTrue(
                torch.equal(env.action_manager.action, state.action)
            )
            self.assertTrue(
                torch.equal(env.action_manager.prev_action, state.prev_action)
            )
            self.assertTrue(
                torch.equal(
                    env.action_manager.prev_prev_action,
                    state.prev_prev_action,
                )
            )
            self.assertTrue(
                torch.equal(command.reference_positions(), state.start_pos)
            )
            self.assertIsNone(env.observation_manager._obs_buffer)

            for _ in range(3):
                env.step(test_actions)
            self.assertTrue(
                torch.allclose(
                    env.sim.data.qpos,
                    expected_qpos_after_replay,
                    atol=POSITION_REPLAY_ATOL,
                    rtol=0.0,
                ),
                msg=(
                    "qpos differs after deterministic replay; "
                    f"max_abs_diff={self._max_abs_diff(env.sim.data.qpos, expected_qpos_after_replay)}"
                ),
            )
            self.assertTrue(
                torch.allclose(
                    env.sim.data.qvel,
                    expected_qvel_after_replay,
                    atol=VELOCITY_REPLAY_ATOL,
                    rtol=0.0,
                ),
                msg=(
                    "qvel differs after deterministic replay; "
                    f"max_abs_diff={self._max_abs_diff(env.sim.data.qvel, expected_qvel_after_replay)}"
                ),
            )
        finally:
            env.close()

    @staticmethod
    def _max_abs_diff(actual, expected) -> float:
        if actual.numel() == 0:
            return 0.0
        if actual.dtype == torch.bool:
            return float(torch.count_nonzero(actual != expected).item())
        return float(torch.max(torch.abs(actual - expected)).item())


if __name__ == "__main__":
    unittest.main()
