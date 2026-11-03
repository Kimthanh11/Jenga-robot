import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("mjlab", "torch"))
if HAS_RUNTIME:
    import torch

    from mjlab_jenga.jenga_mjenv_cfg import TargetBlockCommand
    from mjlab_jenga.sim_state import capture_sim_state, restore_sim_state


@unittest.skipUnless(HAS_RUNTIME, "requires torch and mjlab")
class SimStateTest(unittest.TestCase):
    def make_env(self, *, include_play_mask: bool = True):
        data = SimpleNamespace(
            time=torch.tensor([0.75]),
            qpos=torch.tensor([[1.0, 2.0, 3.0]]),
            qvel=torch.tensor([[4.0, 5.0]]),
            act=torch.tensor([[0.25]]),
            qacc_warmstart=torch.tensor([[0.6, 0.7]]),
            ctrl=torch.tensor([[0.8, 0.9]]),
            qfrc_applied=torch.tensor([[1.1, 1.2]]),
            xfrc_applied=torch.tensor([[[1.3, 1.4]]]),
            eq_active=torch.tensor([[True, False]]),
            mocap_pos=torch.tensor([[[1.5, 1.6, 1.7]]]),
            mocap_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        )
        command = Mock(spec=TargetBlockCommand)
        command.reference_positions.return_value = torch.tensor(
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]
        )
        push_term = SimpleNamespace(
            _raw_actions=torch.tensor([[0.11]]),
            _processed_velocity=torch.tensor([[0.12]]),
        )
        yaw_term = SimpleNamespace(
            _processed_targets=torch.tensor([[0.13]]),
            _needs_home_init=torch.tensor([False]),
        )
        action_terms = {
            "push_stop_retreat": push_term,
            "yaw": yaw_term,
        }
        env = SimpleNamespace(
            sim=SimpleNamespace(data=data, forward=Mock()),
            command_manager=SimpleNamespace(get_term=Mock(return_value=command)),
            observation_manager=SimpleNamespace(
                _obs_buffer={"actor": torch.tensor([[99.0]])}
            ),
            action_manager=SimpleNamespace(
                action=torch.tensor([[0.1, 0.2]]),
                prev_action=torch.tensor([[0.3, 0.4]]),
                prev_prev_action=torch.tensor([[0.5, 0.6]]),
                active_terms=list(action_terms),
                get_term=action_terms.__getitem__,
            ),
            _jenga_missing_block_mask=torch.tensor([[False, True, False]]),
        )
        if include_play_mask:
            env._jenga_play_removed_block_mask = torch.tensor(
                [[True, False, False]]
            )
        return env, command

    def test_capture_clones_complete_state(self):
        env, _ = self.make_env()
        state = capture_sim_state(env)

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
            value = getattr(env.sim.data, field)
            if value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.add_(10.0)
        env._jenga_missing_block_mask.logical_not_()
        env._jenga_play_removed_block_mask.logical_not_()
        env.action_manager.action.add_(30.0)
        env.action_manager.prev_action.add_(40.0)
        env.action_manager.prev_prev_action.add_(50.0)
        push_term = env.action_manager.get_term("push_stop_retreat")
        push_term._processed_velocity.add_(60.0)
        yaw_term = env.action_manager.get_term("yaw")
        yaw_term._needs_home_init.logical_not_()

        self.assertTrue(torch.equal(state.qpos, torch.tensor([[1.0, 2.0, 3.0]])))
        self.assertTrue(torch.equal(state.qvel, torch.tensor([[4.0, 5.0]])))
        for field in simulation_fields:
            self.assertFalse(
                torch.equal(getattr(env.sim.data, field), getattr(state, field))
            )
        self.assertTrue(
            torch.equal(
                state.start_pos,
                torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]),
            )
        )
        self.assertTrue(
            torch.equal(
                state.missing_block_mask,
                torch.tensor([[False, True, False]]),
            )
        )
        self.assertTrue(
            torch.equal(
                state.play_removed_block_mask,
                torch.tensor([[True, False, False]]),
            )
        )
        self.assertTrue(torch.equal(state.action, torch.tensor([[0.1, 0.2]])))
        self.assertTrue(
            torch.equal(state.prev_action, torch.tensor([[0.3, 0.4]]))
        )
        self.assertTrue(
            torch.equal(state.prev_prev_action, torch.tensor([[0.5, 0.6]]))
        )
        self.assertTrue(
            torch.equal(
                state.action_term_state["push_stop_retreat"]["_processed_velocity"],
                torch.tensor([[0.12]]),
            )
        )
        self.assertTrue(
            torch.equal(
                state.action_term_state["yaw"]["_needs_home_init"],
                torch.tensor([False]),
            )
        )

    def test_restore_copies_complete_state_and_updates_simulation(self):
        env, command = self.make_env()
        state = capture_sim_state(env)
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
            getattr(env.sim.data, field).zero_()
        env._jenga_missing_block_mask.zero_()
        env._jenga_play_removed_block_mask.zero_()
        env.action_manager.action.zero_()
        env.action_manager.prev_action.zero_()
        env.action_manager.prev_prev_action.zero_()
        for term_name in env.action_manager.active_terms:
            term = env.action_manager.get_term(term_name)
            for value in vars(term).values():
                if isinstance(value, torch.Tensor):
                    value.zero_()

        restore_sim_state(env, state)

        for field in simulation_fields:
            self.assertTrue(
                torch.equal(getattr(env.sim.data, field), getattr(state, field))
            )
        self.assertTrue(
            torch.equal(env._jenga_missing_block_mask, state.missing_block_mask)
        )
        self.assertTrue(
            torch.equal(
                env._jenga_play_removed_block_mask,
                state.play_removed_block_mask,
            )
        )
        self.assertTrue(torch.equal(env.action_manager.action, state.action))
        self.assertTrue(
            torch.equal(env.action_manager.prev_action, state.prev_action)
        )
        self.assertTrue(
            torch.equal(
                env.action_manager.prev_prev_action,
                state.prev_prev_action,
            )
        )
        for term_name, attributes in state.action_term_state.items():
            term = env.action_manager.get_term(term_name)
            for name, expected in attributes.items():
                self.assertTrue(torch.equal(getattr(term, name), expected))
        env.sim.forward.assert_called_once_with()
        command.restore_reference_positions.assert_called_once_with(state.start_pos)
        self.assertIsNone(env.observation_manager._obs_buffer)

    def test_capture_without_optional_play_mask(self):
        env, _ = self.make_env(include_play_mask=False)

        state = capture_sim_state(env)

        self.assertIsNone(state.play_removed_block_mask)

    def test_restore_clears_play_mask_if_snapshot_has_none(self):
        env, _ = self.make_env(include_play_mask=False)
        state = capture_sim_state(env)
        env._jenga_play_removed_block_mask = torch.ones((1, 3), dtype=torch.bool)

        restore_sim_state(env, state)

        self.assertFalse(env._jenga_play_removed_block_mask.any().item())

    def test_restore_creates_missing_play_mask(self):
        env, _ = self.make_env()
        state = capture_sim_state(env)
        del env._jenga_play_removed_block_mask

        restore_sim_state(env, state)

        self.assertTrue(
            torch.equal(
                env._jenga_play_removed_block_mask,
                state.play_removed_block_mask,
            )
        )
        self.assertIsNot(
            env._jenga_play_removed_block_mask,
            state.play_removed_block_mask,
        )


if __name__ == "__main__":
    unittest.main()
