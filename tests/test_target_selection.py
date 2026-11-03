import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("mjlab", "torch"))
if HAS_RUNTIME:
    import torch
    import mjlab_jenga.jenga_mjenv_cfg as cfg


@unittest.skipUnless(HAS_RUNTIME, "requires the mjlab training environment")
class TargetSelectionTest(unittest.TestCase):
    def make_command(self, targets=("b2_1", "b8_2"), num_envs=128, **kwargs):
        names, _ = cfg._target_block_entries()
        hook = Mock()
        hook.find_joints.return_value = ([0, 1, 2, 3], [])
        scene = {name: Mock() for name in names}
        for block in scene.values():
            block.data.body_link_pos_w = torch.zeros(num_envs, 1, 3)
            block.data.body_link_vel_w = torch.zeros(num_envs, 1, 6)
            block.data.body_link_pose_w = torch.zeros(num_envs, 1, 7)
            block.data.body_link_pose_w[:, :, 3] = 1.0
        scene["hook"] = hook
        env = SimpleNamespace(
            scene=scene,
            sim=Mock(),
            observation_manager=SimpleNamespace(_obs_buffer=object()),
            num_envs=num_envs,
            device="cpu",
            common_step_counter=0,
        )
        command_cfg = cfg.TargetBlockCommandCfg(
            resampling_time_range=(1e9, 1e9), selectable_target_names=targets, **kwargs
        )
        command = command_cfg.build(env)
        cfg._ensure_missing_block_state(env)
        return command, env

    def sample(self, command, ids=None):
        if ids is None:
            ids = torch.arange(command.num_envs)
        with patch.object(cfg, "random_target_block_scale", return_value=1.0), patch.object(
            cfg, "random_target_with_missing_scale", return_value=1.0
        ):
            command._resample_command(ids)

    def test_factory_observes_target_override_in_train_and_play(self):
        targets = ("b2_1", "b6_1", "b7_2", "b8_2")
        with patch.object(cfg, "RANDOM_TARGET_BLOCK_NAMES", targets):
            for play in (False, True):
                self.assertEqual(
                    cfg.jenga_env_cfg(play=play).commands["target_block"].selectable_target_names,
                    targets,
                )

    def test_candidate_is_selected_when_present_but_never_when_missing(self):
        command, env = self.make_command()
        removed = cfg.MISSING_BLOCK_CANDIDATES.index("b8_2")
        env._jenga_missing_block_mask[:64, removed] = True
        seen = set()
        torch.manual_seed(41)
        for _ in range(8):
            self.sample(command)
            present = command._present_by_block()
            self.assertTrue(present.gather(1, command.selected_block_idx[:, None]).all())
            seen.update(command.selected_block_idx[64:].tolist())
        self.assertEqual(seen, {command._all_names.index("b2_1"), command._all_names.index("b8_2")})

    def test_partial_and_empty_reset_preserve_other_targets(self):
        command, _ = self.make_command()
        before = command.selected_block_idx.clone()
        self.sample(command, torch.arange(0, 128, 2))
        self.assertTrue(torch.equal(command.selected_block_idx[1::2], before[1::2]))
        before = command.selected_block_idx.clone()
        self.sample(command, torch.empty(0, dtype=torch.long))
        self.assertTrue(torch.equal(command.selected_block_idx, before))

    def test_all_missing_selectable_targets_fail_clearly(self):
        command, env = self.make_command(targets=("b8_2",))
        env._jenga_missing_block_mask[:, cfg.MISSING_BLOCK_CANDIDATES.index("b8_2")] = True
        with self.assertRaisesRegex(ValueError, "No selectable target is present"):
            self.sample(command)

    def test_forced_missing_target_fails_before_moving_hook(self):
        for forced in ({"force_target_name": "b8_2"}, {"force_target_names": ("b8_2",)}):
            command, env = self.make_command(**forced)
            env._jenga_missing_block_mask[:, cfg.MISSING_BLOCK_CANDIDATES.index("b8_2")] = True
            with self.assertRaisesRegex(ValueError, "fixed or forced target is missing"):
                self.sample(command)
            command._hook.write_joint_position_to_sim.assert_not_called()

    def test_original_random_choice_rng_path_is_unchanged(self):
        command, _ = self.make_command(targets=("b2_1", "b2_2", "b2_3", "b3_1"))
        torch.manual_seed(22)
        torch.rand(command.num_envs)
        torch.rand(command.num_envs)
        expected = command._selectable[torch.randint(0, 4, (command.num_envs,))]
        torch.manual_seed(22)
        self.sample(command)
        self.assertTrue(torch.equal(command.selected_block_idx, expected))

    def test_sequential_play_can_remove_any_tower_block(self):
        _, env = self.make_command()
        block_name = "b2_1"
        asset = env.scene[block_name]
        asset.data.default_root_state = torch.zeros(env.num_envs, 13)
        env_ids = torch.tensor([1, 3], dtype=torch.long)

        cfg.remove_block_from_tower(env, block_name, env_ids=env_ids)

        missing = cfg.current_missing_block_mask(env, block_name)
        self.assertTrue(torch.equal(torch.nonzero(missing).flatten(), env_ids))
        parked_state = asset.write_root_state_to_sim.call_args.args[0]
        self.assertTrue(torch.all(parked_state[:, :3] != 0.0))
        self.assertTrue(torch.equal(parked_state[:, 7:], torch.zeros(2, 6)))
        self.assertTrue(
            torch.equal(asset.write_root_state_to_sim.call_args.kwargs["env_ids"], env_ids)
        )
        env.sim.forward.assert_called_once_with()

    def test_target_change_refreshes_cached_metrics(self):
        command, env = self.make_command()
        with patch.object(command, "_update_metrics") as update:
            command.set_target_block("b2_1")
        update.assert_called_once_with()
        self.assertIsNone(env.observation_manager._obs_buffer)

    def test_rebaseline_and_restore_pristine_reference(self):
        command, _ = self.make_command(num_envs=1)
        original = command._start_pos.clone()
        for index, block in enumerate(command._blocks):
            block.data.body_link_pos_w[0, 0] = torch.tensor(
                [index * 0.01, index * 0.02, index * 0.03]
            )

        command.rebaseline()
        expected = torch.stack(
            [block.data.body_link_pos_w[0, 0] for block in command._blocks]
        )
        self.assertTrue(torch.equal(command._start_pos, expected))

        command.restore_pristine_reference()
        self.assertTrue(torch.equal(command._start_pos, original))

    def test_absolute_tower_state_does_not_depend_on_selected_target(self):
        command, _ = self.make_command(num_envs=1)
        before = command.absolute_tower_state()
        command.set_target_block("b8_2")
        after = command.absolute_tower_state()
        self.assertEqual(before, after)

    def test_empty_and_unknown_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one selectable"):
            self.make_command(targets=())
        with self.assertRaisesRegex(ValueError, "Unknown selectable"):
            self.make_command(targets=("b8_2", "b20_1"))


if __name__ == "__main__":
    unittest.main()
