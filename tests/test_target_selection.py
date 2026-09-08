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
    def make_command(self, targets=("b2_1", "b8_2"), **kwargs):
        names, _ = cfg._target_block_entries()
        hook = Mock()
        hook.find_joints.return_value = ([0, 1, 2, 3], [])
        scene = {name: Mock() for name in names}
        scene["hook"] = hook
        env = SimpleNamespace(scene=scene, num_envs=128, device="cpu", common_step_counter=0)
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

    def test_empty_and_unknown_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one selectable"):
            self.make_command(targets=())
        with self.assertRaisesRegex(ValueError, "Unknown selectable"):
            self.make_command(targets=("b8_2", "b20_1"))


if __name__ == "__main__":
    unittest.main()
