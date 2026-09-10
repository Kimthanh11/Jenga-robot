# Jenga Robot – Policy Playback

This document describes how to replay and inspect a trained Jenga policy using the custom `scripts/play.py` script.

The script extends the standard MJLab playback functionality with additional options specific to the Jenga task, including target-block selection, missing-block configurations, yaw control, and termination debugging.

## Running the Policy

Run the script from the `Jenga-robot/mjlab_jenga/jenga` directory:

```bash
python -m scripts.play --agent trained --checkpoint <CHECKPOINT_PATH>
```

For example:

```bash
python -m scripts.play \
    --agent trained \
    --checkpoint "checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt" \
    --freeze-yaw \
    --missing 3
```

On Windows PowerShell, the command can be written on one line:

```powershell
python -m scripts.play --agent trained --checkpoint "D:\Darmstadt\Semester_3\integrated robot\jenga\Jenga-robot\checkpoints\jenga_low_level_missing3_noyaw_model_8500.pt" --device cpu --freeze-yaw --missing 3
```

## Playback Options

The custom `scripts/play.py` provides additional options for evaluating and debugging the Jenga policy.

| Option | Description |
|---|---|
| `--agent trained` | Replay a trained PPO policy. Requires `--checkpoint`. |
| `--agent zero` | Send zero actions. Useful for inspecting the tower and reset configuration. |
| `--agent random` | Send random actions for debugging or comparison. |
| `--missing N` | Run with `N` missing blocks, e.g. `--missing 3`. Default: `0`. |
| `--target BLOCK` | Force a specific target block, e.g. `--target b2_3`. Otherwise, the target is sampled randomly. |
| `--freeze-yaw` | Keep the hook at its home yaw. Use for policies trained without yaw variation. |
| `--force-yaw` | Enable the full yaw curriculum. Cannot be combined with `--freeze-yaw`. |
| `--debug-target` | Print target and reset geometry for debugging. |
| `--no-terminations` | Disable success and tower-damage terminations. |


## Recommended Playback Configuration

When evaluating a checkpoint, the playback configuration should match the conditions under which the policy was trained.

For the checkpoint:

```text
jenga_low_level_missing3_noyaw_model_8500.pt
```

the corresponding playback command is:

```bash
python -m scripts.play \
    --agent trained \
    --checkpoint "checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt" \
    --missing 3 \
    --freeze-yaw
```

Here:

- `--agent trained` loads the trained policy.
- `--checkpoint` specifies the PPO checkpoint.
- `--missing 3` selects the three-missing-block condition.
- `--freeze-yaw` reproduces the no-yaw training condition.

Using the same task configuration during training and playback is important for evaluating the policy under the intended conditions.