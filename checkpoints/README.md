# Reference Checkpoint

## `jenga_low_level_missing3_noyaw_model_8500.pt`

- Controller: trained PPO policy
- Task: random target block extraction
- Training configuration: up to three missing non-target blocks, yaw frozen
- Original run: `2026-08-29_00-06-31_low_level_missing3_noyaw`
- Original file: `model_8500.pt`
- SHA256: `270a967671fc70b53147a65ffe9d7be2b6879083c5dffc08770f45da2b3a1170`

Run the policy with:

```bash
uv run python play_random_target.py \
  --agent trained \
  --checkpoint checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt \
  --viewer native \
  --device cpu \
  --num-envs 1 \
  --freeze-yaw \
  --missing 3
```

Use `--missing 0` for an intact tower or `--target b2_1` to select a fixed target.
`play_sequential.py` is a test procedure and does not have a separate checkpoint.
