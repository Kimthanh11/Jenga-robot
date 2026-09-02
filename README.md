# Jenga Robot

This repository contains the mjlab environment, PPO training scripts and evaluation
tools for a low-level Jenga block-extraction policy. The policy acts in a target-local
task frame and is trained with randomized target blocks and block physics.

## Setup

```bash
uv sync
```

Training on the IAS cluster is launched through `slurm/train_low_level_stage.sbatch`.
The Python entry point is `scripts/train_low_level_stage.py`.

## Reference checkpoint

[`checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt`](checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt)
is the random-target policy trained with up to three missing blocks and frozen yaw.

```bash
uv run python -m scripts.play_random_target \
  --agent trained \
  --checkpoint checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt \
  --viewer native \
  --device cpu \
  --num-envs 1 \
  --freeze-yaw \
  --missing 3
```

SHA256: `270a967671fc70b53147a65ffe9d7be2b6879083c5dffc08770f45da2b3a1170`

## Evaluation

The report-facing protocol, target splits, policy evaluation, scripted controls and
analysis commands are documented in [docs/evaluation.md](docs/evaluation.md). In
particular, `heldout-legal` contains the 20 legal tower blocks that were never sampled
as training targets.

Repository layout:

- `mjlab_jenga/`: environment and task configuration
- `assets/`: MuJoCo model assets
- `checkpoints/`: published reference checkpoints
- `scripts/`: training, evaluation, calibration and playback tools
- `scripts/legacy/`: early standalone MuJoCo experiments
- `slurm/`: IAS cluster launchers
- `docs/`: maintained experiment protocols
- `tests/`: utility tests

Run the pure utility tests with:

```bash
python -m unittest discover -s tests -v
```
