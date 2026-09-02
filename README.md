# Jenga Robot

This repository contains the mjlab environment, PPO training scripts and evaluation
tools for a low-level Jenga block-extraction policy. The policy acts in a target-local
task frame and is trained with randomized target blocks and block physics.

## Setup

```bash
uv sync
```

Training on the IAS cluster is launched through `train_low_level_stage.sbatch`. The
Python entry point is `train_low_level_stage.py`; stage selection and resume arguments
are documented by `--help` and in the sbatch header.

## Reference checkpoint

The reference low-level policy is available at
[`checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt`](checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt).
It was trained for random target blocks and tower configurations with up to three
missing blocks. Yaw must remain frozen when this checkpoint is evaluated:

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

See [`checkpoints/README.md`](checkpoints/README.md) for provenance and checksum.

## Evaluation

The report-facing protocol, target splits, policy evaluation, scripted controls and
analysis commands are documented in [docs/evaluation.md](docs/evaluation.md). In
particular, `heldout-legal` contains the 20 legal tower blocks that were never sampled
as training targets.

Run the pure utility tests with:

```bash
python -m unittest discover -s tests -v
```
