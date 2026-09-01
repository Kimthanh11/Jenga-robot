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

## Evaluation

The report-facing protocol, target splits, policy evaluation, scripted controls and
analysis commands are documented in [docs/evaluation.md](docs/evaluation.md). In
particular, `heldout-legal` contains the 20 legal tower blocks that were never sampled
as training targets.

Run the pure utility tests with:

```bash
python -m unittest discover -s tests -v
```
