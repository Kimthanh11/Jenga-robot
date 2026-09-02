# Evaluation protocol

The policy is evaluated by target block instead of only by aggregate reward. Each
original vector environment contributes exactly one episode, so a fast successful
environment cannot be counted repeatedly while a slow failure is still running.

## Target sets

- `trained`: all seven targets used during training, including the three top-layer
  blocks.
- `trained-legal`: the four training targets that are legal Jenga moves.
- `heldout-legal`: the 20 legal blocks that never appeared in the training target set.
- `legal`: all 24 non-top-layer blocks.
- `tower`: all 27 blocks, including the top layer.

The old selector `all` is rejected because it previously meant the seven training
targets and was easy to misread as the whole tower.

## Reproducibility

The policy and scripted baselines share reset seeds, exact missing-block patterns,
episode budgets and scenario identifiers. A missing pattern is excluded whenever it
would remove the selected target. Per-episode CSVs record the commit, checkpoint,
target, pattern, seed, physics settings, terminal reason, peak and final tower
deformation, contact statistics and success outcome.

The default evaluation protocol uses five seeds and 50 episodes per seed and target.
It holds the success distance at 112.5 mm and caps every episode at 2000 control steps
(20 seconds). Yaw is frozen by default because the current comparison checkpoint was
trained with yaw frozen. Pass `configured` as the ninth sbatch argument only for a
checkpoint trained with yaw enabled.

## Policy evaluation

```bash
sbatch slurm/evaluate_low_level.sbatch \
  logs/rsl_rl/jenga/<run>/model_<iteration>.pt \
  heldout-legal 0 50 50 "" "" "" freeze "" 1,2,3,4,5 2000
```

Run `trained-legal` with the same arguments to measure the generalization gap. Missing
levels `1`, `2`, and `3` test robustness separately from target generalization.

Large target sets with many timeouts can exceed one job's wall-time. Evaluate one
target per array task while retaining the `heldout-legal` label:

```bash
P_HELD=$(sbatch --array=0-19%2 slurm/evaluate_low_level_array.sbatch \
  logs/rsl_rl/jenga/<run>/model_<iteration>.pt heldout-legal | awk '{print $4}')
```

The twenty episode files are named `eval-<array-job>_<task>-episodes.csv` and can be
passed to the analyzer with one shell glob.

## Scripted controls

```bash
sbatch slurm/evaluate_scripted_baselines.sbatch \
  heldout-legal 0 settle,straight,pulsed,tap 50 50 1,2,3,4,5 2000 \
  20 10 5 0.0 0.0
```

- `settle`: zero velocity command; measures reset drift and false damage.
- `straight`: continuous full-speed push.
- `pulsed`: approach until contact, then fixed push/pause periods.
- `tap`: approach until contact, then fixed push/retreat/pause periods.

The latter two use only a binary contact trigger. They are simple feedback heuristics,
not open-loop controllers and not learned policies.

The final five arguments are push, pause and retreat steps followed by normalized
block-local contact x/z. Keep x/z at zero for the pre-registered centre-contact
comparison. Different duty cycles should be separate jobs, for example `20 20 5` and
`10 30 5`; choosing the best one on the reported test seeds would bias the comparison.

The complete four-controller, five-seed protocol is too long for one cluster job.
Run each controller/seed combination as a bounded array task instead:

```bash
B_TRAIN=$(sbatch --array=0-19%2 slurm/evaluate_scripted_baselines.sbatch \
  trained-legal 0 protocol-array | awk '{print $4}')
B_HELD=$(sbatch --array=0-19%2 slurm/evaluate_scripted_baselines.sbatch \
  heldout-legal 0 protocol-array | awk '{print $4}')
```

Each array element writes a distinct file named
`baseline-<array-job>_<task>-episodes.csv`. Do not combine CSVs left by timed-out
monolithic jobs with these complete runs because they contain duplicate scenarios.

## Analysis

Combine episode CSVs from matching policy and baseline jobs:

```bash
./.venv/bin/python -m scripts.analyze_evaluation \
  logs/eval/eval-<policy-trained-job>-episodes.csv \
  logs/eval/eval-<policy-heldout-array-job>_*-episodes.csv \
  logs/eval/baseline-<trained-array-job>_*-episodes.csv \
  logs/eval/baseline-<heldout-array-job>_*-episodes.csv
```

The script reports micro and per-target rates with Wilson confidence intervals, the
macro generalization gap, and exact paired McNemar tests on shared `scenario_id`s.
