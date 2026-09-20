# paper_eval

Evaluation tooling behind the numbers of the Jenga paper, vendored from `origin/borisbranch`
at commit 56527ca8a5e5b6b864a046cc2d9074e45b6a8845 (files copied with `git show`, `mjlab_jenga`
imports rewritten to `mjlab_jenga_paper`). It is intentionally separate from `mjlab_jenga`:
`mjlab_jenga_paper/` registers no mjlab tasks and shares no code with the timur task configuration.

- `mjlab_jenga_paper/`: `jenga_mjenv_cfg.py`, `evaluation_utils.py`, an `__init__.py` that deliberately registers nothing.
- `paper_eval/scripts/`: `evaluate_low_level.py` (policy), `evaluate_scripted_baselines.py`
  (settle, straight, pulsed, tap, forcelim, stall), `compare_conditions.py`, `analyze_evaluation.py` and
  `analyze_failure_modes.py` (the last three read episode CSVs only).
- `paper_eval/slurm/`: sbatch wrappers; `REPO_DIR` defaults to the timur side clone, output to `logs/paper_eval/`.
- `checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt`: reported checkpoint, sha256 270a9676...
- Not vendored: `tests/test_target_selection.py` (needs mjlab at runtime, imports the original package name) and `jenga_abort_cfg.py`; the `--abort` mode of evaluate_low_level.py is therefore unsupported here.

Run from the repo root in an environment with mjlab:

    python paper_eval/scripts/evaluate_low_level.py --checkpoint checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt --targets b2_2,b4_1 --missing-levels 0 --episodes-per-seed 10 --seeds 1 --episodes-csv p-ep.csv --summary-csv p-sum.csv
    python paper_eval/scripts/evaluate_scripted_baselines.py --controllers straight --targets b2_2,b4_1 --missing-levels 0 --episodes-per-seed 10 --seeds 1 --episodes-csv s-ep.csv --summary-csv s-sum.csv
    python paper_eval/scripts/compare_conditions.py --condition "policy=p-ep.csv" --condition "straight@straight=s-ep.csv"

Cluster: `mkdir -p logs/paper_eval` in the side clone, submit from it with
`sbatch paper_eval/slurm/evaluate_low_level.sbatch <checkpoint> <targets> <levels> <episodes> <num_envs>`
(positional arguments are documented in each sbatch header). Tests: `python -m unittest discover -s paper_eval/tests`.

Options (defaults leave earlier behaviour unchanged; 100 Hz control, so 200 steps = 2 s):
- `--settle-steps N` (both evaluation scripts): stop pushing at success, keep simulating N steps, judge the tower again.
  Adds `settle_steps`, `success_step`, `reached_success`, `safe_after_settle`, `damaged_after_success` columns.
- `--friction-range LO,HI` (both): replaces the sliding-friction sampling range; recorded in `friction_sliding_min/max`.
- Baseline `forcelim`: full push, command scaled by `--force-limit`/force (default 5.0 N) when the contact force exceeds it.
- Baseline `stall`: full push; from first contact, a stall is contact force > `--stall-force` (2.0 N) with target progress
  gain < `--stall-eps` (0.001 m) over `--stall-window` (20) steps; then retreat `--stall-retreat-steps` (100) steps and hold.
- `analyze_failure_modes.py <csv or glob>... [--layers 6,7,8]`: outcome shares, exceeded damage limits (xy > 25 mm, z > 15 mm,
  rot > 15 deg, in metres and degrees as logged) and timeouts with or without extraction.
- sbatch: positional `SETTLE` and `FRICTION` (low-level: 13, 14; array: 9, 10; baselines: 13, 14), `EXTRA_ARGS` for baselines.
