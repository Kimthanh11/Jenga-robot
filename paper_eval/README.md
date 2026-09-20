# paper_eval

Evaluation tooling behind the numbers of the Jenga paper, vendored from `origin/borisbranch`
at commit 56527ca8a5e5b6b864a046cc2d9074e45b6a8845 (files copied with `git show`, `mjlab_jenga`
imports rewritten to `mjlab_jenga_paper`). It is intentionally separate from `mjlab_jenga`:
`mjlab_jenga_paper/` registers no mjlab tasks and shares no code with the timur task configuration.

- `mjlab_jenga_paper/`: `jenga_mjenv_cfg.py`, `evaluation_utils.py`, empty `__init__.py`.
- `paper_eval/scripts/`: `evaluate_low_level.py` (policy), `evaluate_scripted_baselines.py`
  (settle, straight, pulsed, tap), `compare_conditions.py` and `analyze_evaluation.py` (CSV only; also `analyze_evaluation.py <episode csvs>`).
- `paper_eval/slurm/`: sbatch wrappers; `REPO_DIR` defaults to the timur side clone, output to `logs/paper_eval/`.
- `checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt`: reported checkpoint, sha256 270a9676... (`--abort` needs `jenga_abort_cfg.py`, not vendored).

Run from the repo root in an environment with mjlab:

    python paper_eval/scripts/evaluate_low_level.py --checkpoint checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt --targets b2_2,b4_1 --missing-levels 0 --episodes-per-seed 10 --seeds 1 --episodes-csv p-ep.csv --summary-csv p-sum.csv
    python paper_eval/scripts/evaluate_scripted_baselines.py --controllers straight --targets b2_2,b4_1 --missing-levels 0 --episodes-per-seed 10 --seeds 1 --episodes-csv s-ep.csv --summary-csv s-sum.csv
    python paper_eval/scripts/compare_conditions.py --condition "policy=p-ep.csv" --condition "straight@straight=s-ep.csv"

Cluster: `mkdir -p logs/paper_eval` in the side clone, submit from it with
`sbatch paper_eval/slurm/evaluate_low_level.sbatch <checkpoint> <targets> <levels> <episodes> <num_envs>`
(positional arguments are documented in each sbatch header). Tests: `python -m unittest discover -s paper_eval/tests`.
