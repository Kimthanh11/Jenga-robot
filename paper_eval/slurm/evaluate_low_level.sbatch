#!/bin/bash
# Evaluate a checkpoint across targets and missing-block levels.
#
# Usage:
#   sbatch paper_eval/slurm/evaluate_low_level.sbatch checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt
#   sbatch paper_eval/slurm/evaluate_low_level.sbatch checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt heldout-legal 0
#
# Named target sets: trained, trained-legal, heldout-legal, legal, tower.

#SBATCH --job-name=jenga_eval
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 08:00:00
#SBATCH --output=logs/paper_eval/jenga_eval-%j.out
#SBATCH --error=logs/paper_eval/jenga_eval-%j.out

set -euo pipefail

CHECKPOINT="${1:?checkpoint path required}"
TARGETS="${2:-trained}"
LEVELS_CSV="${3:-0,1,2,3}"
EPISODES_PER_SEED="${4:-50}"
NUM_ENVS="${5:-50}"
LEGACY="${6:-}"
IMPRATIO="${7:-}"
CONE="${8:-}"       # pyramidal or elliptic; overrides the config default
YAWLIMIT="${9:-freeze}"  # freeze, configured, or a numeric limit in radians
ABORT="${10:-}"
SEEDS="${11:-1,2,3,4,5}"
MAX_STEPS="${12:-2000}"
SETTLE="${13:-0}"   # control steps simulated after success before the tower is judged again
FRICTION="${14:-}"  # LO,HI sliding-friction range, e.g. 0.20,0.60; empty keeps the configured range

# REPO_DIR defaults to the timur side clone; the venv is the shared one of the main
# project (same pattern as train_incomplete_random.sbatch). Submit from REPO_DIR, or make
# sure REPO_DIR/logs/paper_eval exists first: sbatch opens the --output file before this
# script runs.
REPO_DIR="${REPO_DIR:-/home/share/jenga_project/Jenga-robot-timur}"
VENV_PY=/home/share/jenga_project/Jenga-robot/.venv/bin/python
cd "${REPO_DIR}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/paper_eval

echo "commit:  $(git rev-parse HEAD)"
echo "cwd:     $PWD"
echo "python:  $("${VENV_PY}" -c 'import sys; print(sys.executable)')"
echo "cfg:     $("${VENV_PY}" -c 'import mjlab_jenga_paper.jenga_mjenv_cfg as c; print(c.__file__)')"

IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"
echo "yawlimit: ${YAWLIMIT:-default}"
EPISODES_OUT="logs/paper_eval/eval-${SLURM_JOB_ID}-episodes.csv"
SUMMARY_OUT="logs/paper_eval/eval-${SLURM_JOB_ID}-summary.csv"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

CMD=("${VENV_PY}" paper_eval/scripts/evaluate_low_level.py
  --checkpoint "${CHECKPOINT}"
  --targets "${TARGETS}"
  --missing-levels "${LEVELS[@]}"
  --episodes-per-seed "${EPISODES_PER_SEED}"
  --num-envs "${NUM_ENVS}"
  --seeds "${SEEDS}"
  --max-steps "${MAX_STEPS}"
  --episodes-csv "${EPISODES_OUT}"
  --summary-csv "${SUMMARY_OUT}")
if [ -n "${LEGACY}" ];   then CMD+=(--legacy-distribution); fi
if [ -n "${IMPRATIO}" ]; then CMD+=(--impratio "${IMPRATIO}"); fi
if [ -n "${CONE}" ];     then CMD+=(--cone "${CONE}"); fi
if [ -n "${ABORT}" ]; then CMD+=(--abort); fi
if [ "${SETTLE}" != "0" ]; then CMD+=(--settle-steps "${SETTLE}"); fi
if [ -n "${FRICTION}" ]; then CMD+=(--friction-range "${FRICTION}"); fi
if [ "${YAWLIMIT}" = "freeze" ]; then
  CMD+=(--freeze-yaw)
elif [ "${YAWLIMIT}" = "configured" ]; then
  CMD+=(--enable-yaw)
elif [ -n "${YAWLIMIT}" ]; then
  CMD+=(--enable-yaw --yaw-limit "${YAWLIMIT}")
fi

printf 'cmd:     %s\n' "${CMD[*]}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
