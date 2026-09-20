#!/bin/bash
# Evaluate one target per array task while keeping the named target-set label.
#
# Usage:
#   sbatch --array=0-19%2 paper_eval/slurm/evaluate_low_level_array.sbatch \
#     checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt heldout-legal
#
# Settling check over all legal targets, one seed, 200 steps held after success:
#   sbatch --array=0-23%1 paper_eval/slurm/evaluate_low_level_array.sbatch \
#     checkpoints/jenga_low_level_missing3_noyaw_model_8500.pt legal 0 50 50 1 2000 freeze 200

#SBATCH --job-name=jenga_eval
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 08:00:00
#SBATCH --output=logs/paper_eval/jenga_eval-%A_%a.out
#SBATCH --error=logs/paper_eval/jenga_eval-%A_%a.out

set -euo pipefail

CHECKPOINT="${1:?checkpoint path required}"
TARGETS="${2:?named target set required}"
MISSING_LEVEL="${3:-0}"
EPISODES_PER_SEED="${4:-50}"
NUM_ENVS="${5:-50}"
SEEDS="${6:-1,2,3,4,5}"
MAX_STEPS="${7:-2000}"
YAW_MODE="${8:-freeze}"
SETTLE="${9:-0}"   # control steps simulated after success before the tower is judged again
FRICTION="${10:-}" # LO,HI sliding-friction range, e.g. 0.20,0.60; empty keeps the configured range
TARGET_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script with --array}"
RESULT_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# REPO_DIR defaults to the timur side clone; the venv is the shared one of the main
# project (same pattern as train_incomplete_random.sbatch). The --output path is relative
# to the directory you submit from and sbatch opens it before this script runs, so run
# `mkdir -p logs/paper_eval` there once before the first submission.
REPO_DIR="${REPO_DIR:-/home/share/jenga_project/Jenga-robot-timur}"
VENV_PY=/home/share/jenga_project/Jenga-robot/.venv/bin/python
cd "${REPO_DIR}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/paper_eval

EPISODES_OUT="logs/paper_eval/eval-${RESULT_ID}-episodes.csv"
SUMMARY_OUT="logs/paper_eval/eval-${RESULT_ID}-summary.csv"

CMD=("${VENV_PY}" paper_eval/scripts/evaluate_low_level.py
  --checkpoint "${CHECKPOINT}"
  --targets "${TARGETS}"
  --target-index "${TARGET_INDEX}"
  --missing-levels "${MISSING_LEVEL}"
  --episodes-per-seed "${EPISODES_PER_SEED}"
  --num-envs "${NUM_ENVS}"
  --seeds "${SEEDS}"
  --max-steps "${MAX_STEPS}"
  --episodes-csv "${EPISODES_OUT}"
  --summary-csv "${SUMMARY_OUT}")
if [ "${YAW_MODE}" = "freeze" ]; then
  CMD+=(--freeze-yaw)
elif [ "${YAW_MODE}" = "configured" ]; then
  CMD+=(--enable-yaw)
else
  echo "yaw mode must be freeze or configured" >&2
  exit 2
fi
if [ "${SETTLE}" != "0" ]; then CMD+=(--settle-steps "${SETTLE}"); fi
if [ -n "${FRICTION}" ]; then CMD+=(--friction-range "${FRICTION}"); fi

echo "commit:       $(git rev-parse HEAD)"
echo "checkpoint:   ${CHECKPOINT}"
echo "targets:      ${TARGETS}"
echo "target index: ${TARGET_INDEX}"
echo "result id:    ${RESULT_ID}"
printf 'cmd:          %s\n' "${CMD[*]}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 "${CMD[@]}"
