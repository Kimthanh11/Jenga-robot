#!/bin/bash
# Evaluate simple controllers with the same scenarios and horizon as the policy.
#
# Usage:
#   sbatch paper_eval/slurm/evaluate_scripted_baselines.sbatch heldout-legal 0
#   sbatch paper_eval/slurm/evaluate_scripted_baselines.sbatch trained-legal 0,1,2,3 straight,pulsed,tap
#   sbatch --array=0-19%2 paper_eval/slurm/evaluate_scripted_baselines.sbatch heldout-legal 0 protocol-array
#   sbatch paper_eval/slurm/evaluate_scripted_baselines.sbatch legal 0 straight,forcelim,stall 50 50 1 2000 20 10 5 0.0 0.0 200

#SBATCH --job-name=jenga_base
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 08:00:00
#SBATCH --output=logs/paper_eval/jenga_base-%j.out
#SBATCH --error=logs/paper_eval/jenga_base-%j.out

set -euo pipefail

TARGETS="${1:-trained}"
LEVELS_CSV="${2:-0,1,2,3}"
CONTROLLERS="${3:-settle,straight,pulsed,tap}"
EPISODES_PER_SEED="${4:-50}"
NUM_ENVS="${5:-50}"
SEEDS="${6:-1,2,3,4,5}"
MAX_STEPS="${7:-2000}"
PUSH_STEPS="${8:-20}"
PAUSE_STEPS="${9:-10}"
RETREAT_STEPS="${10:-5}"
CONTACT_X="${11:-0.0}"
CONTACT_Z="${12:-0.0}"
SETTLE="${13:-0}"      # control steps simulated after success before the tower is judged again
FRICTION="${14:-}"     # LO,HI sliding-friction range; empty keeps the configured range
# Optional extra flags of the script, e.g. EXTRA_ARGS="--force-limit 3 --stall-window 30"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Array index 0..19 selects one controller and seed pair.
if [ "${CONTROLLERS}" = "protocol-array" ]; then
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "protocol-array requires: sbatch --array=0-19%N ..." >&2
    exit 2
  fi
  if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge 20 ]; then
    echo "protocol-array task index must be in 0..19" >&2
    exit 2
  fi
  PROTOCOL_CONTROLLERS=(settle straight pulsed tap)
  PROTOCOL_SEEDS=(1 2 3 4 5)
  CONTROLLER_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
  SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
  CONTROLLERS="${PROTOCOL_CONTROLLERS[$CONTROLLER_INDEX]}"
  SEEDS="${PROTOCOL_SEEDS[$SEED_INDEX]}"
fi

# REPO_DIR defaults to the timur side clone; the venv is the shared one of the main
# project (same pattern as train_incomplete_random.sbatch). The --output path is relative
# to the directory you submit from and sbatch opens it before this script runs, so run
# `mkdir -p logs/paper_eval` there once before the first submission.
REPO_DIR="${REPO_DIR:-/home/share/jenga_project/Jenga-robot-timur}"
VENV_PY=/home/share/jenga_project/Jenga-robot/.venv/bin/python
cd "${REPO_DIR}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/paper_eval
IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"

RESULT_ID="${SLURM_JOB_ID:-local}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  RESULT_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
EPISODES_OUT="logs/paper_eval/baseline-${RESULT_ID}-episodes.csv"
SUMMARY_OUT="logs/paper_eval/baseline-${RESULT_ID}-summary.csv"

echo "commit:      $(git rev-parse HEAD)"
echo "targets:     ${TARGETS}"
echo "controllers: ${CONTROLLERS}"
echo "levels:      ${LEVELS_CSV}"
echo "seeds:       ${SEEDS}"
echo "result id:   ${RESULT_ID}"
echo "cycle:       push=${PUSH_STEPS} pause=${PAUSE_STEPS} retreat=${RETREAT_STEPS}"
echo "contact:     x=${CONTACT_X} z=${CONTACT_Z}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 "${VENV_PY}" paper_eval/scripts/evaluate_scripted_baselines.py \
  --controllers "${CONTROLLERS}" \
  --targets "${TARGETS}" \
  --missing-levels "${LEVELS[@]}" \
  --episodes-per-seed "${EPISODES_PER_SEED}" \
  --num-envs "${NUM_ENVS}" \
  --seeds "${SEEDS}" \
  --max-steps "${MAX_STEPS}" \
  --push-steps "${PUSH_STEPS}" \
  --pause-steps "${PAUSE_STEPS}" \
  --retreat-steps "${RETREAT_STEPS}" \
  --contact-x "${CONTACT_X}" \
  --contact-z "${CONTACT_Z}" \
  --device cuda:0 \
  --settle-steps "${SETTLE}" \
  ${FRICTION:+--friction-range "${FRICTION}"} \
  --episodes-csv "${EPISODES_OUT}" \
  --summary-csv "${SUMMARY_OUT}" \
  ${EXTRA_ARGS}
