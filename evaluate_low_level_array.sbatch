#!/bin/bash
# Evaluate one target per array task while keeping the named target-set label.
#
# Usage:
#   sbatch --array=0-19%2 evaluate_low_level_array.sbatch \
#     logs/rsl_rl/jenga/<run>/model_8500.pt heldout-legal

#SBATCH --job-name=jenga_eval
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 08:00:00
#SBATCH --output=logs/slurm/jenga_eval-%A_%a.out
#SBATCH --error=logs/slurm/jenga_eval-%A_%a.out

set -euo pipefail

CHECKPOINT="${1:?checkpoint path required}"
TARGETS="${2:?named target set required}"
MISSING_LEVEL="${3:-0}"
EPISODES_PER_SEED="${4:-50}"
NUM_ENVS="${5:-50}"
SEEDS="${6:-1,2,3,4,5}"
MAX_STEPS="${7:-2000}"
YAW_MODE="${8:-freeze}"
TARGET_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script with --array}"
RESULT_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm logs/eval

EPISODES_OUT="logs/eval/eval-${RESULT_ID}-episodes.csv"
SUMMARY_OUT="logs/eval/eval-${RESULT_ID}-summary.csv"

CMD=(./.venv/bin/python evaluate_low_level.py
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
