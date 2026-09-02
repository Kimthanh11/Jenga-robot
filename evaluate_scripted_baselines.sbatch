#!/bin/bash
# Evaluate simple controllers with the same scenarios and horizon as the policy.
#
# Usage:
#   sbatch evaluate_scripted_baselines.sbatch heldout-legal 0
#   sbatch evaluate_scripted_baselines.sbatch trained-legal 0,1,2,3 straight,pulsed,tap

#SBATCH --job-name=jenga_base
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 08:00:00
#SBATCH --output=logs/slurm/jenga_base-%j.out
#SBATCH --error=logs/slurm/jenga_base-%j.out

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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm logs/eval
IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"

EPISODES_OUT="logs/eval/baseline-${SLURM_JOB_ID}-episodes.csv"
SUMMARY_OUT="logs/eval/baseline-${SLURM_JOB_ID}-summary.csv"

echo "commit:      $(git rev-parse HEAD)"
echo "targets:     ${TARGETS}"
echo "controllers: ${CONTROLLERS}"
echo "levels:      ${LEVELS_CSV}"
echo "seeds:       ${SEEDS}"
echo "cycle:       push=${PUSH_STEPS} pause=${PAUSE_STEPS} retreat=${RETREAT_STEPS}"
echo "contact:     x=${CONTACT_X} z=${CONTACT_Z}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python evaluate_scripted_baselines.py \
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
  --episodes-csv "${EPISODES_OUT}" \
  --summary-csv "${SUMMARY_OUT}"
