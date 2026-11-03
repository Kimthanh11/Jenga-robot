#!/bin/bash

#SBATCH --job-name=jenga_random
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080|v100'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 01:00:00
#SBATCH --output=logs/slurm/jenga_random-%j.out
#SBATCH --error=logs/slurm/jenga_random-%j.out

set -euo pipefail

CHECKPOINT="${1:?checkpoint path required}"
SETTLE_STEPS="${2:-100}"
SEED="${3:-42}"
OUTPUT="${4:-logs/evaluation/jenga_random-${SLURM_JOB_ID}.json}"
MAX_ATTEMPTS="${5:-1}"
GAMES="${6:-1}"
MAX_STEPS="${7:-2000}"
MAX_MOVES="${8:-0}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

./.venv/bin/python -m scripts.evaluate_selection \
  --checkpoint "${CHECKPOINT}" \
  --device cuda:0 \
  --settle-steps "${SETTLE_STEPS}" \
  --seed "${SEED}" \
  --max-steps-per-block "${MAX_STEPS}" \
  --max-attempts-per-block "${MAX_ATTEMPTS}" \
  --max-moves-per-game "${MAX_MOVES}" \
  --games "${GAMES}" \
  --output "${OUTPUT}"
