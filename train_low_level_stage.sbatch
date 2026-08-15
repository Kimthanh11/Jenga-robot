#!/bin/bash
# Train an explicit Jenga low-level stage.
#
# Usage:
#   sbatch train_low_level_stage.sbatch fixed 10000 768
#   sbatch train_low_level_stage.sbatch target 10000 768
#   sbatch train_low_level_stage.sbatch missing1 10000 768

#SBATCH --job-name=jenga_low
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_low-%j.out
#SBATCH --error=logs/slurm/jenga_low-%j.out

set -euo pipefail

STAGE="${1:-fixed}"
ITERS="${2:-10000}"
NUM_ENVS="${3:-768}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

echo "Training Jenga low-level stage"
echo "  stage:      ${STAGE}"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python train_low_level_stage.py "${STAGE}" \
  --iterations "${ITERS}" \
  --num-envs "${NUM_ENVS}"
