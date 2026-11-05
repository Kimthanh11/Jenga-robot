#!/bin/bash
#SBATCH --job-name=jenga_high
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080|v100'
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_high-%j.out
#SBATCH --error=logs/slurm/jenga_high-%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 CHECKPOINT [ITERATIONS] [SEED]"
  exit 2
fi

CHECKPOINT=$1
ITERATIONS=${2:-100}
SEED=${3:-42}

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

./.venv/bin/python -m scripts.train_high_level \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --iterations "$ITERATIONS" \
  --seed "$SEED"
