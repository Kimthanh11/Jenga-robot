#!/bin/bash
# Train Mjlab-Jenga staged curriculum from scratch on the IAS cluster.
#
# Usage from repo root:
#   sbatch slurm/train_curriculum.sbatch
#   sbatch slurm/train_curriculum.sbatch 2000 256
#
# Args:
#   1: iterations, default 10000
#   2: num envs, default 256

#SBATCH --job-name=jenga_curriculum
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_curriculum-%j.out
#SBATCH --error=logs/slurm/jenga_curriculum-%j.out

set -euo pipefail

ITERS="${1:-10000}"
NUM_ENVS="${2:-256}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

echo "Training Mjlab-Jenga from scratch"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python -m scripts.legacy.train Mjlab-Jenga \
  --agent.logger tensorboard \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
