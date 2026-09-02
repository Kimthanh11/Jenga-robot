#!/bin/bash
# Train Timur's integrated incomplete-tower + random-block + task-frame touch/yaw task.
#
# Usage from repo root:
#   sbatch slurm/train_incomplete_random_v2.sbatch
#   sbatch slurm/train_incomplete_random_v2.sbatch 4000 256
#
# Args:
#   1: iterations, default 4000
#   2: num envs, default 256

#SBATCH --job-name=jenga_rand_v2
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_rand_v2-%j.out
#SBATCH --error=logs/slurm/jenga_rand_v2-%j.out

set -euo pipefail

ITERS="${1:-4000}"
NUM_ENVS="${2:-256}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

echo "Training Mjlab-Jenga-IncompleteRandomV2"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python -m scripts.legacy.train Mjlab-Jenga-IncompleteRandomV2 \
  --agent.logger tensorboard \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
