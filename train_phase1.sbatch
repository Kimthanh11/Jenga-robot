#!/bin/bash
# Train the Phase-1 baseline (Mjlab-Jenga-Phase1) on the ICC, for comparison.
#   sbatch train_phase1.sbatch            # defaults: 1000 iters, 256 envs
#   sbatch train_phase1.sbatch 1500 256   # iters, num_envs
#SBATCH --job-name=jenga-phase1
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail

ITERS="${1:-1000}"
NUM_ENVS="${2:-256}"

cd /home/share/jenga_project/Jenga-robot

PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga-Phase1 \
  --agent.logger tensorboard \
  --agent.experiment-name jenga_phase1 \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
