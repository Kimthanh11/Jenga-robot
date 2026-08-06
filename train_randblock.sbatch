#!/bin/bash
# Train the random-block worker (Mjlab-Jenga-RandomBlock) on the ICC.
#   sbatch train_randblock.sbatch            # defaults: 1000 iters, 256 envs
#   sbatch train_randblock.sbatch 1500 256   # iters, num_envs
# NOTE: 256 envs — that is what fits in GPU memory for this 27-block tower. Do NOT raise.
#SBATCH --job-name=jenga-randblock
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

PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga-RandomBlock \
  --agent.logger tensorboard \
  --agent.experiment-name jenga_randblock \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
