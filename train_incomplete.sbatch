#!/bin/bash
#SBATCH --job-name=jenga-incomplete
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lhkimthanh3001@gmail.com

set -euo pipefail

ITERS="${1:-4000}"
NUM_ENVS="${2:-256}"
CHECKPOINT="shared_checkpoints/jenga_curriculum_model_1999.pt"

cd /home/share/jenga_project/Jenga-robot

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga-Incomplete \
  --agent.logger tensorboard \
  --agent.experiment-name jenga_incomplete \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.resume True \
  --agent.load-run curriculum \
  --agent.load-checkpoint model_1999.pt