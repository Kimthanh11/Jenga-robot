#!/bin/bash
# Train the incomplete-tower + random-block V2 policy (touch/yaw curriculum).
#   sbatch train_incomplete_random_v2.sbatch            # defaults: 6000 iters, 256 envs
#   sbatch train_incomplete_random_v2.sbatch 6000 256   # iters, num_envs
# NOTE: 256 envs — that is what fits in GPU memory for this tower. Do NOT raise.
#SBATCH --job-name=jenga-incrand-v2
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=timuruttsal@gmail.com
set -euo pipefail

ITERS="${1:-6000}"
NUM_ENVS="${2:-256}"

# REPO_DIR is overridable so the run can live in a side clone while the shared
# working tree stays untouched. VENV always comes from the main project.
REPO_DIR="${REPO_DIR:-/home/share/jenga_project/Jenga-robot}"
VENV_PY=/home/share/jenga_project/Jenga-robot/.venv/bin/python
SHARED_CKPT_DIR=/home/share/jenga_project/Jenga-robot/shared_checkpoints

cd "${REPO_DIR}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 "${VENV_PY}" train.py Mjlab-Jenga-IncompleteRandomV2 \
  --agent.logger tensorboard \
  --agent.experiment-name jenga_incomplete_randblock_v2 \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"

# Publish the newest checkpoint of this run to the shared checkpoint dir.
RUN_DIR="$(ls -dt logs/rsl_rl/jenga_incomplete_randblock_v2/*/ | head -1)"
LAST_MODEL="$(ls -t "${RUN_DIR}"model_*.pt | head -1)"
cp "${LAST_MODEL}" \
  "${SHARED_CKPT_DIR}/jenga_incomplete_randblock_v2_$(basename "${RUN_DIR}")_$(basename "${LAST_MODEL}")"
