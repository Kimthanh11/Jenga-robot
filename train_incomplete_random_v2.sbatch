#!/bin/bash
# Train the incomplete-tower + random-block V2 policy (touch/yaw curriculum).
# Curriculum lengths (perturbation, success, touch, yaw) scale automatically with
# ITERS via train_incomplete_random_staged.py -- see that script's docstring for why
# (a hardcoded absolute-step schedule silently completes a different fraction of
# itself depending on how many iterations you launch with).
#   sbatch train_incomplete_random_v2.sbatch            # defaults: 6000 iters, 1536 envs
#   sbatch train_incomplete_random_v2.sbatch 6000 1536   # iters, num_envs
# NOTE: 1536 verified stable (nconmax=1024/njmax=512, 2026-08-21 investigation); real ceiling untested above that.
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
NUM_ENVS="${2:-1536}"
RUN_SUFFIX="${3:-}"

# REPO_DIR is overridable so the run can live in a side clone while the shared
# working tree stays untouched. VENV always comes from the main project.
REPO_DIR="${REPO_DIR:-/home/share/jenga_project/Jenga-robot}"
VENV_PY=/home/share/jenga_project/Jenga-robot/.venv/bin/python
SHARED_CKPT_DIR=/home/share/jenga_project/Jenga-robot/shared_checkpoints
EXP_NAME=jenga_incomplete_randblock_v2

cd "${REPO_DIR}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

SUFFIX_ARGS=()
if [[ -n "${RUN_SUFFIX}" ]]; then
  SUFFIX_ARGS=(--run-suffix "${RUN_SUFFIX}")
fi

PYTHONUNBUFFERED=1 "${VENV_PY}" train_incomplete_random_staged.py v2 \
  --iterations "${ITERS}" \
  --num-envs "${NUM_ENVS}" \
  "${SUFFIX_ARGS[@]}"

# Publish the newest checkpoint of this run to the shared checkpoint dir.
RUN_DIR="$(ls -dt "logs/rsl_rl/${EXP_NAME}"/*/ | head -1)"
LAST_MODEL="$(ls -t "${RUN_DIR}"model_*.pt | head -1)"
cp "${LAST_MODEL}" \
  "${SHARED_CKPT_DIR}/${EXP_NAME}_$(basename "${RUN_DIR}")_$(basename "${LAST_MODEL}")"
