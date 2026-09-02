#!/bin/bash
# Resume the incomplete-tower + random-block ABORT (v2 base) run on the IAS cluster.
# The stud partition caps jobs at 24h; a 6000-iter run at 1536 envs takes ~30h, so it
# needs at least one resume to finish. Auto-detects the latest checkpoint if run/
# checkpoint aren't given explicitly.
#
# Usage from repo root:
#   sbatch train_resume_incomplete_random_abort.sbatch
#   sbatch train_resume_incomplete_random_abort.sbatch 6000 1536
#   sbatch train_resume_incomplete_random_abort.sbatch 6000 1536 2026-08-27_13-00-00 model_2300.pt
#
# Args:
#   1: target iterations, default 6000
#   2: num envs, default 1536
#   3: run folder under logs/rsl_rl/jenga_incomplete_randblock_abort_v2, optional
#   4: checkpoint file inside that run folder, optional

#SBATCH --job-name=jenga-incrand-abort-resume
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=timuruttsal@gmail.com

set -euo pipefail

ITERS="${1:-6000}"
NUM_ENVS="${2:-1536}"
LOAD_RUN="${3:-}"
LOAD_CHECKPOINT="${4:-}"

REPO_DIR="${REPO_DIR:-/home/share/jenga_project/Jenga-robot}"
VENV_PY=/home/share/jenga_project/Jenga-robot/.venv/bin/python
SHARED_CKPT_DIR=/home/share/jenga_project/Jenga-robot/shared_checkpoints
EXP_NAME=jenga_incomplete_randblock_abort_v2

cd "${REPO_DIR}"

if [[ -z "${LOAD_RUN}" || -z "${LOAD_CHECKPOINT}" ]]; then
  LATEST_MODEL="$(
    find "logs/rsl_rl/${EXP_NAME}" -name 'model_*.pt' \
      ! -name 'model_0.pt' \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )"

  if [[ -z "${LATEST_MODEL}" ]]; then
    echo "No checkpoint found under logs/rsl_rl/${EXP_NAME}."
    exit 1
  fi

  LOAD_RUN="$(basename "$(dirname "${LATEST_MODEL}")")"
  LOAD_CHECKPOINT="$(basename "${LATEST_MODEL}")"
fi

echo "Resuming ${EXP_NAME}"
echo "  run:        ${LOAD_RUN}"
echo "  checkpoint: ${LOAD_CHECKPOINT}"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

# --iterations is the run's TOTAL target (same as a fresh launch), not "iterations
# remaining" -- train_incomplete_random_staged.py sizes every curriculum's absolute
# step count off it, so passing the same total here keeps the curriculum schedule
# consistent across the resume instead of quietly redefining it.
PYTHONUNBUFFERED=1 "${VENV_PY}" train_incomplete_random_staged.py abort \
  --iterations "${ITERS}" \
  --num-envs "${NUM_ENVS}" \
  --load-run "${LOAD_RUN}" \
  --load-checkpoint "${LOAD_CHECKPOINT}"

RUN_DIR="$(ls -dt "logs/rsl_rl/${EXP_NAME}"/*/ | head -1)"
LAST_MODEL="$(ls -t "${RUN_DIR}"model_*.pt | head -1)"
cp "${LAST_MODEL}" \
  "${SHARED_CKPT_DIR}/${EXP_NAME}_$(basename "${RUN_DIR}")_$(basename "${LAST_MODEL}")"
