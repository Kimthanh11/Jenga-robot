#!/bin/bash
# Resume Mjlab-Jenga training for yaw phase on the IAS cluster.
#
# Usage from repo root:
#   sbatch train_resume_yaw.sbatch
#   sbatch train_resume_yaw.sbatch 2000 256
#   sbatch train_resume_yaw.sbatch 2000 256 2026-07-23_12-34-56 model_1999.pt
#
# Args:
#   1: additional iterations, default 2000
#   2: num envs, default 256
#   3: run folder under logs/rsl_rl/jenga, optional
#   4: checkpoint file inside that run folder, optional

#SBATCH --job-name=jenga_yaw
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 04:00:00
#SBATCH --output=logs/slurm/jenga_yaw-%j.out
#SBATCH --error=logs/slurm/jenga_yaw-%j.out

set -euo pipefail

ITERS="${1:-2000}"
NUM_ENVS="${2:-256}"
LOAD_RUN="${3:-}"
LOAD_CHECKPOINT="${4:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

if [[ -z "${LOAD_RUN}" || -z "${LOAD_CHECKPOINT}" ]]; then
  LATEST_MODEL="$(
    find logs/rsl_rl/jenga -name 'model_*.pt' \
      ! -name 'model_0.pt' \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )"

  if [[ -z "${LATEST_MODEL}" ]]; then
    echo "No checkpoint found under logs/rsl_rl/jenga."
    exit 1
  fi

  LOAD_RUN="$(basename "$(dirname "${LATEST_MODEL}")")"
  LOAD_CHECKPOINT="$(basename "${LATEST_MODEL}")"
fi

echo "Resuming Mjlab-Jenga"
echo "  run:        ${LOAD_RUN}"
echo "  checkpoint: ${LOAD_CHECKPOINT}"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run "${LOAD_RUN}" \
  --agent.load-checkpoint "${LOAD_CHECKPOINT}" \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
