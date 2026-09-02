#!/bin/bash
# Resume Mjlab-Jenga curriculum training from an existing checkpoint.
#
# Usage from repo root:
#   sbatch train_resume_curriculum.sbatch <run_folder> <checkpoint>
#   sbatch train_resume_curriculum.sbatch <run_folder> <checkpoint> 4000 256
#
# Example:
#   sbatch train_resume_curriculum.sbatch 2026-08-12_23-27-05 model_9499.pt
#
# Args:
#   1: run folder under logs/rsl_rl/jenga, required
#   2: checkpoint file, required
#   3: additional iterations, default 4000
#   4: num envs, default 256

#SBATCH --job-name=jenga_resume
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_resume-%j.out
#SBATCH --error=logs/slurm/jenga_resume-%j.out

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch train_resume_curriculum.sbatch <run_folder> <checkpoint> [iterations] [num_envs]" >&2
  exit 2
fi

LOAD_RUN="$1"
CHECKPOINT="$2"
ITERS="${3:-4000}"
NUM_ENVS="${4:-256}"

if [[ "${LOAD_RUN}" == *low_level_* ]]; then
  echo "Refusing to resume a low-level stage with train_resume_curriculum.sbatch." >&2
  echo "Use instead:" >&2
  echo "  sbatch train_resume_low_level.sbatch <stage> ${LOAD_RUN} ${CHECKPOINT} ${ITERS} ${NUM_ENVS}" >&2
  echo "Example:" >&2
  echo "  sbatch train_resume_low_level.sbatch target ${LOAD_RUN} ${CHECKPOINT} ${ITERS} ${NUM_ENVS}" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

echo "Resuming Mjlab-Jenga curriculum"
echo "  load_run:   ${LOAD_RUN}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run "${LOAD_RUN}" \
  --agent.load-checkpoint "${CHECKPOINT}" \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
