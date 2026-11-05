#!/bin/bash
# Resume an explicit Jenga low-level stage.
#
# Usage:
#   sbatch slurm/train_resume_low_level.sbatch <stage> <run_folder> <checkpoint> [additional_iterations] [num_envs]
#
# Example:
#   sbatch slurm/train_resume_low_level.sbatch target 2026-08-19_12-24-50_low_level_target model_2500.pt 9500 768

#SBATCH --job-name=jenga_low
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_low-%j.out
#SBATCH --error=logs/slurm/jenga_low-%j.out

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch slurm/train_resume_low_level.sbatch <stage> <run_folder> <checkpoint> [additional_iterations] [num_envs]" >&2
  exit 2
fi

STAGE="$1"
LOAD_RUN="$2"
CHECKPOINT="$3"
ITERS="${4:-4000}"
NUM_ENVS="${5:-768}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

echo "Resuming Jenga low-level stage"
echo "  stage:      ${STAGE}"
echo "  load_run:   ${LOAD_RUN}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python -m scripts.train_low_level_stage "${STAGE}" \
  --iterations "${ITERS}" \
  --num-envs "${NUM_ENVS}" \
  --load-run "${LOAD_RUN}" \
  --load-checkpoint "${CHECKPOINT}"
