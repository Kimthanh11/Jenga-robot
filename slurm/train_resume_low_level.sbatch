#!/bin/bash
# Resume an explicit Jenga low-level stage.
#
# A resume must repeat every flag that defines the experiment. The run folder records
# only weights, so anything omitted here silently reverts to the module default: the
# targets fall back to RANDOM_TARGET_BLOCK_NAMES and yaw thaws, which is a different
# task than the one being continued.
#
# Usage:
#   sbatch slurm/train_resume_low_level.sbatch <stage> <run_folder> <checkpoint> \
#          [additional_iterations] [num_envs] [freeze_yaw] [targets] [abort]
#
# Example, continuing a frozen-yaw run over all legal targets:
#   sbatch slurm/train_resume_low_level.sbatch missing3 \
#          2026-09-08_20-57-55_low_level_missing3_noyaw_legal model_9700.pt \
#          20000 768 freeze legal

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
FREEZEYAW="${6:-}"
TARGETS="${7:-}"
ABORT="${8:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

echo "Resuming Jenga low-level stage"
echo "  stage:      ${STAGE}"
echo "  load_run:   ${LOAD_RUN}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  iterations: ${ITERS}"
echo "  num_envs:   ${NUM_ENVS}"
echo "  freeze_yaw: ${FREEZEYAW:-no}"
echo "  targets:    ${TARGETS:-module default}"
echo "  abort:      ${ABORT:-no}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

CMD=(./.venv/bin/python -m scripts.train_low_level_stage "${STAGE}"
  --iterations "${ITERS}"
  --num-envs "${NUM_ENVS}"
  --load-run "${LOAD_RUN}"
  --load-checkpoint "${CHECKPOINT}")
if [ -n "${FREEZEYAW}" ]; then CMD+=(--freeze-yaw); fi
if [ -n "${TARGETS}" ]; then CMD+=(--targets "${TARGETS}"); fi
if [ -n "${ABORT}" ]; then CMD+=(--abort); fi

echo "cmd:        ${CMD[*]}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
