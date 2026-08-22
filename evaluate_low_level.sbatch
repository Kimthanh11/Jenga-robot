#!/bin/bash
# Evaluate a checkpoint across targets and missing-block levels.
#
# Usage:
#   sbatch evaluate_low_level.sbatch logs/rsl_rl/jenga/<run>/model_9999.pt
#   sbatch evaluate_low_level.sbatch logs/rsl_rl/jenga/<run>/model_9999.pt b6_1,b7_1 0,1

#SBATCH --job-name=jenga_eval
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 02:00:00
#SBATCH --output=logs/slurm/jenga_eval-%j.out
#SBATCH --error=logs/slurm/jenga_eval-%j.out

set -euo pipefail

CHECKPOINT="${1:?checkpoint path required}"
TARGETS="${2:-all}"
LEVELS_CSV="${3:-0,1,2,3}"
EPISODES="${4:-64}"
NUM_ENVS="${5:-64}"
LEGACY="${6:-}"
IMPRATIO="${7:-}"
CONE="${8:-}"   # pass "legacy" for checkpoints from before the std fix

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm logs/eval

echo "commit:  $(git rev-parse HEAD)"
echo "cwd:     $PWD"
echo "python:  $(./.venv/bin/python -c 'import sys; print(sys.executable)')"
echo "cfg:     $(./.venv/bin/python -c 'import mjlab_jenga.jenga_mjenv_cfg as c; print(c.__file__)')"

IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"
OUT="logs/eval/eval-${SLURM_JOB_ID}.csv"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python evaluate_low_level.py \
  --checkpoint "${CHECKPOINT}" \
  --targets "${TARGETS}" \
  --missing-levels "${LEVELS[@]}" \
  --episodes "${EPISODES}" \
  --num-envs "${NUM_ENVS}" \
  --csv "${OUT}" \
  ${LEGACY:+--legacy-distribution} \
  ${IMPRATIO:+--impratio="${IMPRATIO}"} \
  ${CONE:+--cone="${CONE}"}
