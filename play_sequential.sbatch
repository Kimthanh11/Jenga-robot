#!/bin/bash
# Sequential-play experiment: how many blocks come out of one tower in a row.
#
# Usage:
#   sbatch play_sequential.sbatch <checkpoint> [games] [order] [extra flags]
#
#   sbatch play_sequential.sbatch logs/rsl_rl/jenga/<run>/model_8500.pt
#   sbatch play_sequential.sbatch <ckpt> 20 "b9_1,b2_1,b3_1,b9_3,b2_3,b9_2,b2_2"
#   sbatch play_sequential.sbatch <ckpt> 20 "" --cumulative

#SBATCH --job-name=jenga_seq
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -t 02:00:00
#SBATCH --output=logs/slurm/jenga_seq-%j.out
#SBATCH --error=logs/slurm/jenga_seq-%j.out

set -euo pipefail

CHECKPOINT="${1:?checkpoint path required}"
GAMES="${2:-20}"
ORDER="${3:-}"
shift 3 2>/dev/null || shift $# 

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm logs/eval

echo "commit:     $(git rev-parse HEAD)"
echo "python:     $(./.venv/bin/python -c 'import sys; print(sys.executable)')"
echo "checkpoint: ${CHECKPOINT}"
echo "games:      ${GAMES}"
echo "order:      ${ORDER:-default}"

export WANDB_MODE=disabled
export MUJOCO_GL=egl

CMD=(./.venv/bin/python play_sequential.py
  --checkpoint "${CHECKPOINT}"
  --games "${GAMES}"
  --device cuda:0
  --csv "logs/eval/sequential-${SLURM_JOB_ID:-local}.csv")
if [ -n "${ORDER}" ]; then CMD+=(--order "${ORDER}"); fi
CMD+=("$@")

echo "cmd:        ${CMD[*]}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
