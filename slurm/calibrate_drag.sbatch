#!/bin/bash
# Contact and friction calibration sweep.
#
# Usage:
#   sbatch slurm/calibrate_drag.sbatch
#   sbatch slurm/calibrate_drag.sbatch "0.10,0.20,0.30,0.40" "1.0,10.0,30.0" "pyramidal,elliptic"

#SBATCH --job-name=jenga_calib
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 04:00:00
#SBATCH --output=logs/slurm/jenga_calib-%j.out
#SBATCH --error=logs/slurm/jenga_calib-%j.out

set -euo pipefail

FRICTIONS="${1:-0.10,0.15,0.20,0.28,0.38,0.48}"
IMPRATIO="${2:-1.0,10.0,30.0}"
CONE="${3:-pyramidal,elliptic}"
TARGETS="${4:-b6_1,b6_3,b3_1,b7_1}"
SOLREF="${5:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm logs/feasibility

echo "commit:      $(git rev-parse HEAD)"
echo "cwd:         $PWD"
echo "python:      $(./.venv/bin/python -c 'import sys; print(sys.executable)')"
echo "mujoco:      $(./.venv/bin/python -c 'import mujoco; print(mujoco.__version__)')"
echo "frictions:   ${FRICTIONS}"
echo "impratio:    ${IMPRATIO}"
echo "cone:        ${CONE}"
echo "targets:     ${TARGETS}"
echo "solref:      ${SOLREF:-default}"

export WANDB_MODE=disabled
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python -m scripts.calibrate_drag \
  --targets="${TARGETS}" \
  --frictions="${FRICTIONS}" \
  --impratio="${IMPRATIO}" \
  --cone="${CONE}" \
  ${SOLREF:+--solref="${SOLREF}"} \
  --max-steps 400 \
  --device "cuda:0" \
  --csv "logs/feasibility/calib_drag-${SLURM_JOB_ID:-local}.csv"
