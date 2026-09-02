#!/bin/bash
# Scripted extraction feasibility sweep across targets, contact points, friction, seeds.
#
# Measures which targets this scripted controller can extract. Failure does not prove
# that no feedback controller or learned policy can extract the block.
#
# Usage:
#   sbatch slurm/feasibility_sweep.sbatch
#   sbatch slurm/feasibility_sweep.sbatch legal "-0.5,0.0,0.5" "0.20,0.38" "1,2,3"

#SBATCH --job-name=jenga_feas
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 04:00:00
#SBATCH --output=logs/slurm/jenga_feas-%j.out
#SBATCH --error=logs/slurm/jenga_feas-%j.out

set -euo pipefail

TARGETS="${1:-trained}"
CONTACT_Z="${2:-0.0}"
FRICTIONS="${3:-}"
SEEDS="${4:-42}"
IMPRATIO="${5:-}"
CONE="${6:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm logs/feasibility

echo "commit:  $(git rev-parse HEAD)"
echo "cwd:     $PWD"
echo "python:  $(./.venv/bin/python -c 'import sys; print(sys.executable)')"
echo "mujoco:  $(./.venv/bin/python -c 'import mujoco; print(mujoco.__version__)')"
echo "targets: ${TARGETS} | contact_z: ${CONTACT_Z} | frictions: ${FRICTIONS:-sampled} | seeds: ${SEEDS}"
echo "impratio: ${IMPRATIO:-default} | cone: ${CONE:-default}"

export WANDB_MODE=disabled
export MUJOCO_GL=egl

PYTHONUNBUFFERED=1 ./.venv/bin/python -m scripts.feasibility_sweep \
  --targets="${TARGETS}" \
  --contact-z="${CONTACT_Z}" \
  --seeds="${SEEDS}" \
  ${FRICTIONS:+--frictions="${FRICTIONS}"} \
  ${IMPRATIO:+--impratio="${IMPRATIO}"} \
  ${CONE:+--cone="${CONE}"} \
  --max-steps 1400 \
  --device "cuda:0" \
  --csv "logs/feasibility/feas-${SLURM_JOB_ID:-local}.csv"
