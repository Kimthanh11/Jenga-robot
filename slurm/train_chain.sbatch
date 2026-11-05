#!/bin/bash
# One link of a self-resuming training chain.
#
# A chain is several of these jobs submitted with --dependency=afterany, so training
# continues across the 12-hour wall-time limit without anyone resubmitting by hand.
# Each link resolves its starting point when it starts, not when it is submitted:
# the newest checkpoint of any earlier link if one exists, otherwise the given initial
# checkpoint. This matters because every resume writes into a fresh timestamped run
# directory, so the path of the previous link's output is unknown at submit time.
#
# Every experiment-defining flag is repeated on each link. A run folder stores only
# weights; a flag left off here would silently fall back to the module default.
#
# Usage:
#   sbatch slurm/train_chain.sbatch <stage> <match> <seed> <targets> <init_run> <init_ckpt> [iters]
#
# <match> must be a substring of the resulting run name, so later links can find the
# checkpoints of earlier ones. The run name is built as
#   low_level_<stage>_noyaw_<targetlabel>_s<seed>_<suffix>
# and the suffix passed here is "chain", so a sensible match is "s<seed>_chain".

#SBATCH --job-name=jenga_chain
#SBATCH -p stud
#SBATCH --gres=gpu:1
# The V100s on the DGX are about 40 % slower per iteration than an RTX 3080, but they
# are often free while every RTX node is taken; waiting costs more than the slowdown.
#SBATCH -C 'rtx2080|rtx3080|v100'
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm/jenga_chain-%j.out
#SBATCH --error=logs/slurm/jenga_chain-%j.out

set -euo pipefail

STAGE="${1:?stage required}"
MATCH="${2:?run-name match required}"
SEED="${3:?seed required}"
TARGETS="${4:?targets required}"
INIT_RUN="${5:?initial run folder required}"
INIT_CKPT="${6:?initial checkpoint required}"
ITERS="${7:-20000}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

LATEST=$(ls -t logs/rsl_rl/jenga/*"${MATCH}"*/model_*.pt 2>/dev/null | head -1 || true)
if [ -n "${LATEST}" ]; then
  LOAD_RUN=$(basename "$(dirname "${LATEST}")")
  LOAD_CKPT=$(basename "${LATEST}")
  ORIGIN="previous link"
else
  LOAD_RUN="${INIT_RUN}"
  LOAD_CKPT="${INIT_CKPT}"
  ORIGIN="initial checkpoint"
fi

echo "Training chain link"
echo "  stage:      ${STAGE}"
echo "  seed:       ${SEED}"
echo "  targets:    ${TARGETS}"
echo "  starts at:  ${LOAD_RUN}/${LOAD_CKPT}  (${ORIGIN})"
echo "  commit:     $(git rev-parse HEAD)"
echo "  node:       $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl

CMD=(./.venv/bin/python -m scripts.train_low_level_stage "${STAGE}"
  --iterations "${ITERS}"
  --num-envs 768
  --freeze-yaw
  --seed "${SEED}"
  --targets "${TARGETS}"
  --run-suffix chain
  --load-run "${LOAD_RUN}"
  --load-checkpoint "${LOAD_CKPT}")

echo "cmd:        ${CMD[*]}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
