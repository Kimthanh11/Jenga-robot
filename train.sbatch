#!/bin/bash
# Train Jenga-robot on the ICC using the in-repo uv venv (.venv). Submit from the
# repo root on `mn`:
#   sbatch train.sbatch                 # full run (defaults below)
#   sbatch train.sbatch 3 256           # quick test: 3 iterations, 256 envs
# Prereq: the venv exists (see cluster_documentation/RUNBOOK.md -> `uv sync`).
#SBATCH --job-name=jenga-train
#SBATCH -p stud
#SBATCH --gres=gpu:1
# Our torch (2.12+cu130) has NO kernels for compute-capability 7.0, so exclude the
# V100 (dgx-station). rtx2080=CC7.5 and rtx3080=CC8.6 are both supported.
#SBATCH -C 'rtx2080|rtx3080'
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail

ITERS="${1:-1000}"
NUM_ENVS="${2:-4096}"

cd /home/share/jenga_project/Jenga-robot

# SLURM sets CUDA_VISIBLE_DEVICES (non-empty) via --gres, which is exactly what
# mjlab reads to select the GPU (it falls back to CPU when that var is empty/unset).
# PYTHONUNBUFFERED so the per-iteration reward logs stream live into the .out file.
PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga \
  --agent.logger tensorboard \
  --agent.max-iterations "${ITERS}" \
  --env.scene.num-envs "${NUM_ENVS}"
