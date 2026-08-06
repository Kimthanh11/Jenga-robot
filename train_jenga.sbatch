#!/bin/bash
#SBATCH -J jenga_rl
#SBATCH -t 04:00:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm/%x-%j.out

cd /home/stud_ohmer/Jenga-robot

mkdir -p logs/slurm

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

uv run --no-sync python -u -c "import mjlab_jenga; from mjlab.scripts.train import launch_training, TrainConfig; cfg = TrainConfig.from_task('Mjlab-Jenga'); launch_training('Mjlab-Jenga', cfg)"
