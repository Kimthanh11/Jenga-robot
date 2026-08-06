#!/bin/bash
cd /home/share/jenga_project/Jenga-robot
BASE=/home/share/jenga_project/Jenga-robot/logs/rsl_rl
exec .venv/bin/tensorboard \
  --logdir_spec "phase1:${BASE}/jenga/2026-07-06_17-06-09,randblock:${BASE}/jenga_randblock" \
  --port 6007 --host 127.0.0.1 --reload_interval 15
