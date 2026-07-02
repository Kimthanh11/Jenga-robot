import os
import re
import sys
from pathlib import Path

import mjlab_jenga
from mjlab.scripts.play import main

LOG_DIR = Path("logs/rsl_rl/jenga")


def get_iteration(path):
    m = re.search(r"model_(\d+)\.pt", path.name)
    return int(m.group(1)) if m else -1


# Get all runs (newest first)
runs = sorted(LOG_DIR.glob("*"), key=os.path.getmtime, reverse=True)

if not runs:
    raise RuntimeError(f"No runs found in {LOG_DIR}")

print("\nAvailable runs:\n")
for i, run in enumerate(runs):
    ckpts = sorted(run.glob("model_*.pt"), key=get_iteration)
    last_iter = get_iteration(ckpts[-1]) if ckpts else "No checkpoint"
    print(f"[{i}] {run.name}   (latest model: {last_iter})")

# Select run
run_idx = int(input("\nChoose run number: "))
selected_run = runs[run_idx]

# Get checkpoints in that run
ckpts = sorted(selected_run.glob("model_*.pt"), key=get_iteration)

if not ckpts:
    raise RuntimeError("No checkpoints found in selected run.")

print("\nAvailable checkpoints:\n")
for i, ckpt in enumerate(ckpts):
    print(f"[{i}] {ckpt.name}")

choice = input(
    "\nCheckpoint number (Enter for latest): "
).strip()

if choice == "":
    selected_ckpt = ckpts[-1]
else:
    selected_ckpt = ckpts[int(choice)]

print(f"\nUsing checkpoint: {selected_ckpt}\n")

sys.argv = [
    "play",
    "Mjlab-Jenga",
    "--checkpoint-file",
    str(selected_ckpt),
    "--num-envs",
    "1",
]

main()