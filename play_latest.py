import os
import re
import sys
from pathlib import Path

import mjlab_jenga
from mjlab.scripts.play import main

log_dir = Path("logs/rsl_rl/jenga")

runs = sorted(log_dir.glob("*"), key=os.path.getmtime, reverse=True)
latest_run = runs[0]

def get_iteration(path):
    m = re.search(r"model_(\d+)\.pt", path.name)
    return int(m.group(1)) if m else -1

ckpts = sorted(latest_run.glob("model_*.pt"), key=get_iteration)
last_ckpt = ckpts[-1]

print("Using checkpoint:", last_ckpt)

sys.argv = [
    "play",
    "Mjlab-Jenga",
    "--checkpoint-file",
    str(last_ckpt),
    "--num-envs",
    "1",
]

main()