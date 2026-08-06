import os
import re
import sys
from pathlib import Path

import mjlab_jenga
from mjlab.scripts.play import main


TASK_ID = "Mjlab-Jenga-Navpull"
LOG_DIR = Path("logs/rsl_rl/jenga_navpull")


def get_iteration(checkpoint_path: Path) -> int:
    """Extract the iteration number from names such as model_2000.pt."""
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)

    if match is None:
        return -1

    return int(match.group(1))


def find_latest_checkpoint(log_dir: Path) -> Path:
    """Find the newest run folder and its highest-iteration checkpoint."""
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir.resolve()}")

    run_directories = [
        path
        for path in log_dir.iterdir()
        if path.is_dir()
    ]

    if not run_directories:
        raise RuntimeError(f"No training runs found in: {log_dir.resolve()}")

    # Use the most recently modified run directory.
    run_directories.sort(
        key=os.path.getmtime,
        reverse=True,
    )

    for run_directory in run_directories:
        checkpoints = [
            path
            for path in run_directory.glob("model_*.pt")
            if get_iteration(path) >= 0
        ]

        if not checkpoints:
            continue

        checkpoints.sort(key=get_iteration)
        latest_checkpoint = checkpoints[-1]

        print("Using run:", run_directory)
        print("Using checkpoint:", latest_checkpoint)

        return latest_checkpoint

    raise RuntimeError(
        f"No model_*.pt checkpoints found inside: {log_dir.resolve()}"
    )


def run() -> None:
    checkpoint = find_latest_checkpoint(LOG_DIR)

    sys.argv = [
        "play",
        TASK_ID,
        "--checkpoint-file",
        str(checkpoint),
        "--num-envs",
        "1",
    ]

    main()


if __name__ == "__main__":
    run()