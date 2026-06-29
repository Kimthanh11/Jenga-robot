# Training and Visualization

## Prerequisites

Install Weights & Biases (W&B):

```bash
uv pip install wandb
```

Log in to your W&B account:

```bash
uv run wandb login
```

If you need to switch accounts or log in again:

```bash
uv run wandb login --relogin
```

---

# Training

Before training, make sure you are in the project root directory.

Export the Python path:

```bash
export PYTHONPATH="$PWD/mjlab/src:$PWD:$PYTHONPATH"
```

Verify that the Jenga task is registered correctly:

```bash
uv run --project mjlab python -c "import mjlab_jenga; print('jenga registered successfully')"
```

Start training:

```bash
uv run python -c "import mjlab_jenga, sys; sys.argv=['train','Mjlab-Jenga','--gpu-ids','None']; from mjlab.scripts.train import main; main()"
```

---

# Visualizing a Trained Policy

## Option 1 (Recommended)

The easiest way is to use the helper script, which automatically finds the latest checkpoint:

```bash
uv run python play_latest.py
```

---

## Option 2 (Manual)

### 1. Find the latest checkpoint

```bash
find logs -name "model_*.pt" | sort
```

Example:

```text
logs/rsl_rl/jenga/2026-06-29_12-26-34/model_0.pt
logs/rsl_rl/jenga/2026-06-29_12-26-34/model_50.pt
logs/rsl_rl/jenga/2026-06-29_12-26-34/model_100.pt
```

Choose the checkpoint you want to visualize (usually the latest one).

### 2. Run the policy

Replace the checkpoint path with your desired checkpoint:

```bash
uv run python -c "import mjlab_jenga, sys; sys.argv=['play','Mjlab-Jenga','--checkpoint-file','logs/rsl_rl/jenga/2026-06-29_12-26-34/model_100.pt','--num-envs','4']; from mjlab.scripts.play import main; main()"
```

After the server starts, open the URL (or localhost port) printed in the terminal to view the simulation.
