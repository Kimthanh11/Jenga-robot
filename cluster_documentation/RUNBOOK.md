# Runbook — training Jenga-robot on the ICC

Practical steps for running *this* project on the TU-Darmstadt IAS Compute Cluster
(ICC). For general cluster docs see the other files in this folder.

- Login node: `ssh stud_uttsal@mn.ias.informatik.tu-darmstadt.de`
- Project (shared): `/home/share/jenga_project/Jenga-robot`
- Scheduler: SLURM. **Never run training on the login node `mn`** — only via `sbatch`/`srun`.

We run training in a **uv-managed venv** (`.venv/`), executed **directly on the
compute node** — no Docker. (Docker on the cluster puts the container filesystem in
RAM, and baking torch+CUDA into an image OOM's the login node; a native venv writes
to disk and is simpler. SLURM's `--gres` also sets `CUDA_VISIBLE_DEVICES` for us,
which is exactly what mjlab reads to pick the GPU.)

## One-time setup

### 1. Install uv (once per user)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
The installer adds `. "$HOME/.local/bin/env"` to your `~/.bashrc`, so `uv` is on PATH
in **new** shells. In your current shell, run `source ~/.local/bin/env` (or re-login)
— otherwise `uv` "command not found".

### 2. Create the venv
```bash
cd /home/share/jenga_project/Jenga-robot
uv sync --frozen        # builds .venv/ from uv.lock (~400 pkgs incl. torch+CUDA wheels)
```
`.venv/` is gitignored and lives on shared NFS, so it's visible on every compute node.

## Updating

**Pull code changes**
```bash
cd /home/share/jenga_project/Jenga-robot
git fetch origin && git checkout timur && git pull --ff-only
```

**After dependencies change** (someone edited `pyproject.toml` / `uv.lock`)
```bash
uv sync --frozen        # re-syncs .venv to the lockfile
```
Rebuild the venv from scratch if it gets corrupted: `rm -rf .venv && uv sync --frozen`.

> Note: `train.py` → `mjlab_jenga` builds the whole MuJoCo scene in code
> (`MjSpec.from_string`); it does **not** read `jenga.xml`. `generate_tower.py` /
> `jenga.xml` belong to the old standalone `pull_hook.py` path — irrelevant here.

## Running training

Full run (defaults: 1000 iterations, 4096 envs):
```bash
cd /home/share/jenga_project/Jenga-robot
sbatch train.sbatch
```
Quick smoke test (3 iterations, 256 envs):
```bash
sbatch train.sbatch 3 256
# or interactively, streaming to your terminal:
srun -p stud --gres=gpu:1 -C 'rtx2080|rtx3080' -c 4 --mem-per-cpu=4G -t 0:20:00 \
  bash -lc 'PYTHONUNBUFFERED=1 ./.venv/bin/python train.py Mjlab-Jenga \
            --agent.logger tensorboard --agent.max-iterations 3 --env.scene.num-envs 256'
```

**GPU gotcha:** our torch (`2.12+cu130`) ships **no kernels for compute capability
7.0**, so it can't use the **V100** (`dgx-station`). `train.sbatch` pins
`-C 'rtx2080|rtx3080'` (CC 7.5 / 8.6) to avoid it. Symptom if you hit a V100:
`UserWarning: Found GPU0 Tesla V100 ... compute capability (CC) 7.0` then failures.

## Monitoring

```bash
squeue -u $USER                       # your jobs -> JOBID, node, state
csinfo                                # live free CPU/RAM/GPU per node
ias-job-nvidia-smi JOBID              # live GPU utilization of your job
ias-job-top JOBID                     # live CPU/RAM of your job
tail -f slurm-JOBID.out               # live training log (per-iteration rewards)
scancel JOBID                         # stop a job
```
Logs/checkpoints: `logs/rsl_rl/jenga/<timestamp>/`. Reward terms to watch:
`Episode_Reward/block_dx`, `Episode_Reward/extraction_success`,
`Episode_Termination/time_out` — success = extraction_success climbs above 0.

**TensorBoard** (from your laptop):
```bash
ssh -L 10000:127.0.0.1:6006 stud_uttsal@mn.ias.informatik.tu-darmstadt.de
# then on mn:
cd /home/share/jenga_project/Jenga-robot && tensorboard --port 6006 --logdir=logs
# open http://127.0.0.1:10000
```

## Reference: throughput
rtx3080, 256 envs: **~2000 env-steps/s** (vs ~21 on an Apple M4 — ~100×).
