# Jenga Incomplete on ICC

Short guide to train, visualize the policy and check rewards.

## 1. Train

First you need to log in, then enter this directory

```bash
cd /home/share/jenga_project/Jenga-robot
```

The starting checkpoint must be here:

```text
logs/rsl_rl/jenga_incomplete/curriculum/model_1999.pt
```

If it is still in `shared_checkpoints`, copy it once:

```bash
mkdir -p logs/rsl_rl/jenga_incomplete/curriculum
cp shared_checkpoints/jenga_curriculum_model_1999.pt \
  logs/rsl_rl/jenga_incomplete/curriculum/model_1999.pt
```


Smoke test:

```bash
sbatch train_incomplete.sbatch 3 256
```

Full training:

```bash
sbatch train_incomplete.sbatch 2000 256
```

- `2000` = additional training iterations.
- `256` = Jenga environments running in parallel.
- Do not use 2000 environments; that will probably exceed GPU memory.

## 2. Monitor training

```bash
squeue -u "$USER"
```

For job `123456`, read output:

slurm_[find_job_id].out


## 3. Find saved checkpoints

```bash
ls -lht logs/rsl_rl/jenga_incomplete
```

Each training run is saved in a timestamped folder:

```text
logs/rsl_rl/jenga_incomplete/<timestamp>/
```

The completed run from 20 July is:

```text
logs/rsl_rl/jenga_incomplete/2026-07-20_22-07-21/model_5998.pt
```

## 4. See reward graphs

On the cluster, terminal 1:

```bash
cd /home/share/jenga_project/Jenga-robot
./.venv/bin/tensorboard --logdir logs/rsl_rl/jenga_incomplete \
  --port 16010 --host 127.0.0.1
```

Open <http://localhost:6010> and keep both terminals open.

To check convergence, focus on:

- `successful_extract`: should rise and stay high.
- `success_last`: should rise and stabilize.
- `block_progress_last`: should increase.
- `tower_large_pertub`: failures should become less frequent.

Do not judge the policy only by total reward.

## 5. Visualize the policy

```bash
 uv run python play_latest_incomplete.py  
```