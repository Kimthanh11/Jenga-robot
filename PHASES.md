# Training phases — Jenga-robot (PPO baseline → Fazeli)

Experiment log for the RL policy that extracts a Jenga block. Baseline is Fazeli's
weak PPO baseline (MLP 64×64, target block `b6_1`, privileged block pos/vel). We move
it toward Fazeli's full reward + a tower-state feature, one phase at a time.

Reference reward (Fazeli et al., "See, Feel, Act", Suppl.):
`R = Σ_t [ b1·dx_t − b2·D1(p_t) − b3·D2(p_t) ] + b4·D3`, gains `b=(0.1, 0.2, 100, 4)`,
where `dx` = block displacement along its major axis, `p` = tower perturbation
(p=0.4 ↔ ~20 mm / 30°), `D1` moderate (linear on p∈(0.08,0.4)), `D2` large (=100 for
p>0.4), `D3` = terminal extraction. Everything lives in `mjlab_jenga/jenga_mjenv_cfg.py`.

Run any phase on the cluster with `sbatch train.sbatch <iters> <num_envs>` (see
`cluster_documentation/RUNBOOK.md`). rtx3080 fits ~512 envs; rtx2080 ~512 too.

---

## Phase 1 — normalization + Fazeli reward (learn to PUSH)
**Goal:** a policy that actually pushes the block out. Fix the "do nothing" local
optimum from the old reward.
**Changes:**
- `obs_normalization=True` on actor+critic (`EmpiricalNormalization`) — meeting note
  "normalise features, otherwise unstable".
- Reward switched to Fazeli's push term: `block_dx` = per-step displacement along −x
  (velocity·dt), weight `b1=0.1` — replaces the old `block_progress` (which integrated
  absolute position and rewarded *holding* a displaced block).
- `extraction_success` terminal bonus, weight `b4=4`.
- `torque_penalty` / `action_rate` kept but weight **0** (Fazeli's RL reward has no
  effort term; they were the source of the "freeze" optimum). Not deleted — reusable.
**Config:** rewards `{block_dx:0.1, extraction_success:4, torque_penalty:0, action_rate:0}`.
**Status:** running as the push-only **baseline** (job on rtx3080). Watching whether
`Episode_Reward/block_dx` and `extraction_success` climb above 0.

## Phase 2 — tower-perturbation penalty + topple termination (Fazeli D1/D2 + curriculum)
**Goal:** full Fazeli reward, but stability introduced *after* the policy can push
(meeting note: "small stability penalty first, then higher; phases/warmup/scheduler").
**Changes:**
- `tower_perturbation(p)`: horizontal shift of the tower CoM (26 blocks, excluding
  target) from rest, scaled so 20 mm → 0.4. x,y only → robust to vertical settling.
- Rewards `perturb_moderate` (Fazeli D1, linear on (0.08,0.4)) and `perturb_large`
  (D2 = 100 for p>0.4).
- Termination `topple` (**true** termination, `time_out=False`) when p>threshold —
  distinct from the existing `time_out` **truncation** (meeting note: "truncation, not
  only termination").
- **Curriculum** (`common_step_counter = iters×32`):
  | term | step 0 | ~iter 300 | ~iter 500 |
  |---|---|---|---|
  | perturb_moderate weight | 0 | −0.1 | −0.2 (b2) |
  | perturb_large weight | 0 | −100 (b3) | −100 |
  | topple threshold | ∞ | 0.4 | 0.4 |
**Status:** running (rtx2080, job `jenga-p2`). Plumbing validated on GPU (curriculum
manager active, terms logging). Reward reaches full Fazeli gains by ~iter 500.

## Phase 3 — tower CoM as an observation feature (cheapest upgrade)
**Goal:** let the policy *see* the tower's overall state and pre-empt leaning, instead
of only being penalized after the fact (meeting note: "CoM excluding the moving block
as a feature").
**Changes:**
- `tower_com_deviation`: tower CoM (excluding target) minus its rest pose, 3-D, added
  to the actor+critic observation group. Deviation (not absolute CoM) → centered ~0 at
  reset, clean for obs-normalization. Metric reused from Phase 2 (free).
- Observation dim 11 → 14.
**Status:** launched as a job (builds on the Phase 2 reward/curriculum).

---

## Meeting-notes coverage
| Note | Status |
|---|---|
| Normalise features | ✅ Phase 1 |
| CoM (excl. moving block) as a feature | ✅ Phase 3 (as obs); reward use in Phase 2 |
| Jenga-paper reward/losses | ✅ Phase 1+2 (Fazeli reward 1:1) |
| Truncation, not only termination | ✅ Phase 2 (`time_out` trunc + `topple` term) |
| Small stability penalty first, then higher / phases | ✅ Phase 2 curriculum |
| Learn to push first, then stability | 🟡 structurally done; awaiting confirmation push is learned |
| depth = pusher→block distance (Lidar/stereo) | ⬜ deferred (perception layer, APPLE) |
| Predict abstract features via sensors | ⬜ deferred (we use privileged sim features) |
| Combined loss | ⬜ deferred (needs algorithm change, APPLE direction) |
| Point cloud of the tower | ⬜ deferred (CoM is the scalar summary) |
