# High-level Jenga policy

This document records the implementation decisions and current status of the
high-level block-selection policy. It deliberately separates the completed learner
milestone from the remaining runtime and PPO integration work.

## Objective

The high-level policy selects one physical Jenga block. A frozen low-level controller
then performs a complete extraction attempt for that block. A high-level episode keeps
the same physical tower across successful moves and ends on tower damage or when no
legal action remains.

One high-level transition is therefore a semi-Markov macro action:

```text
tower state -> block index -> complete low-level attempt -> next tower state
```

The low-level attempt may contain up to 2000 simulation control steps. The high-level
policy receives one reward after the attempt completes.

## Completed learner milestone

The first implementation milestone was developed step by step in
`mjlab_jenga/high_level_env.py` and is covered by `tests/test_high_level_env.py`.

It provides:

- a stable action-to-object mapping;
- dynamic action masks based on Jenga legality and physical presence;
- consecutive attempts without resetting a successful tower;
- injectable low-level execution, observation and reset functions;
- pilot rewards and terminal-state handling;
- a serial `reset()` / `step()` interface;
- unit tests using fake executors and observations.

The related `TargetBlockCommand.absolute_tower_state_tensor()` method provides a
GPU-resident tensor of shape `(num_envs, num_blocks, 14)`. Per-block features are:

| Offset | Feature |
|---:|---|
| 0-2 | world position `(x, y, z)` |
| 3-6 | world quaternion `(w, x, y, z)` |
| 7-9 | linear velocity `(vx, vy, vz)` |
| 10-12 | angular velocity `(wx, wy, wz)` |
| 13 | presence flag (`1.0` present, `0.0` absent) |

## Stable identities and action masks

`target_groups(cfg)["tower"]` defines one stable action identity per physical block.
For the current 9-by-3 tower this yields 27 actions ordered from `b1_1` to `b9_3`.
The identity does not change when a block moves vertically; object identity and current
pose are intentionally separate.

The policy must never compress the action list after extraction. Instead, a Boolean
mask aligned with all 27 actions is computed as:

```text
legal according to the game rule AND still physically present
```

The current base rule excludes the initial top layer, giving 24 initially legal
actions. Successfully removed blocks become masked while retaining their action index.

## Dependency interfaces

`HighLevelJengaEnv` receives three callables rather than hard-coding one controller or
representation:

```text
low_level_executor(block_name) -> (reason, control_steps)
observation_provider()         -> current tower representation
environment_resetter()         -> reset physical simulator and references
```

This allows comparisons between a scripted and learned low-level controller, and
between raw, CNN and memory-based tower representations, without rewriting the
high-level transition logic.

The low-level executor contract is important: before returning `success`, it must have
completed the settling phase, parked/removed the extracted block and rebaselined the
tower. The following high-level observation must describe a stable next decision state.

## Pilot transition semantics

The current pilot rewards are intentionally simple:

| Low-level result | Reward | Episode end |
|---|---:|---|
| `success` | `+1.0` | only if no legal action remains |
| `step_cap` | `-0.2` | no |
| `settle_failed` | `-0.2` | no |
| `tower_damage` | `-2.0` | yes |

These are engineering defaults for an initial smoke test, not final experimental
hyperparameters. They should become explicit training configuration before scientific
experiments.

## Real serial runtime

`mjlab_jenga/low_level_executor.py` now owns the reusable learned low-level macro
action. `LearnedLowLevelExecutor(block_name)` runs the policy, performs the settling
check and, on success, parks the block and rebaselines the tower before returning.

`mjlab_jenga/high_level_runtime.py` connects this executor to `HighLevelJengaEnv`.
Its initial observation is the flattened absolute tower tensor with shape
`(1, num_blocks * 14)`. The action mask stays separate in `info`; it is not a physical
tower feature.

The detailed random-baseline logger continues to call the lower-level
`extract_block()` function directly because it records the settled state before the
successful block is parked. Both code paths now use the same attempt implementation.

## PPO pilot

The first end-to-end training entry point is `scripts/train_high_level.py`. It loads a
frozen low-level checkpoint, assembles the serial macro environment and starts RSL-RL
PPO. `slurm/train_high_level.sbatch` is the corresponding cluster entry point.

RSL-RL's installed default distribution is continuous Gaussian. The project therefore
adds `mjlab_jenga/masked_categorical.py` rather than pretending a block index is a
continuous action. The actor produces one logit per stable block identity, applies the
current Boolean mask, and samples one integer index. RSL-RL storage still sees a
one-column action tensor.

This pilot is intentionally serial and uses the flattened raw tower state. It is meant
to establish correctness and collect timing/learning evidence, not to be the final
large-scale experiment.

Run the smallest real cluster smoke test with one PPO iteration as follows:

```bash
sbatch slurm/train_high_level.sbatch \
  logs/rsl_rl/jenga/<low-level-run>/model_<iteration>.pt 1 42
```

One iteration collects eight high-level macro steps by default. Since each macro step
may run the low-level policy for up to 2000 control steps, even this smoke test is a
real simulator workload. First verify that a checkpoint is written and that the log
contains multiple completed block attempts; only then increase the iteration count.

## Verification status

The local suite verifies action identities and masks, physical-state snapshots,
serial runtime wiring, categorical sampling, and one complete synthetic RSL-RL PPO
rollout/update. A CPU MuJoCo integration test also verifies simulator-state round trips.
The remaining unverified boundary is the learned low-level checkpoint running through
the complete high-level training script on a cluster GPU.

## Remaining implementation work

1. Run a serial cluster integration test with at least two consecutive real attempts.
2. Generalize target selection, references and completion handling per environment for
   efficient vectorized training.
3. Evaluate the learned selector against random selection on identical seeds and towers.
4. Replace or compare the raw state with the learned tower encoder and later memory.

## Known boundaries

- The current high-level class is serial Python control logic; it is not yet an RSL-RL
  `VecEnv`.
- The current sequential game parks successful blocks instead of placing them on top.
- `TargetBlockCommand.rebaseline()` currently requires `num_envs == 1`.
- `TargetBlockCommand.set_target_block()` currently forces the same target in every
  parallel environment.
- A complete PPO checkpoint cannot change its action count when tower size changes.
  The name generation scales with the tower configuration, but the trained network's
  output layer remains fixed.
