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

## Remaining implementation work

1. Extract the reusable learned low-level attempt from
   `scripts/evaluate_selection.py` into `mjlab_jenga/low_level_executor.py`.
2. Wire the real `TargetBlockCommand` tensor provider and physical reset operation.
3. Run a serial cluster integration test with at least two consecutive real attempts.
4. Define the first PPO observation (raw flattened state versus learned encoder).
5. Add a masked categorical action distribution. The installed RSL-RL stack currently
   exposes Gaussian continuous actions only.
6. Add a high-level runner/configuration and SLURM entry point.
7. Generalize target selection, references and completion handling per environment for
   efficient vectorized training.
8. Evaluate against the random selector on identical seeds and towers.

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
