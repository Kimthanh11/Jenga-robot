"""Calibrate MuJoCo contact parameters using the drag ratio of the Jenga tower.

The drag ratio quantifies how strongly the tower is dragged along when a block is
pushed out of it:

    drag ratio = max horizontal displacement over all non-target blocks
                 / horizontal displacement of the target block

sampled once the target has travelled a fixed reference distance.

A physical tower yields a small ratio. The block resting on the layer above spans
three stones, so friction from the target acts on roughly one third of its underside
while the remaining two thirds hold it under the same load.

The ratio bounds which blocks are extractable. A safe success requires every
non-target block to stay within TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT (12 mm)
while the target travels the full success distance (112.5 mm), so the feasibility
threshold is 12 / 112.5 = 0.107. Measured ratios reproduce the outcome of the
scripted sweep in feasibility_sweep.py.

Measured parameter sensitivities, pyramidal cone, default solref:

    impratio    1     drag 0.664   (0.896 at mu = 0.48)
    impratio   10     drag 0.302
    impratio   30     drag 0.251   (0.194 at mu = 0.48)

`impratio` is the stiffness of friction constraints relative to normal ones. At the
MuJoCo default of 1 both are equally compliant and the stack shears elastically. The
friction coefficient shifts the ratio by up to a factor of two, but the direction
depends on `impratio` -- rising with mu at impratio 1, falling at impratio 30 -- so it
cannot compensate for an incorrect stiffness. The elliptic cone yields ratios above
1.9 at every stiffness and destroys the tower during the reset settle in half of all
runs.

Emits one CSV row per (target, cone, impratio, solref, friction, seed) combination.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _append_row(csv_path, row) -> None:
    """Append one result row to the CSV, creating the file and header if needed.

    Rows are written as they are produced rather than buffered until the end, so that
    a run terminated by its scheduler time limit still yields the cases it completed.

    Args:
        csv_path: Destination path, or None to disable writing.
        row: Mapping of column name to value; its keys define the header.
    """
    if csv_path is None:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()), lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate contact physics via drag ratio.")
    parser.add_argument(
        "--targets",
        default="b6_1,b6_3,b3_1,b7_1",
        help="Targets to average over. Top-layer blocks have nothing above them and "
        "therefore carry no information about drag.",
    )
    parser.add_argument(
        "--frictions",
        default="0.10,0.15,0.20,0.28,0.38,0.48",
        help="Sliding friction values to sweep. Current config samples 0.28-0.48.",
    )
    parser.add_argument("--impratio", default="1.0", help="Comma-separated impratio values.")
    parser.add_argument(
        "--solref",
        default="",
        help="Comma-separated solref timeconst values for the block geoms (dampratio "
        "fixed at 1). Empty keeps MuJoCo's default of 0.02. Lower is stiffer; the "
        "floor is 2 * timestep = 0.004.",
    )
    parser.add_argument("--cone", default="pyramidal", help="pyramidal and/or elliptic.")
    parser.add_argument(
        "--reference-progress",
        type=float,
        default=0.020,
        help="Target displacement (m) at which the drag ratio is sampled.",
    )
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--success-fraction",
        type=float,
        default=0.75,
        help="Fraction of the block length that counts as extracted, held constant. "
        "0.75 = 112.5 mm, the full task.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    import mjlab_jenga.jenga_mjenv_cfg as cfg

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict] = []

    solrefs: list[float | None] = _parse_floats(args.solref) or [None]

    for solref in solrefs:
      for cone in [c.strip() for c in args.cone.split(",") if c.strip()]:
        for impratio in _parse_floats(args.impratio):
            # solref is baked into the block XML, so it needs a fresh spec each time.
            cfg.BLOCK_SOLREF = None if solref is None else (solref, 1.0)
            cfg.apply_low_level_stage("fixed")
            cfg.YAW_CURRICULUM_START = cfg.YAW_CURRICULUM_END = 0.0
            cfg.SUCCESS_CURRICULUM_START = args.success_fraction
            cfg.SUCCESS_CURRICULUM_END = args.success_fraction
            env_cfg = cfg.jenga_env_cfg(play=True)
            env_cfg.scene.num_envs = len(targets)
            env_cfg.auto_reset = False
            env_cfg.commands["target_block"].force_target_names = tuple(targets)
            env_cfg.sim.mujoco.cone = cone
            env_cfg.sim.mujoco.impratio = impratio

            env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
            try:
                action = torch.zeros(
                    (len(targets), env.action_manager.total_action_dim), device=env.device
                )
                action[:, 0] = -1.0
                for friction in _parse_floats(args.frictions):
                    cfg.RESET_FRICTION_SLIDING_RANGE = (friction, friction)
                    for seed in seeds:
                        rows.extend(
                            _measure(env, cfg, torch, action, targets,
                                     cone, impratio, friction, seed, args, solref)
                        )
            finally:
                env.close()

    if args.csv is not None:
        print(f"Wrote {Path(args.csv)} ({len(rows)} rows)", flush=True)


def _measure(env, cfg, torch, action, targets, cone, impratio, friction, seed, args, solref=None):
    num_envs = len(targets)
    env.reset(seed=seed)
    cmd = env.command_manager.get_term("target_block")
    actual = [cmd._all_names[i] for i in cmd.selected_block_idx.tolist()]
    if actual != targets:
        raise RuntimeError(f"target assignment failed: {actual} != {targets}")

    names = cmd._all_names
    start = cmd._start_pos
    env_arange = torch.arange(num_envs, device=env.device)
    sel = cmd.selected_block_idx.clone()

    sampled = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    drag = torch.zeros(num_envs, device=env.device)
    drag_block = [""] * num_envs
    target_at_sample = torch.zeros(num_envs, device=env.device)
    damaged_first = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    steps_taken = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    # A sampled environment is frozen by zeroing its action rather than allowed to
    # terminate and reset. Each reset invokes randomize_block_physics, which rewrites
    # geom_friction and pseudo_inertia for all 27 blocks and triggers a full model
    # constant recompute; repeated resets dominate the runtime of the sweep.
    live_action = action.clone()

    for step in range(1, args.max_steps + 1):
        env.step(live_action)
        pos = torch.stack([b.data.body_link_pos_w[:, 0, :] for b in cmd._blocks], dim=0)
        shift = torch.norm((pos - start.unsqueeze(1))[:, :, :2], dim=-1)     # [P, N]
        target_shift = shift[sel, env_arange]                                # [N]

        others = shift.clone()
        others[sel, env_arange] = -1.0
        other_max, other_idx = others.max(dim=0)

        damaged = cfg.tower_damage(env)
        reached = (target_shift >= args.reference_progress) & ~sampled
        closing = (reached | (damaged & ~sampled))
        if bool(closing.any().item()):
            for i in closing.nonzero(as_tuple=False).squeeze(-1).tolist():
                drag_block[i] = names[int(other_idx[i].item())]
            ratio = other_max / target_shift.clamp_min(1e-9)
            drag = torch.where(closing, ratio, drag)
            target_at_sample = torch.where(closing, target_shift, target_at_sample)
            damaged_first |= damaged & closing & ~reached
            steps_taken = torch.where(closing, torch.full_like(steps_taken, step), steps_taken)
            sampled |= closing
            live_action[closing] = 0.0        # freeze: stop pushing, stop terminating

        if bool(sampled.all().item()):
            break

        # auto_reset is off, so an env that terminated must be reset before the next
        # step. Frozen envs no longer terminate, so this fires at most once per env.
        done_ids = (cfg.tower_damage(env) | cfg.success_block_extract(env)) \
            .nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            env.reset(env_ids=done_ids)
            sel = cmd.selected_block_idx.clone()

    rows = []
    for i, name in enumerate(targets):
        rows.append({
            "target": name,
            "layer": int(name[1:].split("_")[0]),
            "cone": cone,
            "impratio": impratio,
            "solref": "default" if solref is None else solref,
            "friction": friction,
            "seed": seed,
            "reached_reference": bool(sampled[i].item()) and not bool(damaged_first[i].item()),
            "damaged_before_reference": bool(damaged_first[i].item()),
            "target_shift_mm": round(float(target_at_sample[i].item()) * 1000, 3),
            "drag_ratio": round(float(drag[i].item()), 4),
            "drag_block": drag_block[i],
            "steps": int(steps_taken[i].item()),
        })
        _append_row(args.csv, rows[-1])
        r = rows[-1]
        print(
            f"cone={cone:<9} impratio={impratio:<5} solref={('default' if solref is None else solref)!s:<7} mu={friction:<5} {name:>5} "
            f"drag={r['drag_ratio']:.3f} via {r['drag_block'] or '-':>5} "
            f"at target={r['target_shift_mm']:.1f}mm "
            f"{'(DAMAGE first)' if r['damaged_before_reference'] else ''}",
            flush=True,
        )
    return rows


if __name__ == "__main__":
    main()
