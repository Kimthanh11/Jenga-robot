"""Train the incomplete-tower + random-block task (v1/v2/abort) with curriculum
lengths scaled to the run's OWN length, not hardcoded absolute env-steps.

Why this exists: jenga_incomplete_random_cfg.py / _v2_cfg.py / _abort_cfg.py define
their curricula (perturbation penalty ramp, success-distance ramp, touch/yaw ramp,
abort-threshold ramp) as absolute env-step counts (e.g. PERTURBATION_CURRICULUM_STEPS
= 100_000). Those numbers only make sense for whatever run length someone had in mind
when they were tuned -- launch a shorter or longer run via train.py's
--agent.max-iterations and the curriculum silently completes a different FRACTION of
its schedule than intended (a 1200-iteration smoke test only reaches 38% of a
100_000-step schedule, not the "end" state its own 100% of iterations would suggest).

This script takes --iterations (the run's total target iteration count, same meaning
as train.py's --agent.max-iterations) and computes each curriculum's absolute step
count as a FRACTION of this run's total env-steps (iterations * num_steps_per_env),
so every curriculum reaches its intended point in the run regardless of how long the
run is. Fractions are overridable per curriculum for ablations; defaults encode the
intended pacing (see each --*-frac help string).

On a resume, pass the same --iterations you'd pass for the full run (the TOTAL target,
matching train.py's own convention for --agent.max-iterations) so the curriculum
schedule stays consistent across the resume rather than being redefined against
whatever iteration count remains.
"""

from __future__ import annotations

import argparse

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import register_mjlab_task

import mjlab_jenga.jenga_incomplete_random_cfg as v1_cfg
import mjlab_jenga.jenga_incomplete_random_v2_cfg as v2_cfg
import mjlab_jenga.jenga_incomplete_random_abort_cfg as abort_cfg

NUM_STEPS_PER_ENV = 32  # matches jenga_ppo_runner_cfg() in jenga_incomplete_random_cfg.py

# NOT the same IDs mjlab_jenga/__init__.py registers at package-import time (that
# import is unavoidable -- importing mjlab_jenga.jenga_incomplete_random_cfg below
# runs the package's __init__.py first) -- register_mjlab_task raises ValueError on a
# duplicate task_id, it does not overwrite. "-Staged" keeps these distinct.
TASK_IDS = {
    "v1": "Mjlab-Jenga-IncompleteRandom-Staged",
    "v2": "Mjlab-Jenga-IncompleteRandomV2-Staged",
    "abort": "Mjlab-Jenga-IncompleteRandomAbort-Staged",
}
EXPERIMENT_NAMES = {
    "v1": "jenga_incomplete_randblock",
    "v2": "jenga_incomplete_randblock_v2",
    "abort": "jenga_incomplete_randblock_abort_v2",
}
TASK_MODULES = {"v1": v1_cfg, "v2": v2_cfg, "abort": abort_cfg}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=("v1", "v2", "abort"))
    parser.add_argument("--iterations", type=int, default=6000,
                         help="Total target iterations for this run (same meaning as "
                              "--agent.max-iterations; on resume, pass the FULL target, "
                              "not the remaining count).")
    parser.add_argument("--num-envs", type=int, default=1536)
    parser.add_argument("--load-run", default=None)
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--run-suffix", default=None,
                         help="Appended to the run name to avoid two runs launched in "
                              "the same wall-clock second colliding on rsl_rl's "
                              "timestamp-only run directory (FileExistsError on "
                              "Jenga-robot.diff). Give concurrent launches distinct "
                              "suffixes.")

    parser.add_argument("--success-frac", type=float, default=0.6,
                         help="Fraction of the run's total env-steps over which the "
                              "required success distance ramps from easy to hard. "
                              "(v1/v2/abort)")
    parser.add_argument("--perturbation-frac", type=float, default=0.7,
                         help="Fraction of the run's total env-steps over which the "
                              "tower-instability penalty ramps to full weight. Slower "
                              "than success on purpose (Fazeli-style: push first, care "
                              "about stability once pushing is reliable). (v1/v2/abort)")
    parser.add_argument("--touch-begin-frac", type=float, default=0.05,
                         help="Fraction of the run at which the touch (y/z contact "
                              "point) action starts unfreezing. (v2/abort)")
    parser.add_argument("--touch-span-frac", type=float, default=0.25,
                         help="Fraction of the run's total env-steps over which touch "
                              "ramps from 0 to full scale, starting at --touch-begin-frac. "
                              "(v2/abort)")
    parser.add_argument("--yaw-begin-frac", type=float, default=0.40,
                         help="Fraction of the run at which the yaw action starts "
                              "unfreezing -- after touch has had time to stabilize. "
                              "(v2/abort)")
    parser.add_argument("--yaw-span-frac", type=float, default=0.60,
                         help="Fraction of the run's total env-steps over which yaw "
                              "ramps from 0 to full scale, starting at --yaw-begin-frac. "
                              "Deliberately slow (begin+span can exceed 1.0, meaning yaw "
                              "may not reach full scale within one run) -- yaw grinding "
                              "into the tower at full deflection caused a physics-NaN "
                              "crash previously (job 116256). (v2/abort)")
    parser.add_argument("--abort-frac", type=float, default=0.20,
                         help="Fraction of the run's total env-steps over which the "
                              "abort threshold eases from its near-unreachable start to "
                              "its minimum. Short relative to the others so abort is a "
                              "genuinely usable option well before the run ends, instead "
                              "of only in the last stretch. (abort only)")

    args = parser.parse_args()

    if (args.load_run is None) != (args.load_checkpoint is None):
        parser.error("--load-run and --load-checkpoint must be given together")

    total_steps = args.iterations * NUM_STEPS_PER_ENV

    def steps(frac: float) -> int:
        return max(1, round(total_steps * frac))

    # v1 curricula (read by all three variants, since v2/abort build on v1's
    # _make_env_cfg() and reuse its reward/curriculum functions as-is).
    v1_cfg.SUCCESS_CURRICULUM_STEPS = steps(args.success_frac)
    v1_cfg.PERTURBATION_CURRICULUM_STEPS = steps(args.perturbation_frac)

    # v2 curricula (touch/yaw) -- only meaningful for v2/abort, but harmless to set
    # unconditionally since v1 never reads them.
    v2_cfg.TOUCH_CURRICULUM_BEGIN_STEP = steps(args.touch_begin_frac)
    v2_cfg.TOUCH_CURRICULUM_STEPS = steps(args.touch_span_frac)
    v2_cfg.YAW_CURRICULUM_BEGIN_STEP = steps(args.yaw_begin_frac)
    v2_cfg.YAW_CURRICULUM_STEPS = steps(args.yaw_span_frac)

    # abort curriculum -- abort only.
    abort_cfg.ABORT_CURRICULUM_STEPS = steps(args.abort_frac)

    task = TASK_MODULES[args.variant]
    task_id = TASK_IDS[args.variant]
    experiment_name = EXPERIMENT_NAMES[args.variant]

    env_cfg = task.jenga_env_cfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = task.jenga_ppo_runner_cfg()
    agent_cfg.experiment_name = experiment_name
    agent_cfg.max_iterations = args.iterations
    if args.run_suffix:
        agent_cfg.run_name = args.run_suffix
    if args.load_run is not None:
        agent_cfg.resume = True
        agent_cfg.load_run = args.load_run
        agent_cfg.load_checkpoint = args.load_checkpoint

    print(
        f"variant={args.variant} task_id={task_id} iterations={args.iterations} "
        f"total_steps={total_steps} num_envs={args.num_envs}\n"
        f"  success_curriculum_steps={v1_cfg.SUCCESS_CURRICULUM_STEPS} "
        f"(reaches full difficulty at iter {v1_cfg.SUCCESS_CURRICULUM_STEPS // NUM_STEPS_PER_ENV})\n"
        f"  perturbation_curriculum_steps={v1_cfg.PERTURBATION_CURRICULUM_STEPS} "
        f"(reaches full weight at iter {v1_cfg.PERTURBATION_CURRICULUM_STEPS // NUM_STEPS_PER_ENV})\n"
        f"  touch_curriculum=({v2_cfg.TOUCH_CURRICULUM_BEGIN_STEP}, "
        f"+{v2_cfg.TOUCH_CURRICULUM_STEPS}) "
        f"(begins iter {v2_cfg.TOUCH_CURRICULUM_BEGIN_STEP // NUM_STEPS_PER_ENV}, "
        f"full by iter {(v2_cfg.TOUCH_CURRICULUM_BEGIN_STEP + v2_cfg.TOUCH_CURRICULUM_STEPS) // NUM_STEPS_PER_ENV})\n"
        f"  yaw_curriculum=({v2_cfg.YAW_CURRICULUM_BEGIN_STEP}, "
        f"+{v2_cfg.YAW_CURRICULUM_STEPS}) "
        f"(begins iter {v2_cfg.YAW_CURRICULUM_BEGIN_STEP // NUM_STEPS_PER_ENV}, "
        f"full by iter {(v2_cfg.YAW_CURRICULUM_BEGIN_STEP + v2_cfg.YAW_CURRICULUM_STEPS) // NUM_STEPS_PER_ENV})\n"
        f"  abort_curriculum_steps={abort_cfg.ABORT_CURRICULUM_STEPS} "
        f"(reaches easiest threshold at iter {abort_cfg.ABORT_CURRICULUM_STEPS // NUM_STEPS_PER_ENV})",
        flush=True,
    )

    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=task.jenga_env_cfg(play=True),
        rl_cfg=agent_cfg,
    )

    launch_training(task_id, TrainConfig(env=env_cfg, agent=agent_cfg))


if __name__ == "__main__":
    main()
