"""Push each target block "till failure": keep pushing while the block still
moves, and stop as soon as it stalls (it's either fully out or jammed). Unlike
pull_hook_pd.py this has no fixed time schedule — each push ends when the block
stops responding, then the next hook takes over.

The push force is capped by a height-dependent budget (lots of force low in the
tower, little up top) so high blocks aren't launched. Each hook drives its block
through real hook-to-block contact.

Run with the viewer (macOS): mjpython pull_hook_pd_till_failure.py
Optional log file:           mjpython pull_hook_pd_till_failure.py --log-file run.txt
"""

import argparse
import logging
import sys

import numpy as np
import mujoco
import mujoco.viewer

from tower_state import TowerState


# --- logging: stdout by default, optional file via --log-file -----------------
parser = argparse.ArgumentParser(description="Push Jenga blocks till failure")
parser.add_argument("--log-file", default=None,
                    help="write logs to this file instead of stdout")
args = parser.parse_args()

handler = (
    logging.FileHandler(args.log_file, mode="w")
    if args.log_file
    else logging.StreamHandler(sys.stdout)
)
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[handler])
log = logging.getLogger("jenga")


model = mujoco.MjModel.from_xml_path("jenga.xml")
data = mujoco.MjData(model)

LAYERS = 9

# PD position controller (same gains as pull_hook_pd.py).
KP = 80.0
KD = 10.0
FORCE_MAX = 10.0        # force budget at the bottom layer (N)
TARGET_QPOS = -0.30     # push the hook well past full extraction; we stop on stall

SETTLE_TIME = 1.0       # let the tower settle before pushing
STALL_SPEED = 0.005     # block speed (m/s) below which it counts as "not moving"
STALL_TIME = 1.0        # seconds stalled before we declare the push finished
MAX_PUSH_TIME = 20.0    # hard cap per block so we never push forever

# Each pusher: hook joint + the block it targets (layer, pos).
# These three were picked by auto_search.py as SAFE & fully extractable.
pushers = [
    ("hook_slide",  2, 2),   # low
    ("hook_slide2", 5, 1),   # middle
    ("hook_slide3", 7, 3),   # high
]

dt = model.opt.timestep


def force_budget(layer):
    """Height-dependent force cap: FORCE_MAX at the bottom, small at the top."""
    return FORCE_MAX * (LAYERS - layer + 1) / LAYERS


def hook_addr(joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def block_speed(block_name):
    """Linear speed (m/s) of a block from its free-joint velocity."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    dof = model.jnt_dofadr[model.body_jntadr[bid]]
    return float(np.linalg.norm(data.qvel[dof:dof + 3]))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


state = TowerState(model, data, LAYERS)
recorded = False
current = 0              # index into `pushers`
stall_timer = 0.0        # how long the current block has been stalled
push_start = 0.0         # sim time the current push began

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        # Snapshot the settled tower once, then start the first push.
        if not recorded and data.time >= SETTLE_TIME:
            state.record_initial()
            recorded = True
            push_start = data.time
            j, layer, pos = pushers[current]
            log.info(f"[{data.time:5.1f}s] pushing b{layer}_{pos} "
                     f"(Fmax={force_budget(layer):.1f}N)...")

        # Drive the current hook until its block stalls.
        if recorded and current < len(pushers):
            joint, layer, pos = pushers[current]
            block = f"b{layer}_{pos}"
            qpos_addr, dof_addr = hook_addr(joint)
            fmax = force_budget(layer)

            force = KP * (TARGET_QPOS - data.qpos[qpos_addr]) - KD * data.qvel[dof_addr]
            data.qfrc_applied[dof_addr] = clamp(force, -fmax, fmax)

            if block_speed(block) < STALL_SPEED:
                stall_timer += dt
            else:
                stall_timer = 0.0

            elapsed = data.time - push_start
            if stall_timer >= STALL_TIME or elapsed >= MAX_PUSH_TIME:
                slide = state.slide(block)
                collapsed = state.is_collapsed(exclude=block)
                reason = "stalled" if stall_timer >= STALL_TIME else "timeout"
                log.info(f"[{data.time:5.1f}s] {block} done ({reason}): "
                         f"slide={slide:.3f}m  tower_collapsed={collapsed}")

                current += 1
                stall_timer = 0.0
                push_start = data.time
                if current < len(pushers):
                    j, nl, npos = pushers[current]
                    log.info(f"[{data.time:5.1f}s] pushing b{nl}_{npos} "
                             f"(Fmax={force_budget(nl):.1f}N)...")
                else:
                    log.info(f"[{data.time:5.1f}s] all pushes done. "
                             f"final: {state.summary()}")

        mujoco.mj_step(model, data)
        viewer.sync()
