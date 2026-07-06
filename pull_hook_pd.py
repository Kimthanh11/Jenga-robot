import argparse
import logging
import sys

import mujoco
import mujoco.viewer

from tower_state import TowerState


# --- logging: stdout by default, optional file via --log-file -----------------
parser = argparse.ArgumentParser(description="PD-controlled Jenga pusher")
parser.add_argument(
    "--log-file",
    default=None,
    help="write logs to this file instead of stdout (default: stdout)",
)
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


def joint_addresses(joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")

    return model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]


hooks = [
    joint_addresses("hook_slide"),
    joint_addresses("hook_slide2"),
    joint_addresses("hook_slide3"),
]

# PD position controller:
# force = KP * position_error - KD * velocity, capped by MAX_FORCE.
KP = 80.0
KD = 10.0
MAX_FORCE = 5.0
TARGET_QPOS = -0.20

SETTLE_TIME = 1.0
PRINT_EVERY = 0.5  # seconds between state printouts

schedule = [
    (0, 1.0, 10.0),
    (1, 11.0, 51.0),
    (2, 52.0, 92.0),
]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def apply_position_controller(hook_id, target_qpos):
    qpos_addr, dof_addr = hooks[hook_id]
    position = data.qpos[qpos_addr]
    velocity = data.qvel[dof_addr]

    force = KP * (target_qpos - position) - KD * velocity
    data.qfrc_applied[dof_addr] = clamp(force, -MAX_FORCE, MAX_FORCE)


state = TowerState(model, data)
recorded = False
next_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        for hook_id, start_time, end_time in schedule:
            if start_time <= data.time < end_time:
                apply_position_controller(hook_id, TARGET_QPOS)
                break

        mujoco.mj_step(model, data)

        # Snapshot the settled tower once, just before pushing begins.
        if not recorded and data.time >= SETTLE_TIME:
            state.record_initial()
            recorded = True
            log.info(f"[{data.time:5.1f}s] tower settled, reference recorded")

        # Print live state readings at a fixed cadence.
        if recorded and data.time >= next_print:
            log.info(f"[{data.time:5.1f}s] {state.summary()}")
            next_print = data.time + PRINT_EVERY

        viewer.sync()
