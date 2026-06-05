import mujoco
import mujoco.viewer


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


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        for hook_id, start_time, end_time in schedule:
            if start_time <= data.time < end_time:
                apply_position_controller(hook_id, TARGET_QPOS)
                break

        mujoco.mj_step(model, data)
        viewer.sync()
