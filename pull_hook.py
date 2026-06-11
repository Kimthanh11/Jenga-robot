import mujoco
import mujoco.viewer
import time


model = mujoco.MjModel.from_xml_path("jenga.xml")
data = mujoco.MjData(model)

print("Number of actuators:", model.nu)

SETTLE_TIME = 1.0
PUSH_DURATION = 40
PAUSE_DURATION = 1.0
PUSH_CTRL = -1.0

schedule = [
    (0, 16.0, 21.0),
    (1, 22.0, 28.0),
    (2, 29.0, 87.0),
]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0

        for actuator_id, start_time, end_time in schedule:
            if start_time <= data.time < end_time and actuator_id < model.nu:
                data.ctrl[actuator_id] = PUSH_CTRL
                break

        mujoco.mj_step(model, data)
        viewer.sync()
