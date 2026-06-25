import mujoco
import mujoco.viewer
import time

from tower_state import TowerState


model = mujoco.MjModel.from_xml_path("jenga.xml")
data = mujoco.MjData(model)

print("Number of actuators:", model.nu)

SETTLE_TIME = 1.0
PUSH_DURATION = 40
PAUSE_DURATION = 1.0
PUSH_CTRL = -1.0
PRINT_EVERY = 0.5  # seconds between state printouts

schedule = [
    (0, 3.0, 6.0),
    (1, 7.0, 10),
    (2, 10.0, 87.0),
]

state = TowerState(model, data)
recorded = False
next_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0

        active_actuator_id = None

        for actuator_id, start_time, end_time in schedule:
            if start_time <= data.time < end_time and actuator_id < model.nu:
                data.ctrl[actuator_id] = PUSH_CTRL
                active_actuator_id = actuator_id
                break

        mujoco.mj_step(model, data)

        # Snapshot the settled tower once, just before pushing begins.
        if not recorded and data.time >= SETTLE_TIME:
            state.record_initial()
            recorded = True
            print(f"[{data.time:5.1f}s] tower settled, reference recorded")

        # Print live state readings at a fixed cadence.
        if recorded and data.time >= next_print:
            print(f"[{data.time:5.1f}s] {state.summary()}")
            next_print = data.time + PRINT_EVERY

        viewer.sync()
