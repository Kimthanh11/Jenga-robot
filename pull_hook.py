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
    (0, 3.0, 6.0),
    (1, 7.0, 10),
    (2, 10.0, 87.0),
]

# get body name+++++++++++++++++++++++
def body_name(body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def is_jenga_block(body_name):
    return body_name is not None and body_name.startswith("b")


def pusher_body_id_from_actuator(actuator_id):
    joint_id = model.actuator_trnid[actuator_id][0]
    return model.jnt_bodyid[joint_id]


def touched_blocks_by_actuator(actuator_id):
    pusher_body_id = pusher_body_id_from_actuator(actuator_id)
    touched_blocks = set()

    for i in range(data.ncon):
        contact = data.contact[i]

        body1_id = model.geom_bodyid[contact.geom1]
        body2_id = model.geom_bodyid[contact.geom2]

        body1_name = body_name(body1_id)
        body2_name = body_name(body2_id)

        if body1_id == pusher_body_id and is_jenga_block(body2_name):
            touched_blocks.add(body2_name)

        if body2_id == pusher_body_id and is_jenga_block(body1_name):
            touched_blocks.add(body1_name)

    return sorted(touched_blocks)
# end get body name++++++++++

printed_contacts = set()

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

        if active_actuator_id is not None:
            blocks = touched_blocks_by_actuator(active_actuator_id)

            for block in blocks:
                key = (active_actuator_id, block)

                if key not in printed_contacts:
                    print(f"{data.time:.2f}s actuator {active_actuator_id} touches {block}")
                    printed_contacts.add(key)

        viewer.sync()