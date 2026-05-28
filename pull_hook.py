import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("jenga.xml")
data = mujoco.MjData(model)

#get the id of the joint "hook_slide" and vonvert it from name to id
joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "hook_slide"
)

#get the position/index of the joint out of the data.qpos array
qpos_addr = model.jnt_qposadr[joint_id]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # pull outward to the right
        if data.time < 5.0:
            data.ctrl[0] = -1


        mujoco.mj_step(model, data)
        viewer.sync()
        #time.sleep(model.opt.timestep)