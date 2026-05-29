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

# 2. Get the velocity/force address (dof address) for this specific joint.
dof_addr = model.jnt_dofadr[joint_id]

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 200
    while viewer.is_running():
        
        # pull outward to the right
        if data.time < 1.0:
            data.qfrc_applied[dof_addr] = 0.0 
        elif data.time < 2:
            data.qfrc_applied[dof_addr] = -5.0 
        else:
            data.qfrc_applied[dof_addr] = 0.0


        mujoco.mj_step(model, data)
        viewer.sync()
        #time.sleep(model.opt.timestep)