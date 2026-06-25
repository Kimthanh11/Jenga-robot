import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("jenga.xml")
data = mujoco.MjData(model)

x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hook_x_motor")
yaw_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hook_yaw_pos")
y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hook_y_pos")
z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hook_z_pos")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0

        data.ctrl[y_act_id] = 0.0
        data.ctrl[z_act_id] = 0.0
        data.ctrl[yaw_act_id] = 0.0

        if data.time < 1.0:
            pass
        elif data.time < 2.0:
            data.ctrl[yaw_act_id] = 0.5
        elif data.time < 4.0:
            data.ctrl[yaw_act_id] = 0.5
            data.ctrl[x_act_id] = -1.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)