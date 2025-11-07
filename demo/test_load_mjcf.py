# demo/test_load_mjcf.py
import mujoco
import mujoco.viewer


MODEL_PATH = "assets/humanoid/unitree_robots/g1/g1_29dof.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
print(f"Loaded model: {model.nbody} bodies, {model.njnt} joints")

for _ in range(200):
    mujoco.mj_step(model, data)
print("✅ Simulation stepped successfully (headless mode)")
