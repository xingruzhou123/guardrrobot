# services/humanoid_controller.py
import mujoco
import numpy as np

class HumanoidController:
    def __init__(self, model_path="assets/humanoid/unitree_robots/g1/g1_29dof.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.target_vel = 0.0
        self.turn_rate = 0.0

    def set_speed(self, v):
        self.target_vel = np.clip(v, -0.7, 0.7)  # safety clamp

    def turn(self, angle_rate):
        self.turn_rate = np.clip(angle_rate, -0.5, 0.5)

    def stop(self):
        self.target_vel = 0.0
        self.turn_rate = 0.0

    def step(self, n_steps=1):
        """Simple velocity-based stepping loop."""
        for _ in range(n_steps):
            # here you can insert PD control or a simple COM velocity update
            mujoco.mj_step(self.model, self.data)

    def run_episode(self, steps=1000):
        for _ in range(steps):
            self.step()
        print("✅ Finished episode.")

if __name__ == "__main__":
    ctrl = HumanoidController()
    ctrl.set_speed(0.5)
    ctrl.run_episode(200)
