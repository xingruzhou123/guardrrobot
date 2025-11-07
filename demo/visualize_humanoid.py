# demo/visualize_guarded_walk.py
"""
Visualize the humanoid walking under guardrail control.
Combines SafetySpec + PlanChecker + HumanoidController in MuJoCo viewer.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

from services.humanoid.humanoid_controller import HumanoidController
from services.humanoid.plan_checker import PlanChecker

MODEL_PATH = Path("assets/humanoid/unitree_robots/g1/g1_29dof.xml")


def main():
    # --- Load model and data ---
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # --- Initialize guardrail system ---
    controller = HumanoidController()
    checker = PlanChecker()

    # Example plan (unsafe, will be repaired)
    raw_plan = [
        {"type": "set_speed", "value": 1.5},   # Unsafe, exceeds 0.7
        {"type": "walk_to", "zone": "lobby"},  # Safe zone
    ]

    print("Original plan:", raw_plan)
    safe_plan = checker.check_plan(raw_plan)
    print("→ Repaired plan:", safe_plan)
    print(checker.summary())

    # Apply repaired speed limit
    for cmd in safe_plan:
        if cmd["type"] == "set_speed":
            controller.set_speed(cmd["value"])

    # --- Launch viewer ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✅ Guarded walking demo started. Close window to stop.")
        t0 = time.time()

        while viewer.is_running():
            elapsed = time.time() - t0

            # simulate periodic walking for 10 s
            if elapsed < 10:
                controller.step(1)
            else:
                controller.stop()
                controller.step(1)

            mujoco.mj_step(model, data)
            viewer.sync()

    print("✅ Visualization finished.")


if __name__ == "__main__":
    main()
