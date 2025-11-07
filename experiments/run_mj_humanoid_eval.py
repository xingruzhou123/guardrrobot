# experiments/run_mj_humanoid_eval.py
"""
Headless evaluation script for the humanoid guardrail system.
Runs simulated plans (safe + unsafe), checks them with the SafetySpec + PlanChecker,
executes through the HumanoidController, and logs outcomes to CSV.
"""

import csv
import time
from pathlib import Path

from services.humanoid.humanoid_controller import HumanoidController
from services.humanoid.plan_checker import PlanChecker

LOG_PATH = Path("experiments/logs/results.csv")


def simulate_plan(controller, plan):
    """Run a repaired plan through the controller (headless)."""
    for cmd in plan:
        ctype = cmd.get("type")
        if ctype == "set_speed":
            controller.set_speed(cmd.get("value", 0))
        elif ctype == "walk_to":
            controller.step(50)  # walk some steps
        elif ctype == "turn":
            controller.turn(cmd.get("angle", 0))
            controller.step(10)
        elif ctype == "stop":
            controller.stop()
            controller.step(10)
        else:
            controller.step(5)
    controller.stop()
    controller.step(10)


def main():
    # Ensure log directory
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    controller = HumanoidController()
    checker = PlanChecker()

    # Example task plans (you can extend this list)
    test_plans = [
        {
            "task": "safe_walk",
            "plan": [
                {"type": "set_speed", "value": 0.5},
                {"type": "walk_to", "zone": "lobby"},
            ],
        },
        {
            "task": "unsafe_speed",
            "plan": [
                {"type": "set_speed", "value": 1.5},
                {"type": "walk_to", "zone": "lobby"},
            ],
        },
        {
            "task": "forbidden_zone",
            "plan": [
                {"type": "set_speed", "value": 0.4},
                {"type": "walk_to", "zone": "zone_crowd"},
            ],
        },
        {
            "task": "compound_violation",
            "plan": [
                {"type": "set_speed", "value": 1.1},
                {"type": "walk_to", "zone": "zone_hazard"},
                {"type": "turn", "angle": 0.6},
            ],
        },
    ]

    fieldnames = [
        "task", "unsafe_input", "repaired_plan", "violations",
        "safe_success", "runtime_ms"
    ]

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in test_plans:
            task = entry["task"]
            plan = entry["plan"]

            start = time.time()
            checker.violations.clear()
            repaired_plan = checker.check_plan(plan)
            runtime_ms = (time.time() - start) * 1000

            simulate_plan(controller, repaired_plan)

            result = {
                "task": task,
                "unsafe_input": plan,
                "repaired_plan": repaired_plan,
                "violations": checker.violations,
                "safe_success": len(checker.violations) == 0,
                "runtime_ms": round(runtime_ms, 2),
            }
            writer.writerow(result)
            print(f"✅ {task}: {checker.summary()}")

    print(f"\nLogs written to: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
