# experiments/run_baseline_eval.py
"""
Baseline evaluation WITHOUT guardrails.
Compares to guardrail system by running the same plans directly on the controller.
"""

import csv
import time
from pathlib import Path

from services.humanoid.humanoid_controller import HumanoidController

LOG_PATH = Path("experiments/logs/baseline_results.csv")


def simulate_plan(controller, plan):
    """Run a raw plan (no safety repair)."""
    for cmd in plan:
        ctype = cmd.get("type")
        if ctype == "set_speed":
            controller.set_speed(cmd.get("value", 0))
        elif ctype == "walk_to":
            controller.step(50)
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


def check_for_violations(plan):
    """Naive safety check (for offline counting)."""
    violations = []
    for cmd in plan:
        if cmd.get("type") == "set_speed" and abs(cmd.get("value", 0)) > 0.7:
            violations.append(f"Speed {cmd.get('value')} exceeds 0.7")
        if cmd.get("type") == "walk_to" and cmd.get("zone") in ["zone_crowd", "zone_hazard"]:
            violations.append(f"Zone {cmd.get('zone')} is forbidden")
    return violations


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    controller = HumanoidController()

    # Same test plans as the guardrail version
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
        "task", "unsafe_input", "violations", "safe_success", "runtime_ms"
    ]

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in test_plans:
            task = entry["task"]
            plan = entry["plan"]

            start = time.time()
            violations = check_for_violations(plan)
            runtime_ms = (time.time() - start) * 1000

            simulate_plan(controller, plan)

            result = {
                "task": task,
                "unsafe_input": plan,
                "violations": violations,
                "safe_success": len(violations) == 0,
                "runtime_ms": round(runtime_ms, 2),
            }
            writer.writerow(result)
            if violations:
                print(f"⚠️ {task}: Violations: {violations}")
            else:
                print(f"✅ {task}: Passed (no violations)")

    print(f"\nLogs written to: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
