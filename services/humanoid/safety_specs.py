# services/humanoid/safety_specs.py
import json

class SafetySpec:
    def __init__(self):
        # Default guardrail parameters
        self.max_speed = 0.7
        self.min_distance_to_human = 0.6
        self.forbidden_zones = ["zone_crowd", "zone_hazard"]
        self.posture_limits = {"knee_bend_max": 0.9, "torso_pitch_max": 0.6}

    def check_command(self, command):
        """Check a single command dict and return (is_safe, repaired_command, reason)."""
        repaired = command.copy()
        reason = None
        safe = True

        if command.get("type") == "set_speed":
            v = command.get("value", 0.0)
            if abs(v) > self.max_speed:
                safe = False
                repaired["value"] = max(min(v, self.max_speed), -self.max_speed)
                reason = f"Speed {v} exceeds max {self.max_speed}"
        elif command.get("type") == "walk_to":
            zone = command.get("zone")
            if zone in self.forbidden_zones:
                safe = False
                repaired["zone"] = "lobby"
                reason = f"Zone {zone} is forbidden"
        return safe, repaired, reason

    def to_json(self):
        return json.dumps({
            "max_speed": self.max_speed,
            "min_distance_to_human": self.min_distance_to_human,
            "forbidden_zones": self.forbidden_zones,
            "posture_limits": self.posture_limits,
        }, indent=2)
