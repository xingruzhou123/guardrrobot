# services/humanoid/plan_checker.py
from services.humanoid.safety_specs import SafetySpec

class PlanChecker:
    def __init__(self):
        self.spec = SafetySpec()
        self.violations = []

    def check_plan(self, plan):
        """Check and repair a list of commands."""
        safe_plan = []
        for cmd in plan:
            is_safe, repaired, reason = self.spec.check_command(cmd)
            if not is_safe:
                self.violations.append(reason)
            safe_plan.append(repaired)
        return safe_plan

    def summary(self):
        if not self.violations:
            return "✅ Plan passed all checks."
        return f"⚠️ Violations: {len(self.violations)}\n- " + "\n- ".join(self.violations)
