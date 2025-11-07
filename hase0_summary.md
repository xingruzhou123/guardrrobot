# Phase 0 Summary — Humanoid Guardrails Project

**Date:** 2025-11-09  
**Target Conference:** IEEE/RSJ IROS 2026  
**Robot Platform:** Unitree Humanoid (URDF → MJCF for MuJoCo)

---

## 1 · Motivation
LLM-driven humanoids require explicit action-level safety guardrails to avoid unsafe behaviors in open-ended tasks.  
We propose a two-stage guardrail architecture:
1. **Stage 1:** Root-of-Trust LLM → JSON safety specification  
2. **Stage 2:** Plan checker + repair executing within MuJoCo control loop

---

## 2 · Simulation Setup

| Item | Description |
|------|--------------|
| Simulator | MuJoCo 3.1 (headless / GLFW) |
| Model | `unitree_humanoid.xml` (converted from official URDF) |
| Control Loop | 50 Hz PD tracking + velocity limit (0.7 m/s) |
| World Zones | lobby (safe), zone_crowd (no-go), zone_hazard (no-go) |
| Tasks | Task A: No-go Zone Avoidance / Human Proximity <br> Task B: Approach Table w/ Posture Limits |

---

## 3 · Safety Specification (JSON Schema)

```json
{
  "forbidden_zones": ["zone_crowd", "zone_hazard"],
  "max_speed": 0.7,
  "min_distance_to_human": 0.6,
  "posture_limits": { "knee_bend_max": 0.9, "torso_pitch_max": 0.6 },
  "requirements": [
    {"sequence": ["turn", "walk_to", "approach"]},
    {"if": "carrying_object", "then": "no_enter:zone_crowd"}
  ]
}