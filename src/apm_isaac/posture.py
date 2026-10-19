"""Body adapters for the expressive program: generic upper-body targets -> bounded, rate-limited joint offsets.

The offsets are added to joint *targets* of joints that each robot's locomotion policy does not control:
- T1: head, arms and waist (the official leg policy controls only the twelve leg joints);
- NERVA: the four head joints (its controller holds them at fixed targets).
Leg targets, root pose and contacts are never touched. An offset is applied only while the body is upright (tilt
below its trip angle and not braking); otherwise it decays to zero.

Sign constants were checked against rendered pose stills (`isaac/posture_check.py`). Limits are kept inside each
body's validated range (NERVA: programme H's admitted head pitch/yaw ±0.4 rad at 0.75 rad/s; neck pitch and head
roll are new and limited to ±0.25 rad, pending the posture stability check).
"""
from __future__ import annotations

import numpy as np

# T1 upper joints, in the candidate's joint order (apm_isaac.booster.JOINTS[:11])
T1_UPPER = ("AAHead_yaw", "Head_pitch", "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
            "Left_Elbow_Yaw", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            "Waist")
# directions (+1/-1) of each descriptor on each joint; verified by posture_check stills
T1_SIGNS = {"head_pitch_up": -1.0, "shoulder_forward": -1.0, "elbow_bend": 1.0, "arm_out_left": 1.0,
            "waist_left": 1.0}
NERVA_SIGNS = {"neck_back": -1.0, "head_pitch_up": 1.0, "head_roll": 1.0}


def t1_offsets(g: dict, side: float = 1.0) -> np.ndarray:
    """Generic targets -> 11 T1 upper-body offsets (rad). side: which way to turn side-on (+1 left)."""
    s = T1_SIGNS
    o = np.zeros(11)
    o[0] = np.clip(g["gaze_yaw"], -0.6, 0.6)
    o[1] = s["head_pitch_up"] * np.clip(g["gaze_pitch"] + 0.15 * g["openness"] - 0.15 * g["guard"], -0.3, 0.3)
    guard, openness = g["guard"], g["openness"]
    # guarded: upper arms forward, elbows bent, forearms in front of the trunk;
    # open: arms slightly away from the body, elbows nearly straight
    for k, mirror in ((2, 1.0), (6, -1.0)):
        o[k] += s["shoulder_forward"] * (0.55 * guard)
        o[k + 1] += mirror * s["arm_out_left"] * (0.25 * openness - 0.15 * guard)
        o[k + 2] += s["elbow_bend"] * (0.9 * guard)
    o[10] = s["waist_left"] * side * 0.35 * g["side_on"]
    return o


def nerva_offsets(g: dict) -> np.ndarray:
    """Generic targets -> NERVA head offsets (neck_pitch, head_pitch, head_yaw, head_roll), rad."""
    s = NERVA_SIGNS
    neck = s["neck_back"] * (0.25 * g["guard"] - 0.2 * g["openness"])
    pitch = s["head_pitch_up"] * (g["gaze_pitch"] + 0.25 * g["openness"] - 0.2 * g["guard"])
    return np.array([np.clip(neck, -0.25, 0.25), np.clip(pitch, -0.4, 0.4), np.clip(g["gaze_yaw"], -0.4, 0.4),
                     np.clip(s["head_roll"] * 0.25 * g["tilt"], -0.25, 0.25)])


class RateLimited:
    """Moves current offsets toward targets at a bounded rate; decays to zero when not allowed."""

    def __init__(self, n: int, rate: float):
        self.value: np.ndarray = np.zeros(n)
        self.rate = float(rate)

    def step(self, target: np.ndarray, dt: float, allowed: bool = True) -> np.ndarray:
        goal = target if allowed else np.zeros_like(self.value)
        self.value = self.value + np.clip(goal - self.value, -self.rate * dt, self.rate * dt)
        return self.value.copy()
