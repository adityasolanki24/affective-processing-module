"""A coordinated expressive movement program from the learned experience state (designed, not learned).

Input: `ExperienceState.readout(identity)`, i.e. wariness, uncertainty and familiarity. These come from what the
robot sensed in earlier meetings and never from an assigned label. Output: one plan for a whole encounter, phase
by phase:

- **observe** (before moving): how long the robot watches the person; steady fixation vs looking around.
- **approach**: path shape (direct vs arcing); gaze locked on the person vs relaxed and looking about.
- **interact** (body posture while near the person):
  - guarded: arms in, torso turned side-on, head drawn back;
  - open: arms relaxed, square-on, head up and forward;
  - head tilt when uncertain but unthreatened.
- **depart**: backing away while still facing the person vs turning and walking off directly.

Speed and stopping distance can be fixed (`matched=True`), so that the remaining differences are carried by
timing, gaze, posture, path shape and departure, not by how fast or how close.

The plan uses movement descriptors (guard, openness, fixation, scanning), not emotion names. The body adapters
turn them into joint offsets within each body's validated limits. This mapping is designed; the memory it reads
is learned.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

NEUTRAL_STATE = {"wariness": 0.0, "uncertainty": 0.0, "familiarity": 0.0}


@dataclass(frozen=True)
class ExpressivePlan:
    observe_s: float          # stand and watch before moving
    fixation: float           # 0..1, gaze held on the person (1) vs free
    scan_amp: float           # rad, amplitude of looking around when not fixating
    scan_hz: float
    directness: float         # 1 straight, 0 strongly arcing approach
    tempo: float              # 0..1 adapter tempo (1 = continuous progress)
    proximity: float          # 0..1 adapter proximity
    hesitation: float         # 0..1
    guard: float              # 0..1 interaction posture: guarded
    openness: float           # 0..1 interaction posture: open, head up and forward
    tilt: float               # 0..1 head tilt (inspection)
    side_on: float            # 0..1 torso / body turned side-on to the person
    retreat_m: float          # back away this far before turning to leave
    depart_turn_rad_s: float  # turning speed when leaving

    def as_dict(self) -> dict:
        return asdict(self)


def plan(state: dict | None, matched: bool = False) -> ExpressivePlan:
    """Designed monotone mapping. `state=None` is the neutral program (no expression)."""
    if state is None:
        return ExpressivePlan(observe_s=0.5, fixation=0.0, scan_amp=0.0, scan_hz=0.0, directness=1.0, tempo=1.0,
                              proximity=0.75, hesitation=0.0, guard=0.0, openness=0.0, tilt=0.0, side_on=0.0,
                              retreat_m=0.0, depart_turn_rad_s=0.5)
    w = min(1.0, max(0.0, float(state["wariness"])))
    u = min(1.0, max(0.0, float(state["uncertainty"])))
    f = min(1.0, max(0.0, float(state["familiarity"])))
    calm = (1.0 - w) * f  # familiar and safe in experience
    p = ExpressivePlan(
        observe_s=0.5 + 3.0 * w + 1.0 * u * (1.0 - w),
        fixation=w,
        scan_amp=0.35 * (1.0 - w) * (0.4 + 0.6 * max(u, calm)),
        scan_hz=0.25,
        directness=1.0 - 0.7 * w,
        tempo=1.0 - 0.4 * w,
        proximity=0.75 - 0.35 * w,
        hesitation=0.5 * w,
        guard=w,
        openness=calm,
        tilt=u * (1.0 - w),
        side_on=w,
        retreat_m=0.35 * w,
        depart_turn_rad_s=0.5 + 0.3 * calm - 0.25 * w)
    if matched:  # speed- and distance-matched: only timing, gaze, posture, path shape and departure differ
        from dataclasses import replace
        p = replace(p, tempo=1.0, proximity=0.75, hesitation=0.0)
    return p


def upper_targets(p: ExpressivePlan, phase: str, t: float, bearing: float) -> dict:
    """Generic upper-body targets at time t (s) of a phase. bearing: the person's direction relative to the
    robot's heading (rad). Returns gaze yaw/pitch (rad, relative) and posture descriptors in 0..1."""
    scan = p.scan_amp * math.sin(2 * math.pi * p.scan_hz * t) if p.scan_amp > 0 else 0.0
    look = p.fixation * bearing + (1.0 - p.fixation) * scan
    if phase in ("settle", "observe"):
        return {"gaze_yaw": look if p.fixation > 0 else scan, "gaze_pitch": 0.0, "guard": 0.5 * p.guard,
                "openness": 0.0, "tilt": 0.0, "side_on": 0.0}
    if phase == "approach":
        return {"gaze_yaw": look, "gaze_pitch": 0.0, "guard": 0.3 * p.guard, "openness": 0.3 * p.openness,
                "tilt": 0.0, "side_on": 0.0}
    if phase == "interact":
        return {"gaze_yaw": p.fixation * bearing + (1.0 - p.fixation) * 0.5 * scan,
                "gaze_pitch": 0.15 * p.openness - 0.1 * p.guard, "guard": p.guard, "openness": p.openness,
                "tilt": p.tilt, "side_on": p.side_on}
    if phase == "depart":
        return {"gaze_yaw": p.fixation * bearing, "gaze_pitch": 0.0, "guard": 0.5 * p.guard, "openness": 0.0,
                "tilt": 0.0, "side_on": 0.0}
    return {"gaze_yaw": 0.0, "gaze_pitch": 0.0, "guard": 0.0, "openness": 0.0, "tilt": 0.0, "side_on": 0.0}
