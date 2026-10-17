"""H7 second embodiment: a small differential-drive (unicycle) robot, kinematic with acceleration limits, and its
`EmbodimentExpressionAdapter` for the same `ExpressionIntent`.

This is **not** a physics simulation. It stands in for a wheeled base whose low-level controller tracks (v, ω)
commands within limits. The wheeled robot has no head and no torso, so `lean` and `head_elevation` are ignored. Its
channels:

| channel | realisation |
|---|---|
| tempo | continuous cruise speed, 0.06–0.20 m/s (floor 25%), within acceleration limits |
| hesitation | Poisson pauses and short reverse moves (same rates and caps as the biped adapter) |
| proximity | stopping distance 0.6 → 0.2 m (same rule as the biped) |
| directness | lateral waypoint offset (same rule as the biped); the path is followed with smooth curvature |
| gaze | orient toward / away from the target when stationary |
| scanning | rotate the body back and forth when stationary |

The episode loop mirrors `apm_express.episode.run_episode` and records the arrays `apm_express.features` expects.
`command` is reported as the nearest biped command class, so that temporal features (pauses, switches) are computed
identically for both bodies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from apm_express.adapter import (
    AVERT_RAD,
    COMMAND_NAMES,
    D_FAR,
    D_NEAR,
    GAZE_DEADBAND,
    RETREAT_CAP,
    SCAN_HZ,
    TEMPO_FLOOR,
    WAYPOINT_OFFSET_M,
    WAYPOINT_REACHED_M,
    wrap,
)
from apm_express.intent import ExpressionIntent

V_MAX, W_MAX = 0.20, 1.2  # m/s, rad/s
A_MAX, ALPHA_MAX = 0.4, 3.0  # m/s², rad/s²
DT = 0.02
BODY_SCAN_RAD = 0.6


@dataclass
class WheeledState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    w: float = 0.0


class WheeledAdapter:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t0 = 0.0
        self.pause_until = self.retreat_until = -1.0
        self.arrived = False
        self.waypoint = None
        self.side = 1.0 if self.rng.random() < 0.5 else -1.0
        self.fwd_s = self.back_s = 0.0
        self.near_done = False

    def reset(self, pose, target, intent: ExpressionIntent) -> None:
        intent = intent.sanitized()
        start = np.array(pose[:2])
        d = np.asarray(target) - start
        L = float(np.linalg.norm(d))
        if intent.directness < 0.95 and L > 2 * WAYPOINT_REACHED_M:
            n = np.array([-d[1], d[0]]) / L
            self.waypoint = start + 0.5 * d + self.side * WAYPOINT_OFFSET_M * (1 - intent.directness) * n

    def _orient(self, t, pose, target, intent) -> tuple[float, float]:
        bearing = wrap(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
        offset = 0.0
        if intent.gaze < -GAZE_DEADBAND:
            offset = math.copysign(abs(intent.gaze) * AVERT_RAD, bearing if bearing else 1.0)
        if intent.scanning > 0.05:
            offset += intent.scanning * BODY_SCAN_RAD * (1.0 if math.sin(2 * math.pi * SCAN_HZ * t) >= 0 else -1.0)
        elif abs(intent.gaze) <= GAZE_DEADBAND:
            return 0.0, 0.0
        err = wrap(bearing - offset)
        return 0.0, float(np.clip(2.0 * err, -W_MAX, W_MAX)) if abs(err) > 0.1 else 0.0

    def request(self, t: float, pose, target, intent: ExpressionIntent) -> tuple[float, float]:
        intent = intent.sanitized()
        pos = np.array(pose[:2])
        dist = float(np.linalg.norm(np.asarray(target) - pos))
        d_stop = D_FAR - (D_FAR - D_NEAR) * intent.proximity
        if not self.arrived and dist <= d_stop:
            self.arrived = True
        if self.arrived:
            return self._orient(t, pose, target, intent)
        if t < self.retreat_until:
            self.back_s += DT
            return -0.5 * V_MAX * max(TEMPO_FLOOR, intent.tempo), 0.0
        if t < self.pause_until:
            return self._orient(t, pose, target, intent)
        if t - self.t0 > 1.0:
            rate = 0.25 * intent.hesitation
            if dist <= d_stop + 0.25 and not self.near_done:
                self.near_done = True
                if self.rng.random() < intent.hesitation:
                    rate = 1.0 / DT
            if self.rng.random() < rate * DT:
                pause = 0.8 + 1.2 * intent.hesitation
                retreat = 0.6 * intent.hesitation
                if self.rng.random() < intent.hesitation and self.back_s + retreat <= RETREAT_CAP * self.fwd_s:
                    self.retreat_until = t + retreat
                    self.pause_until = self.retreat_until + pause
                else:
                    self.pause_until = t + pause
                return 0.0, 0.0
            if self.rng.random() < 0.15 * intent.scanning * DT:
                self.pause_until = t + 1.5
                return 0.0, 0.0
        goal = np.asarray(target, dtype=float)
        if self.waypoint is not None:
            if float(np.linalg.norm(self.waypoint - pos)) <= WAYPOINT_REACHED_M:
                self.waypoint = None
            else:
                goal = self.waypoint
        err = wrap(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - pose[2])
        w = float(np.clip(2.0 * err, -W_MAX, W_MAX))
        v = V_MAX * max(TEMPO_FLOOR, intent.tempo) * max(0.0, math.cos(err)) ** 2
        self.fwd_s += DT
        return v, w


def run_wheeled(intent: ExpressionIntent, target, seed: int, seconds: float = 60.0) -> dict:
    """Closed loop for the wheeled body; returns arrays compatible with `apm_express.features.episode_features`."""
    rng = np.random.default_rng(seed)
    s = WheeledState(yaw=float(rng.normal(0.0, 0.02)))
    ad = WheeledAdapter(seed)
    ad.reset((s.x, s.y, s.yaw), target, intent)
    rows = []
    n = int(round(seconds / DT))
    for k in range(n):
        t = k * DT
        v_cmd, w_cmd = ad.request(t, (s.x, s.y, s.yaw), target, intent)
        s.v += float(np.clip(v_cmd - s.v, -A_MAX * DT, A_MAX * DT))
        s.w += float(np.clip(w_cmd - s.w, -ALPHA_MAX * DT, ALPHA_MAX * DT))
        s.x += s.v * math.cos(s.yaw) * DT
        s.y += s.v * math.sin(s.yaw) * DT
        s.yaw = wrap(s.yaw + s.w * DT)
        if abs(v_cmd) > 0.01:
            cmd = "forward" if v_cmd > 0 else "backward"
        elif abs(w_cmd) > 0.05:
            cmd = "turn_left" if w_cmd > 0 else "turn_right"
        else:
            cmd = "rest"
        rows.append({"t": t + DT, "x": s.x, "y": s.y, "yaw": s.yaw, "pitch": 0.0, "tilt": 0.0,
                     "vx_world": s.v * math.cos(s.yaw), "vy_world": s.v * math.sin(s.yaw),
                     "command": COMMAND_NAMES.index(cmd), "power": abs(s.v) * 10 + abs(s.w), "tau_sq": 0.0})
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}
