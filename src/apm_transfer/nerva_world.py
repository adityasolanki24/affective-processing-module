"""NERVA encounter world for programme J (J3–J5): entities that physically interact with the biped.

**Entity model.** An entity stands beside the robot's stopping point. It occasionally lunges toward the robot, as a
Poisson process. Each lunge has a reach and a speed drawn from its type's ranges. If the reach exceeds the robot's
current distance d, it makes contact, and a horizontal push is applied to NERVA's simulated body:

    |Δv| = min(CAP, speed · min(1, (reach − d) / 0.2) · 0.1)    with CAP = 0.10 m/s

CAP is the envelope validated in programme H (H1b, stop-transition tolerance). The push points from the entity toward
the robot, with ±20° jitter.

Contact and its magnitude therefore depend on the entity's *type* and on the robot's *own approach*: its stopping
distance, and how long it stays within reach. This is simulation dynamics, not a scripted emotional outcome. The
entity is kinematic; its contact is an impulse, as in NERVA's own push tests. A free MuJoCo body is not allowed by
NERVA's scene contract.

**Robot-observable evidence** per encounter window (`Evidence`):
- IMU gyro and accelerometer, read from NERVA's MuJoCo sensors, the same signals its policy observes, plus NERVA's
  training noise model (uniform ±0.1 rad/s gyro, ±0.05 accelerometer);
- IMU tilt;
- foot-contact sensors;
- actuator effort (Σ τ², from the servos' load).

Simulator body velocity, entity state and the type are recorded separately as `truth`, for evaluation only.

Entity types (pilot set, fixed in the J3 addendum):

| type | lunges per s | reach (m) | speed (m/s) | character |
|---|---|---|---|---|
| calm | 0.05 | 0.10–0.45 | 0.1–0.4 | rare, light |
| gentle | 0.30 | 0.25–0.60 | 0.1–0.3 | frequent, light |
| forceful | 0.10 | 0.30–0.80 | 0.6–1.0 | rare, strong |
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CAP = 0.10
REACH_RAMP_M = 0.2
GYRO_NOISE, ACC_NOISE = 0.1, 0.05

TYPES: dict[str, dict] = {
    "calm": {"rate": 0.05, "reach": (0.10, 0.45), "speed": (0.1, 0.4)},
    "gentle": {"rate": 0.30, "reach": (0.25, 0.60), "speed": (0.1, 0.3)},
    "forceful": {"rate": 0.10, "reach": (0.30, 0.80), "speed": (0.6, 1.0)},
}


def lunges(rng: np.random.Generator, etype: str, seconds: float) -> list[tuple[float, float, float]]:
    """(time, reach, speed) of each lunge in a window of `seconds`."""
    p = TYPES[etype]
    n = int(rng.poisson(p["rate"] * seconds))
    times = np.sort(rng.uniform(0.0, seconds, n))
    return [(float(t), float(rng.uniform(*p["reach"])), float(rng.uniform(*p["speed"]))) for t in times]


def impulse(reach: float, speed: float, distance: float) -> float:
    """Push magnitude (m/s) of one lunge at the robot's current distance; 0 if out of reach."""
    if reach <= distance:
        return 0.0
    return min(CAP, speed * min(1.0, (reach - distance) / REACH_RAMP_M) * 0.1)


@dataclass
class Evidence:
    """Robot-observable summary of one encounter window."""
    peak_gyro: float
    peak_acc: float
    stab_time: float
    contact_changes: int
    effort_ratio: float
    tilt_peak: float
    fell: bool

    def as_list(self) -> list:
        return [round(self.peak_gyro, 5), round(self.peak_acc, 5), round(self.stab_time, 3), self.contact_changes,
                round(self.effort_ratio, 4), round(self.tilt_peak, 3), self.fell]


@dataclass
class ImuLogger:
    """Per-control-step robot-observable readings with NERVA's training noise."""
    rng: np.random.Generator
    gyro: list[float] = field(default_factory=list)
    acc: list[np.ndarray] = field(default_factory=list)
    contacts: list[tuple] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)
    tilt: list[float] = field(default_factory=list)

    def read(self, ctrl, row: dict) -> None:
        inf, d = ctrl.sim.inf, ctrl.sim.data
        g = np.asarray(inf.get_gyro(d), float) + self.rng.uniform(-GYRO_NOISE, GYRO_NOISE, 3)
        a = np.asarray(inf.get_accelerometer(d), float) + self.rng.uniform(-ACC_NOISE, ACC_NOISE, 3)
        self.gyro.append(float(np.linalg.norm(g)))
        self.acc.append(a[:2].copy())
        self.contacts.append(tuple(bool(c) for c in row["contacts"]))
        self.effort.append(float(row["tau_sq"]))
        self.tilt.append(float(row["tilt"]))

    def summarise(self, start: int, baseline: slice, dt: float, gyro_thr: float = 0.35) -> Evidence:
        g = np.array(self.gyro[start:])
        acc = np.array(self.acc)
        base_acc = acc[baseline].mean(0) if len(acc[baseline]) else np.zeros(2)
        a = np.linalg.norm(acc[start:] - base_acc, axis=1)
        above = np.flatnonzero(g > gyro_thr)
        if len(above):
            quiet = int(round(0.5 / dt))
            end = len(g)
            for k in range(above[0], len(g) - quiet):
                if (g[k:k + quiet] <= gyro_thr).all():
                    end = k
                    break
            stab = (end - above[0]) * dt
        else:
            stab = 0.0
        c = self.contacts[start:]
        changes = sum(c[k] != c[k - 1] for k in range(1, len(c)))
        base_eff = np.mean(self.effort[baseline]) if len(self.effort[baseline]) else 1.0
        eff = np.max(self.effort[start:]) / max(1e-9, base_eff) if len(self.effort[start:]) else 1.0
        tilt = np.array(self.tilt)
        tilt_peak = float(tilt[start:].max() - tilt[baseline].mean()) if len(tilt[baseline]) else 0.0
        return Evidence(float(g.max()) if len(g) else 0.0, float(a.max()) if len(a) else 0.0, float(stab),
                        int(changes), float(eff), tilt_peak, bool(np.max(self.tilt[start:]) > 45.0))


def standing_encounter(seed: int, etype: str, distance: float, dwell_s: float = 4.0, settle_s: float = 1.0,
                       policy: str = "neutral") -> tuple[Evidence, dict]:
    """J3/J4: the robot stands at `distance` from the entity for `dwell_s` (after settling); lunges during the dwell.

    Returns (robot-observable evidence, evaluation-only truth). Needs NERVA."""
    from apm_express.adapter import NEUTRAL_REQUEST
    from apm_express.biped import NervaBiped

    rng = np.random.default_rng(seed)
    ctrl = NervaBiped(policy=policy, seed=int(rng.integers(2**31)))
    ctrl.apply(NEUTRAL_REQUEST)
    log = ImuLogger(np.random.default_rng(int(rng.integers(2**31))))
    n_settle, n_dwell = int(round(settle_s / ctrl.dt)), int(round(dwell_s / ctrl.dt))
    events = {int(round(t / ctrl.dt)): (reach, speed) for t, reach, speed in lunges(rng, etype, dwell_s)}
    pushes = []
    for k in range(n_settle + n_dwell):
        j = k - n_settle
        if j in events:
            reach, speed = events[j]
            m = impulse(reach, speed, distance)
            if m > 0:
                ang = math.pi + float(rng.uniform(-math.radians(20), math.radians(20)))  # entity ahead, push back
                ctrl.push(m * math.cos(ang), m * math.sin(ang))
                pushes.append(round(m, 4))
        row = ctrl.advance()
        log.read(ctrl, row)
    ev = log.summarise(n_settle, slice(max(0, n_settle - 25), n_settle), ctrl.dt)
    truth = {"type": etype, "lunges": len(events), "contacts": len(pushes), "max_push": max(pushes, default=0.0),
             "sum_push": round(sum(pushes), 4)}
    return ev, truth
