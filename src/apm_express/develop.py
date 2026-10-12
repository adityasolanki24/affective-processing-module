"""Developmental testbed for H3–H6: robot lifetimes of encounters with entities whose outcomes are physically felt.

A lifetime is a sequence of encounters. In each one the robot approaches an entity (an identity with a visible kind)
in a visible floor zone, stops, and the entity either does nothing or "jolts" it. A jolt is a real horizontal push
applied in NERVA's MuJoCo simulation, within the controller's validated robustness (0.30 m/s, NERVA's gate level).

What the robot *observes* is the measured physical consequence: peak tilt above its pre-push baseline and base
speed. It never observes the world's hidden jolt probability, and nothing labels the robot's state. Whether a jolt
happens is decided by the world schedule (the experimental manipulation). The outcome is measured from physics.

Design simplification (stated in the H3/H4 addendum): experience streams are generated with neutral motion (X0), so
every internal-state model sees an *identical* experience stream (common random numbers). Expression is therefore
communicative in H4: it does not change what the robot experiences.

World schedules (`HISTORIES`): the experimental histories. Each is defined only by jolt probabilities per
kind / zone / phase.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

KINDS = ("A", "B", "C")
ZONES = ("plain", "marked")
IDENTITIES = ("A1", "A2", "B1", "B2", "C1", "C2")  # seen during lifetimes
NOVEL = ("A3", "B3", "C3")  # held out: never encountered during a lifetime, used only in evaluation probes
UNKNOWN_KIND = "D"  # a kind never encountered, for disposition probes ("D1")
LOW, HIGH = 0.05, 0.85
JOLT_SPEED = 0.30  # m/s, NERVA gate push magnitude
N_ENCOUNTERS = 48


def kind_of(identity: str) -> str:
    return identity[0]


def jolt_probability(history: str, identity: str, zone: str, k: int, n: int = N_ENCOUNTERS) -> float:
    """The world's hidden rule (never observed by the robot)."""
    kind = kind_of(identity)
    first_half = k < n // 2
    if history == "safe":
        return LOW
    if history == "risky_B":
        return HIGH if kind == "B" else LOW
    if history == "risky_A":
        return HIGH if kind == "A" else LOW
    if history == "unpredictable":
        return 0.35
    if history == "change_B":  # B risky in the first half, then safe: revision
        return (HIGH if first_half else LOW) if kind == "B" else LOW
    if history == "late_B":  # safe first, B risky in the second half: acquisition late in life
        return (LOW if first_half else HIGH) if kind == "B" else LOW
    if history == "zone_B":  # B risky only in the marked zone: context dependence
        return HIGH if (kind == "B" and zone == "marked") else LOW
    raise ValueError(history)


HISTORIES = ("safe", "risky_B", "risky_A", "unpredictable", "change_B", "late_B", "zone_B")


@dataclass(frozen=True)
class Event:
    k: int  # encounter index within the lifetime
    identity: str
    kind: str
    zone: str
    jolted: bool  # world draw (hidden from the robot; kept for evaluation only)
    peak_tilt: float  # observed: max tilt above pre-push baseline over 4 s [deg]
    peak_speed: float  # observed: max horizontal base speed over 4 s [m/s]
    perturbed: bool  # observed outcome label: peak_tilt >= TILT_EVENT_DEG

    def as_dict(self) -> dict:
        return asdict(self)


TILT_EVENT_DEG = 3.0  # observed outcome threshold; fixed from the H3 pilot in the addendum (default here)


def schedule(history: str, rng: np.random.Generator, n: int = N_ENCOUNTERS) -> list[tuple[str, str, bool]]:
    """(identity, zone, jolted) per encounter; identities and zones uniform; jolts drawn from the hidden rule."""
    out = []
    for k in range(n):
        ident = IDENTITIES[int(rng.integers(len(IDENTITIES)))]
        zone = ZONES[int(rng.integers(len(ZONES)))]
        out.append((ident, zone, bool(rng.random() < jolt_probability(history, ident, zone, k, n))))
    return out


def measure_encounter(seed: int, jolted: bool, direction: float, settle_s: float = 1.0, after_s: float = 4.0,
                      policy: str = "neutral") -> tuple[float, float]:
    """Physically simulate the outcome moment: the robot standing (neutral rest), optionally pushed.

    Returns (peak tilt above the pre-push mean, peak horizontal speed) over `after_s` seconds. Needs NERVA."""
    from apm_express.adapter import NEUTRAL_REQUEST
    from apm_express.biped import NervaBiped

    ctrl = NervaBiped(policy=policy, seed=seed)
    ctrl.apply(NEUTRAL_REQUEST)
    pre = [ctrl.advance()["tilt"] for _ in range(int(round(settle_s / ctrl.dt)))]
    if jolted:
        ctrl.push(JOLT_SPEED * math.cos(direction), JOLT_SPEED * math.sin(direction))
    rows = [ctrl.advance() for _ in range(int(round(after_s / ctrl.dt)))]
    tilt = np.array([r["tilt"] for r in rows])
    speed = np.hypot([r["vx_world"] for r in rows], [r["vy_world"] for r in rows])
    return float(tilt.max() - np.mean(pre[-25:])), float(speed.max())
