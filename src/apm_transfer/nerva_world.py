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
| pushy | 0.25 | 0.35–0.80 | 0.5–0.9 | frequent, moderate (added after pilot 1) |

Pilot 1 showed that most contacts were below the range the IMU can sense, hence the pushy type.

Evidence v2 (pilot 2), appended after the v1 fields, all robot-observable:
- `base_peak_acc`: peak horizontal acceleration deviation during the robot's own quiet settle window. This is the
  robot's self-calibrated noise reference.
- `acc_energy`: Σ over the dwell of max(0, |acc deviation| − base_peak_acc)², a filtered impact energy.
- `detected`: peak dwell acceleration deviation > 1.5 × base_peak_acc (self-calibrated detector).
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
    "pushy": {"rate": 0.25, "reach": (0.35, 0.80), "speed": (0.5, 0.9)},
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
    base_peak_acc: float = 0.0
    acc_energy: float = 0.0
    detected: bool = False
    quiet_frames: int = -1  # frames left after efference-copy gating (-1: no gating, a standing encounter)

    def as_list(self) -> list:
        return [round(self.peak_gyro, 5), round(self.peak_acc, 5), round(self.stab_time, 3), self.contact_changes,
                round(self.effort_ratio, 4), round(self.tilt_peak, 3), self.fell, round(self.base_peak_acc, 5),
                round(self.acc_energy, 6), self.detected, self.quiet_frames]


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

    def summarise(self, start: int, baseline: slice, dt: float, gyro_thr: float = 0.35,
                  quiet: np.ndarray | None = None, stop: int | None = None) -> Evidence:
        """Evidence over frames start..stop (default: to the last frame). `quiet` (efference-copy gate) keeps only
        frames the robot did not move itself."""
        g = np.array(self.gyro[start:stop])
        acc = np.array(self.acc)
        base_acc = acc[baseline].mean(0) if len(acc[baseline]) else np.zeros(2)
        a = np.linalg.norm(acc[start:stop] - base_acc, axis=1)
        n_quiet = -1
        if quiet is not None:
            mask = np.asarray(quiet[start:start + len(g)], bool)
            g, a = g[mask], a[mask]
            n_quiet = int(mask.sum())
        above = np.flatnonzero(g > gyro_thr)
        if len(above):
            settle_n = int(round(0.5 / dt))
            end = len(g)
            for k in range(above[0], len(g) - settle_n):
                if (g[k:k + settle_n] <= gyro_thr).all():
                    end = k
                    break
            stab = (end - above[0]) * dt
        else:
            stab = 0.0
        c = self.contacts[start:stop]
        changes = sum(c[k] != c[k - 1] for k in range(1, len(c)))
        base_eff = np.mean(self.effort[baseline]) if len(self.effort[baseline]) else 1.0
        eff = np.max(self.effort[start:stop]) / max(1e-9, base_eff) if len(self.effort[start:stop]) else 1.0
        tilt = np.array(self.tilt)
        tilt_peak = float(tilt[start:stop].max() - tilt[baseline].mean()) if len(tilt[baseline]) else 0.0
        base_dev = np.linalg.norm(acc[baseline] - base_acc, axis=1) if len(acc[baseline]) else np.zeros(1)
        base_peak = float(np.max(base_dev))
        energy = float(np.sum(np.clip(a - base_peak, 0.0, None) ** 2)) if len(a) else 0.0
        peak = float(np.max(a)) if len(a) else 0.0
        return Evidence(float(g.max()) if len(g) else 0.0, peak, float(stab), int(changes), float(eff), tilt_peak,
                        bool(np.max(self.tilt[start:stop]) > 45.0), base_peak, energy,
                        bool(peak > 1.5 * base_peak),
                        n_quiet)


TARGET = (0.9, 0.0)  # the entity stands at the approach target (as H's ball), 0.9 m ahead


POST_ARRIVAL_SKIP_S = 1.0  # evidence starts 1 s after arrival, past the gait-stopping transient
SELF_MOTION_HOLD_S = 0.6   # efference copy: frames within 0.6 s after the robot's own commanded motion are ignored
MIN_QUIET_FRAMES = 25      # fewer quiet frames than this (0.5 s): the encounter is unobservable (no evidence)
WINDOW_S = 4.0             # programme K: one sensing window (the J3/J5 window length)
MAX_WINDOWS = 3            # programme K: at most 3 windows per encounter; lunge streams are drawn for this horizon


@dataclass(frozen=True)
class Camera:
    """Programme K **sensor extension** (not on NERVA's simulated robot today): a head camera that sees the entity's
    lunges, whether or not they reach the robot. A lunge during a sensing window is seen with probability `p_detect`
    if the entity is within `fov` of the robot's heading and within `range_m`. Its reach is measured with Gaussian
    noise `sigma` (m). It reads the entity's *motion*, never its type."""
    fov: float = math.radians(40)
    range_m: float = 2.0
    p_detect: float = 0.9
    sigma: float = 0.05


def camera_code(reaches: list[float]) -> int:
    """8 categories: 0 none seen; 1-3 one lunge, max reach < 0.35 / 0.35-0.55 / > 0.55 m; 4-6 two lunges, same
    bins; 7 three or more."""
    if not reaches:
        return 0
    if len(reaches) >= 3:
        return 7
    b = int(np.searchsorted((0.35, 0.55), max(reaches)))
    return 1 + 3 * (len(reaches) - 1) + b


def closed_loop_encounter(seed: int, etype: str, intent, channels: dict, dwell_s: float = 5.0, settle_s: float = 2.0,
                          max_approach_s: float = 45.0, sensing_pause: bool = True, scene_extender=None,
                          frame_hook=None, windows: int | None = None, camera: Camera | None = None,
                          abort_on_detect: bool = True) -> tuple[Evidence, dict, dict]:
    """J5: one encounter in which the robot's own expression shapes what happens.

    The robot settles, then approaches the entity under `intent`, through H's `BipedExpressionAdapter` with the
    admitted `channels` and H's `SafetyMonitor` (neutral fallback). It stands for `dwell_s` after arriving. The
    entity's lunges happen throughout the approach and the dwell; each contacts only if its reach exceeds the robot's
    *current* distance to the entity. So the stopping distance, approach timing and hesitation (exposure) change what
    happens.

    Evidence is summarised over the standing dwell from 1 s after arrival (past the gait-stopping transient) to the
    end: 4 s, the same window length as J3's bank. The baseline is the settle period.

    **Efference-copy gate.** The robot knows its own commands. Frames during, and up to 0.6 s after, its own commanded
    motion are excluded: walking or turning commands, head-offset changes, or the safety fallback. Expressive motion
    after arrival (gaze turns, scanning, head movement) is therefore not mistaken for being pushed. The J5 pilot
    showed that without this gate, beliefs saturate in the closed loop.

    **Sensing pause** (`sensing_pause=True`, J5 protocol after pilot 2). From arrival to the end of the dwell, the
    robot holds the neutral request (rest, head returning to neutral) instead of continuing post-arrival expression
    (gaze turns, scanning). Head motion alone raises the IMU peak to 1.1–1.9 (still: about 0.45), so a continuously
    scanning robot has no quiet frames and cannot sense contact at all. Expression still sets the stopping distance,
    approach tempo, hesitation and path, i.e. everything that shapes exposure.

    **Rendering only** (`j_videos.py`): `scene_extender` adds visual-only geometry; `frame_hook(ctrl, info)` is called
    after every control step with the entity's lunge on that step (reach, contact or not), the distance and the phase.
    Neither changes the simulation.

    **Programme K options** (the defaults reproduce J exactly):
    - `windows=n` (1-3): the robot stands for 1 s plus n sensing windows of 4 s each (protected observation). Each
      window is summarised separately, in `outcome["windows"]`. Lunge streams are drawn for the longest possible
      encounter, so systems choosing different n meet the same entity behaviour. With `abort_on_detect`, if the
      robot's own detector fires in a window and more windows were planned, it stops observing and leaves: it never
      stays to collect repeated pushes.
    - `camera=Camera()`: the sensor extension above. Its codes per window go in `outcome["camera"]`.

    Returns:
    - robot-observable evidence;
    - outcome: arrival time, arrived, final distance, safety trips, fallback steps, fell, stopping distance — task and
      safety measures;
    - truth: contacts and pushes, for evaluation only.
    """
    from apm_express.adapter import NEUTRAL_REQUEST, BipedChannels, BipedExpressionAdapter
    from apm_express.biped import NervaBiped
    from apm_express.safety import MonitorLimits, SafetyMonitor

    rng = np.random.default_rng(seed)
    ch = BipedChannels(min_dwell_s=channels["min_dwell_s"], pitch=channels["pitch"],
                       head=channels["head"], head_pitch_range=tuple(channels["head_pitch_range"]),
                       head_yaw_range=tuple(channels["head_yaw_range"]), head_rate=channels["head_rate"],
                       head_while_moving=False)
    limits = MonitorLimits(joint_margin_rad=channels.get("joint_margin_rad"),
                           saturation_steps=channels.get("saturation_steps"))
    ctrl = NervaBiped(policy=channels["policy"], seed=int(rng.integers(2**31)), head_channel=ch.head,
                      max_head_rate=1.5, max_pitch_rate=2.0, scene_extender=scene_extender)
    adapter, monitor = BipedExpressionAdapter(ch, seed=int(rng.integers(2**31))), SafetyMonitor(limits)
    log = ImuLogger(np.random.default_rng(int(rng.integers(2**31))))
    n_settle = int(round(settle_s / ctrl.dt))
    ctrl.apply(NEUTRAL_REQUEST)
    for _ in range(n_settle):
        log.read(ctrl, ctrl.advance())
    if windows is not None:
        dwell_s = POST_ARRIVAL_SKIP_S + WINDOW_S * windows
        window = max_approach_s + POST_ARRIVAL_SKIP_S + WINDOW_S * MAX_WINDOWS
    else:
        window = max_approach_s + dwell_s
    events = {n_settle + int(round(t / ctrl.dt)): (reach, speed) for t, reach, speed in lunges(rng, etype, window)}
    cam_rng = np.random.default_rng([seed, 7])
    win_n = int(round(WINDOW_S / ctrl.dt))
    skip_n = int(round(POST_ARRIVAL_SKIP_S / ctrl.dt))
    cam_seen: list[list[float]] = [[] for _ in range(windows or 0)]
    win_ev: list[Evidence] = []
    aborted = False
    adapter.reset(ctrl.t, ctrl.pose(), TARGET, intent)
    pushes, t_arrive, k_arrive, fell, fallback = [], None, None, False, 0
    own = [False] * n_settle  # per frame: did the robot itself command motion?
    prev_head: np.ndarray = np.zeros(4)
    k = n_settle
    while True:
        x, y, _ = ctrl.pose()
        dist = math.hypot(TARGET[0] - x, TARGET[1] - y)
        lunge = None
        if k in events:
            reach, speed = events[k]
            m = impulse(reach, speed, dist)
            lunge = (reach, m)
            if camera is not None and windows is not None and k_arrive is not None and k >= k_arrive + skip_n:
                w = (k - k_arrive - skip_n) // win_n
                bearing = math.atan2(TARGET[1] - y, TARGET[0] - x) - ctrl.pose()[2]
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                if (w < windows and abs(bearing) <= camera.fov and dist <= camera.range_m
                        and cam_rng.random() < camera.p_detect):
                    cam_seen[w].append(round(reach + float(cam_rng.normal(0.0, camera.sigma)), 3))
            if m > 0:
                away = math.atan2(y - TARGET[1], x - TARGET[0]) + float(rng.uniform(-math.radians(20),
                                                                                    math.radians(20)))
                ctrl.push(m * math.cos(away), m * math.sin(away))
                pushes.append((round(ctrl.t, 2), round(m, 4), k_arrive is not None))
        tripped = monitor.update(ctrl.t, **ctrl.physical())
        if tripped:
            fallback += 1
            adapter.notify_fallback(ctrl.t)
            request = NEUTRAL_REQUEST
        elif sensing_pause and k_arrive is not None:
            request = NEUTRAL_REQUEST  # hold still to sense (the gate still removes the head's return transient)
        else:
            request = adapter.request(ctrl.t, ctrl.pose(), TARGET, intent)
        ctrl.apply(request)
        head = np.asarray(request.head, float)
        own.append(bool(tripped or request.command != "rest" or np.abs(head - prev_head).max() > 1e-3))
        prev_head = head
        row = ctrl.advance()
        log.read(ctrl, row)
        if frame_hook is not None:
            frame_hook(ctrl, {"lunge": lunge, "dist": dist, "arrived": k_arrive is not None, "fallback": tripped})
        if row["tilt"] > 45.0 or not row["finite"]:
            fell = True
            break
        if k_arrive is None and adapter._arrived:
            k_arrive, t_arrive = k + 1, round(ctrl.t - settle_s, 2)
        k += 1
        if (windows is not None and k_arrive is not None and k > k_arrive + skip_n
                and (k - k_arrive - skip_n) % win_n == 0):
            w0 = k_arrive + skip_n + len(win_ev) * win_n
            hold_n = int(round(SELF_MOTION_HOLD_S / ctrl.dt))
            mv = np.asarray(own, bool)
            q = ~np.array([mv[max(0, j - hold_n):j + 1].any() for j in range(len(mv))])
            win_ev.append(log.summarise(w0, slice(max(0, n_settle - 50), n_settle), ctrl.dt, quiet=q, stop=k))
            if abort_on_detect and win_ev[-1].detected and len(win_ev) < windows:
                aborted = True
                break
        if k_arrive is not None and k - k_arrive >= int(round(dwell_s / ctrl.dt)):
            break
        if k - n_settle >= int(round(window / ctrl.dt)):
            break
    skip = int(round(POST_ARRIVAL_SKIP_S / ctrl.dt))
    start = k_arrive + skip if k_arrive is not None else max(n_settle, k - int(round((dwell_s - 1) / ctrl.dt)))
    hold = int(round(SELF_MOTION_HOLD_S / ctrl.dt))
    moved = np.asarray(own, bool)
    quiet = ~np.array([moved[max(0, j - hold):j + 1].any() for j in range(len(moved))])
    if windows is not None and win_ev:
        ev = win_ev[0]
    else:
        ev = log.summarise(start, slice(max(0, n_settle - 50), n_settle), ctrl.dt, quiet=quiet)
    x, y, _ = ctrl.pose()
    outcome: dict = {"arrived": k_arrive is not None, "t_arrive": t_arrive, "final_distance": round(math.hypot(
        TARGET[0] - x, TARGET[1] - y), 3), "trips": monitor.trips, "fallback_steps": fallback, "fell": fell}
    if windows is not None:
        outcome["windows"] = [e.as_list() for e in win_ev]
        outcome["aborted"] = aborted
        outcome["dwell_s"] = round((k - k_arrive) * ctrl.dt, 2) if k_arrive is not None else 0.0
        if camera is not None:
            outcome["camera"] = [camera_code(c) for c in cam_seen[:len(win_ev)]]
            outcome["camera_reaches"] = cam_seen[:len(win_ev)]
    truth = {"type": etype, "contacts": len(pushes), "contacts_dwell": sum(p[2] for p in pushes),
             "max_push": max((p[1] for p in pushes), default=0.0), "pushes": pushes}
    return ev, outcome, truth


def standing_encounter(seed: int, etype: str, distance: float, dwell_s: float = 4.0, settle_s: float = 2.0,
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
    ev = log.summarise(n_settle, slice(max(0, n_settle - 50), n_settle), ctrl.dt)
    truth = {"type": etype, "lunges": len(events), "contacts": len(pushes), "max_push": max(pushes, default=0.0),
             "sum_push": round(sum(pushes), 4)}
    return ev, truth
