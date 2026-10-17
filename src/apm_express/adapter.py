"""EmbodimentExpressionAdapter for NERVA's biped (programme H, H0).

The validated controller accepts only the seven trained neutral commands (rest, forward/backward/left/right at
0.074 m/s, turns at ±0.60 rad/s) with a fixed 0.54 s gait clock. Expression therefore lives in:
- *which* validated command is issued *when* (timing, pauses, path, orientation);
- the E1′ torso-pitch input;
- a head-offset channel, only if H0 admits it.

The adapter never produces anything else. Every output passes three filters:
- the command must be one of the seven;
- a command must be held for at least `min_dwell_s` before it changes;
- pitch and head offsets are clipped to the admitted envelope and rate-limited.

Behaviour per intent channel (`apm_express.intent`), approaching a target:
- **proximity**: final stopping distance, 0.6 m (proximity 0) to 0.2 m (proximity 1).
- **directness**: below 0.95, the path goes through a lateral waypoint offset (1 − directness)·0.4 m from the
  straight line.
- **tempo**: progress duty cycle within a 2.5 s rhythm. Forward bursts last at least 1.5 s. Tempo is floored at
  0.25.
- **task guarantee** (H0b): if average approach progress drops below 0.025 m/s (after 4 s), the adapter switches
  to catch-up. In catch-up there are no new pauses, retreats or scanning stops, and tempo is 1, until progress
  recovers to 0.03 m/s. Expression can slow the approach but cannot stall it.
- **hesitation**: Poisson pauses at 0.25·h per second, lasting 0.8 + 1.2·h seconds. With probability h a pause is
  preceded by a short backward step, as long as cumulative backward time stays ≤ 25% of forward time. An extra
  pause is likely when the target is near.
- **scanning**: look-around pauses at 0.15·s per second. During pauses and after arrival the robot sweeps: with
  the head channel it moves the head; without it, it turns its body (amplitude s·0.6 rad).
- **gaze**: after arrival and during pauses the robot turns toward the target (gaze > 0.3) or away from it by
  |gaze|·1.2 rad (gaze < −0.3). With the head channel, head yaw follows the same rule.
- **lean**: the E1′ e_pitch value (+ leans forward), if that channel is admitted.
- **head_elevation**: head_pitch, if the head channel is admitted (+ face up, as measured in NERVA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from apm_express.intent import ExpressionIntent

COMMANDS = {"rest": (0.0, 0.0, 0.0), "forward": (0.074, 0.0, 0.0), "backward": (-0.074, 0.0, 0.0),
            "left": (0.0, 0.074, 0.0), "right": (0.0, -0.074, 0.0), "turn_left": (0.0, 0.0, 0.60),
            "turn_right": (0.0, 0.0, -0.60)}
COMMAND_NAMES = tuple(COMMANDS)
TRANSLATING = frozenset({"forward", "backward", "left", "right"})
D_FAR, D_NEAR = 0.60, 0.20  # m, stopping distance at proximity 0 and 1
TURN_START, TURN_STOP = 0.35, 0.15  # rad, bearing error hysteresis for turning in place
RHYTHM_S = 2.5
TEMPO_FLOOR = 0.25
WAYPOINT_OFFSET_M = 0.40
WAYPOINT_REACHED_M = 0.15
GAZE_DEADBAND = 0.3
AVERT_RAD = 1.2
BODY_SCAN_RAD = 0.6
SCAN_HZ = 0.25
HEAD_SCAN_HZ = 0.3
HEAD_EPS = 0.02  # rad: the head counts as returned to neutral
RETREAT_CAP = 0.25  # cumulative backward time <= this fraction of forward time, so progress always dominates
MIN_BURST_S = 1.5  # forward segments of the tempo rhythm last at least this long (shorter bursts barely start the gait)
V_MIN_PROGRESS = 0.025  # m/s: average approach progress below this switches on catch-up (task guarantee, H0b)
PROGRESS_GRACE_S = 4.0


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class BipedChannels:
    """What H0 admitted. Defaults are the conservative interface: commands only, 1 s dwell, no pitch, no head."""

    min_dwell_s: float = 1.0
    pitch: bool = False
    pitch_rate: float = 1.0  # max |de_pitch/dt| per second
    head: bool = False
    head_pitch_range: tuple[float, float] = (0.0, 0.0)  # rad
    head_yaw_range: tuple[float, float] = (0.0, 0.0)  # rad
    head_rate: float = 1.0  # rad/s per joint
    head_while_moving: bool = False  # False: head offsets only while resting; the head returns before moving


@dataclass(frozen=True)
class MotionRequest:
    command: str
    e_pitch: float
    head: tuple[float, float, float, float]  # neck_pitch, head_pitch, head_yaw, head_roll [rad]


NEUTRAL_REQUEST = MotionRequest("rest", 0.0, (0.0, 0.0, 0.0, 0.0))


@dataclass
class BipedExpressionAdapter:
    channels: BipedChannels = field(default_factory=BipedChannels)
    seed: int = 0
    dt: float = 0.02

    def __post_init__(self) -> None:
        self.reset(0.0, (0.0, 0.0, 0.0), None)

    # ── state ───────────────────────────────────────────────────────────────
    def reset(self, t: float, pose, target, intent: ExpressionIntent | None = None) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._command, self._since = "rest", t - 1e9
        self._e_pitch = 0.0
        self._head: np.ndarray = np.zeros(4)
        self._turning = False
        self._arrived = False
        self._pause_until = -1.0
        self._retreat_until = -1.0
        self._scan_pause_until = -1.0
        self._near_pause_done = False
        self._catch_up = False
        self._d0 = None if target is None else float(np.linalg.norm(np.asarray(target, dtype=float)
                                                                     - np.asarray(pose[:2], dtype=float)))
        self._t_target = t
        self._forward_s = 0.0
        self._backward_s = 0.0
        self._t0 = t
        self._ref_yaw = float(pose[2])  # orientation reference when there is no target
        self._target = None if target is None else np.asarray(target, dtype=float)
        self._waypoint = None
        self._waypoint_side = 1.0 if self._rng.random() < 0.5 else -1.0
        if self._target is not None and intent is not None:
            self._plan_waypoint(np.asarray(pose[:2], dtype=float), intent.sanitized())

    def notify_fallback(self, t: float) -> None:
        """The safety monitor took over: forget timers and offsets, restart from neutral rest."""
        self._command, self._since = "rest", t
        self._e_pitch, self._head = 0.0, np.zeros(4)
        self._turning = False
        self._pause_until = self._retreat_until = self._scan_pause_until = -1.0

    def _plan_waypoint(self, start: np.ndarray, intent: ExpressionIntent) -> None:
        self._waypoint = None
        if intent.directness < 0.95 and self._target is not None:
            d = self._target - start
            length = float(np.linalg.norm(d))
            if length > 2 * WAYPOINT_REACHED_M:
                normal = np.array([-d[1], d[0]]) / length
                offset = self._waypoint_side * WAYPOINT_OFFSET_M * (1 - intent.directness)
                self._waypoint = start + 0.5 * d + offset * normal

    # ── main entry ──────────────────────────────────────────────────────────
    def request(self, t: float, pose, target, intent: ExpressionIntent) -> MotionRequest:
        intent = intent.sanitized()
        if target is not None:
            target = np.asarray(target, dtype=float)
            if self._target is None or not np.allclose(target, self._target):
                self._target = target
                self._arrived = False
                self._near_pause_done = False
                self._catch_up = False
                self._d0 = float(np.linalg.norm(target - np.asarray(pose[:2], dtype=float)))
                self._t_target = t
                self._plan_waypoint(np.asarray(pose[:2], dtype=float), intent)
        else:
            self._target = None
        desired = self._desired_command(t, pose, intent)
        if desired not in COMMANDS:  # defensive: the planner only produces known names
            desired = "rest"
        head_hold = self.channels.head and not self.channels.head_while_moving
        wants_motion = desired != "rest"
        if head_hold and wants_motion and float(np.max(np.abs(self._head))) > HEAD_EPS:
            desired = "rest"  # bring the head back to neutral before any locomotion command
        if desired != self._command and t - self._since >= self.channels.min_dwell_s - 1e-9:
            self._command, self._since = desired, t
        if self._command == "forward":
            self._forward_s += self.dt
        elif self._command == "backward":
            self._backward_s += self.dt
        head_allowed = not head_hold or (self._command == "rest" and not wants_motion)
        return MotionRequest(self._command, self._pitch_step(intent), self._head_step(t, pose, intent, head_allowed))

    # ── channels ────────────────────────────────────────────────────────────
    def _pitch_step(self, intent: ExpressionIntent) -> float:
        goal = float(np.clip(intent.lean, -1.0, 1.0)) if self.channels.pitch else 0.0
        step = self.channels.pitch_rate * self.dt
        self._e_pitch = float(np.clip(goal, self._e_pitch - step, self._e_pitch + step))
        return self._e_pitch

    def _head_step(self, t: float, pose, intent: ExpressionIntent,
                   allowed: bool = True) -> tuple[float, float, float, float]:
        goal = np.zeros(4)
        ch = self.channels
        if ch.head and allowed:
            lo_p, hi_p = ch.head_pitch_range
            lo_y, hi_y = ch.head_yaw_range
            goal[1] = intent.head_elevation * (hi_p if intent.head_elevation > 0 else -lo_p)
            yaw_max = min(abs(lo_y), abs(hi_y))
            bearing = self._bearing(pose, self._target) if self._target is not None else 0.0
            if intent.gaze > GAZE_DEADBAND:
                gaze_yaw = bearing
            elif intent.gaze < -GAZE_DEADBAND:
                gaze_yaw = -math.copysign(abs(intent.gaze) * yaw_max, bearing if bearing != 0 else 1.0)
            else:
                gaze_yaw = 0.0
            scan = intent.scanning * yaw_max * math.sin(2 * math.pi * HEAD_SCAN_HZ * (t - self._t0))
            goal[2] = gaze_yaw + scan
            goal[1] = float(np.clip(goal[1], lo_p, hi_p))
            goal[2] = float(np.clip(goal[2], lo_y, hi_y))
        step = ch.head_rate * self.dt
        self._head = np.clip(goal, self._head - step, self._head + step)
        h = [float(v) for v in self._head]
        return h[0], h[1], h[2], h[3]

    # ── planner ─────────────────────────────────────────────────────────────
    @staticmethod
    def _bearing(pose, point) -> float:
        dx, dy = float(point[0]) - float(pose[0]), float(point[1]) - float(pose[1])
        return wrap(math.atan2(dy, dx) - float(pose[2]))

    def _turn_toward(self, error: float) -> str:
        if self._turning:
            if abs(error) < TURN_STOP:
                self._turning = False
                return "rest"
        elif abs(error) > TURN_START:
            self._turning = True
        if not self._turning:
            return "rest"
        return "turn_left" if error > 0 else "turn_right"

    def _orient(self, t: float, pose, intent: ExpressionIntent) -> str:
        """Orientation behaviour while stationary (pauses and after arrival)."""
        if self._target is not None:
            bearing = self._bearing(pose, self._target)
        else:
            bearing = wrap(self._ref_yaw - float(pose[2]))
        offset = 0.0
        if intent.gaze < -GAZE_DEADBAND:
            offset = math.copysign(abs(intent.gaze) * AVERT_RAD, bearing if bearing != 0 else 1.0)
        if not self.channels.head and intent.scanning > 0.05:
            phase = math.sin(2 * math.pi * SCAN_HZ * (t - self._t0))
            offset += intent.scanning * BODY_SCAN_RAD * (1.0 if phase >= 0 else -1.0)
        elif abs(intent.gaze) <= GAZE_DEADBAND:
            self._turning = False
            return "rest"
        return self._turn_toward(wrap(bearing - offset))

    def _desired_command(self, t: float, pose, intent: ExpressionIntent) -> str:
        if self._target is None:
            return self._orient(t, pose, intent) if intent.scanning > 0.05 else "rest"
        pos = np.asarray(pose[:2], dtype=float)
        dist = float(np.linalg.norm(self._target - pos))
        d_stop = D_FAR - (D_FAR - D_NEAR) * intent.proximity
        if self._arrived and dist > d_stop + 0.15:
            self._arrived = False
        if not self._arrived and dist <= d_stop:
            self._arrived = True
            self._turning = False
        if self._arrived:
            return self._orient(t, pose, intent)

        elapsed = t - self._t0
        since = t - self._t_target
        if self._d0 is not None and since > PROGRESS_GRACE_S:
            rate = (self._d0 - dist) / since
            if rate < V_MIN_PROGRESS:
                self._catch_up = True
            elif rate >= 1.2 * V_MIN_PROGRESS:
                self._catch_up = False
        if t < self._retreat_until:
            return "backward"
        if t < self._pause_until or t < self._scan_pause_until:
            return self._orient(t, pose, intent)
        if elapsed > 1.0 and not self._catch_up:
            rate = 0.25 * intent.hesitation
            near = dist <= d_stop + 0.25
            if near and not self._near_pause_done:
                self._near_pause_done = True
                if self._rng.random() < intent.hesitation:
                    rate = 1.0 / self.dt  # certain pause at the approach boundary
            if self._rng.random() < rate * self.dt:
                pause = 0.8 + 1.2 * intent.hesitation
                retreat = max(self.channels.min_dwell_s, 0.6 * intent.hesitation)
                if (self._rng.random() < intent.hesitation
                        and self._backward_s + retreat <= RETREAT_CAP * self._forward_s):
                    self._retreat_until = t + retreat
                    self._pause_until = self._retreat_until + pause
                    return "backward"
                self._pause_until = t + pause
                return self._orient(t, pose, intent)
            if self._rng.random() < 0.15 * intent.scanning * self.dt:
                self._scan_pause_until = t + 1.5
                return self._orient(t, pose, intent)

        goal = self._target
        if self._waypoint is not None:
            if float(np.linalg.norm(self._waypoint - pos)) <= WAYPOINT_REACHED_M:
                self._waypoint = None
            else:
                goal = self._waypoint
        error = self._bearing(pose, goal)
        turn = self._turn_toward(error)
        if turn != "rest":
            return turn
        tempo = 1.0 if self._catch_up else max(TEMPO_FLOOR, intent.tempo)
        if tempo < 1.0:
            go = max(tempo * RHYTHM_S, MIN_BURST_S, self.channels.min_dwell_s)
            period = go / tempo
            if (elapsed % period) >= go:
                return "rest"
        return "forward"
