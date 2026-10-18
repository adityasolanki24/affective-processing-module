"""Programme K4: one continuous NERVA lifetime in a single, never-reset MuJoCo simulation.

The robot lives in one physical world for its whole lifetime. Each cycle (an *encounter*):
1. **Home.** It stands still at its home spot for `settle_s`. These quiet frames are its detector's baseline.
2. **Approach.** It walks toward the entity under the chosen `ExpressionIntent`, through H's bounded adapter and
   H's safety monitor.
3. **Sense.** It holds still for 1 s, then for `windows` sensing windows of 4 s each: the efference-copy gate, the
   self-calibrated detector, and the optional `Camera` extension. It leaves early if its detector fires and more
   windows were planned.
4. **Return.** It walks back toward home.

The entity stands at `TARGET` and lunges as a Poisson process in continuous time, according to its *current* type.
The type is set by the caller per encounter and may change mid-lifetime. A lunge pushes the robot only if its reach
exceeds the robot's current distance, so where the robot walks and stands decides what happens to it. Nothing is
stitched: the trajectory, controller state, IMU and entity process are continuous across encounters.

The caller owns memory and decisions. `encounter()` returns the robot-observable evidence (per-window IMU summaries,
camera codes) and, separately, the evaluation-only truth.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from apm_transfer.nerva_world import (
    MIN_QUIET_FRAMES,
    POST_ARRIVAL_SKIP_S,
    SELF_MOTION_HOLD_S,
    TARGET,
    TYPES,
    WINDOW_S,
    Camera,
    ImuLogger,
    camera_code,
    impulse,
)

HOME_TARGET = (-0.45, 0.0)  # with a 0.4 m stopping distance the robot comes to rest near the origin
HOME_PROXIMITY = 0.5


class ContinuousNerva:
    def __init__(self, seed: int, channels: dict, camera: Camera | None = None, scene_extender=None,
                 frame_hook=None, max_leg_s: float = 45.0):
        from apm_express.adapter import BipedChannels, BipedExpressionAdapter
        from apm_express.biped import NervaBiped
        from apm_express.safety import MonitorLimits, SafetyMonitor

        self.rng = np.random.default_rng(seed)
        self.entity_rng = np.random.default_rng([seed, 1])
        self.cam_rng = np.random.default_rng([seed, 7])
        ch = BipedChannels(min_dwell_s=channels["min_dwell_s"], pitch=channels["pitch"], head=channels["head"],
                           head_pitch_range=tuple(channels["head_pitch_range"]),
                           head_yaw_range=tuple(channels["head_yaw_range"]), head_rate=channels["head_rate"],
                           head_while_moving=False)
        self.ctrl = NervaBiped(policy=channels["policy"], seed=int(self.rng.integers(2**31)), head_channel=ch.head,
                               max_head_rate=1.5, max_pitch_rate=2.0, scene_extender=scene_extender)
        self.adapter = BipedExpressionAdapter(ch, seed=int(self.rng.integers(2**31)))
        self.monitor = SafetyMonitor(MonitorLimits(joint_margin_rad=channels.get("joint_margin_rad"),
                                                   saturation_steps=channels.get("saturation_steps")))
        self.log = ImuLogger(np.random.default_rng(int(self.rng.integers(2**31))))
        self.camera, self.frame_hook, self.max_leg_s = camera, frame_hook, max_leg_s
        self.own: list[bool] = []
        self.prev_head: np.ndarray = np.zeros(4)
        self.etype = "calm"
        self.fell = False
        self.phase = "home"
        self.pushes: list[tuple[float, float]] = []
        self.lunges: list[tuple[float, float, float]] = []  # (t, reach, push) for every lunge (truth)
        self.t0 = self.ctrl.t
        # programme L Phase G: rendered vision. The entity's next lunge is drawn ahead (same Poisson process), so the
        # visible column can start moving before the lunge peaks. Without a VisionCamera nothing here is used and the
        # per-step draws reproduce K4 exactly.
        self.vision = camera is not None and type(camera).__name__ == "VisionCamera"
        self.k = 0
        self.events: dict[int, tuple[float, float]] = {}
        self.next_k: int | None = None
        self.vis: Any = None
        self.vis_renderer: Any = None  # one depth renderer for the whole lifetime (see VisionSession)
        self.vis_j0 = 0
        self.vis_end = 0

    def _draw_next(self) -> None:
        if self.next_k is not None:
            self.events.pop(self.next_k, None)
        p = TYPES[self.etype]
        wait = float(self.entity_rng.exponential(1.0 / p["rate"]))
        self.next_k = self.k + max(1, int(round(wait / self.ctrl.dt)))
        self.events[self.next_k] = (float(self.entity_rng.uniform(*p["reach"])),
                                    float(self.entity_rng.uniform(*p["speed"])))
        for kk in [kk for kk in self.events if kk < self.k - 200]:
            del self.events[kk]

    def entity_centre(self) -> tuple[float, float, float]:
        """Visual only: where the entity column stands now (vision mode), in world coordinates."""
        from apm_transfer.vision import ENTITY_H, LungeAnimator
        x, y, _ = self.ctrl.pose()
        dist = math.hypot(TARGET[0] - x, TARGET[1] - y)
        s = LungeAnimator(self.events, self.ctrl.dt).displacement(self.k, dist, 0.06) if self.vision else 0.0
        ox, oy = self.ctrl._origin
        n = dist or 1.0
        return (ox + TARGET[0] + s * (x - TARGET[0]) / n, oy + TARGET[1] + s * (y - TARGET[1]) / n, ENTITY_H / 2)

    # one control step ---------------------------------------------------------------------------------------
    def _step(self, request, info: dict) -> None:
        from apm_express.adapter import NEUTRAL_REQUEST

        x, y, _ = self.ctrl.pose()
        dist = math.hypot(TARGET[0] - x, TARGET[1] - y)
        lunge = None
        p = TYPES[self.etype]
        if self.vision:
            if self.next_k is None:
                self._draw_next()
            fire = self.k == self.next_k
        else:
            fire = self.entity_rng.random() < p["rate"] * self.ctrl.dt
        if fire:
            if self.vision:
                reach, speed = self.events[self.k]
                self.next_k = None  # keep the fired lunge in `events` (its visible return; truth)
                self._draw_next()
            else:
                reach, speed = float(self.entity_rng.uniform(*p["reach"])), float(self.entity_rng.uniform(*p["speed"]))
            m = impulse(reach, speed, dist)
            lunge = (reach, m)
            self.lunges.append((round(self.ctrl.t, 2), round(reach, 3), round(m, 4)))
            if m > 0:
                away = math.atan2(y - TARGET[1], x - TARGET[0]) + float(self.entity_rng.uniform(-0.35, 0.35))
                self.ctrl.push(m * math.cos(away), m * math.sin(away))
                self.pushes.append((round(self.ctrl.t, 2), round(m, 4)))
            cam_w = info.get("cam_window")
            if self.camera is not None and not self.vision and cam_w is not None:
                bearing = math.atan2(TARGET[1] - y, TARGET[0] - x) - self.ctrl.pose()[2]
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                if (self.camera.within_fov(bearing) and dist <= self.camera.range_m
                        and self.cam_rng.random() < self.camera.p_detect):
                    info["cam_seen"][cam_w].append(round(reach + float(self.cam_rng.normal(0, self.camera.sigma)),
                                                         3))
        tripped = self.monitor.update(self.ctrl.t, **self.ctrl.physical())
        if tripped:
            self.adapter.notify_fallback(self.ctrl.t)
            request = NEUTRAL_REQUEST
        self.ctrl.apply(request)
        head = np.asarray(request.head, float)
        self.own.append(bool(tripped or request.command != "rest" or np.abs(head - self.prev_head).max() > 1e-3))
        self.prev_head = head
        row = self.ctrl.advance()
        self.log.read(self.ctrl, row)
        if self.vis is not None:
            from apm_transfer.vision import WARM_S
            j = self.k - self.vis_j0
            if -int(round(WARM_S / self.ctrl.dt)) <= j < self.vis_end:
                self.vis.step(self.k, j, dist, self.own[-1])
        self.k += 1
        if row["tilt"] > 45.0 or not row["finite"]:
            self.fell = True
        if self.frame_hook is not None:
            self.frame_hook(self.ctrl, {"lunge": lunge, "dist": dist, "phase": self.phase, "fallback": tripped,
                                        "etype": self.etype})

    def _quiet(self, k0: int) -> np.ndarray:
        """Efference-copy gate for frames k0.. (frames before k0 are left True and never read)."""
        hold = int(round(SELF_MOTION_HOLD_S / self.ctrl.dt))
        mv = np.asarray(self.own, bool)
        q = np.ones(len(mv), bool)
        q[k0:] = ~np.array([mv[max(0, j - hold):j + 1].any() for j in range(k0, len(mv))], bool)
        return q

    def _walk(self, target, intent) -> float | None:
        self.adapter.reset(self.ctrl.t, self.ctrl.pose(), target, intent)
        t_start = self.ctrl.t
        while self.ctrl.t - t_start < self.max_leg_s and not self.fell:
            req = self.adapter.request(self.ctrl.t, self.ctrl.pose(), target, intent)
            self._step(req, {})
            if self.adapter._arrived:
                return round(self.ctrl.t - t_start, 2)
        return None

    def _hold(self, seconds: float, info: dict | None = None) -> None:
        from apm_express.adapter import NEUTRAL_REQUEST
        for _ in range(int(round(seconds / self.ctrl.dt))):
            if self.fell:
                return
            self._step(NEUTRAL_REQUEST, info or {})

    # one encounter --------------------------------------------------------------------------------------------
    def encounter(self, etype: str, intent, windows: int, settle_s: float = 2.0, abort_on_detect: bool = True,
                  standoff: float = 0.0) -> dict:
        if self.vision and etype != self.etype:
            self.etype = etype
            self._draw_next()  # the entity's behaviour changed: its pending lunge follows the new type
        self.etype = etype
        n_push0, n_lunge0 = len(self.pushes), len(self.lunges)
        self.phase = "home"
        k_base0 = len(self.log.gyro)
        self._hold(settle_s)
        baseline = slice(max(k_base0, len(self.log.gyro) - 50), len(self.log.gyro))
        self.phase = "approach"
        t_arrive = self._walk((TARGET[0] - standoff, TARGET[1]), intent)
        x, y, _ = self.ctrl.pose()
        d_stop = round(math.hypot(TARGET[0] - x, TARGET[1] - y), 3)
        self.phase = "sense"
        vis_out = None
        if self.vision and t_arrive is not None:
            import mujoco

            from apm_transfer.vision import VisionSession
            if self.vis_renderer is None:
                self.vis_renderer = mujoco.Renderer(self.ctrl.sim.model, self.camera.height,  # type: ignore[union-attr]
                                                    self.camera.width)  # type: ignore[union-attr]
            self.vis = VisionSession(self.camera, self.ctrl, self.events, self.ctrl.dt,  # type: ignore[arg-type]
                                     TARGET, seed=int(self.rng.integers(2**31)), renderer=self.vis_renderer)
            self.vis_j0 = self.k + int(round(POST_ARRIVAL_SKIP_S / self.ctrl.dt))
            self.vis_end = windows * int(round(WINDOW_S / self.ctrl.dt))
        self._hold(POST_ARRIVAL_SKIP_S)
        win_ev: list = []
        cam_seen: list[list[float]] = [[] for _ in range(windows)]
        aborted = False
        for w in range(windows if t_arrive is not None else 0):
            k0 = len(self.log.gyro)
            self._hold(WINDOW_S, {"cam_window": w, "cam_seen": cam_seen})
            ev = self.log.summarise(k0, baseline, self.ctrl.dt, quiet=self._quiet(k0), stop=len(self.log.gyro))
            win_ev.append(ev)
            if self.vis is not None:
                xx, yy, _ = self.ctrl.pose()
                cam_seen[w] = self.vis.window(math.hypot(TARGET[0] - xx, TARGET[1] - yy))
            if self.fell:
                break
            if abort_on_detect and ev.detected and w + 1 < windows:
                aborted = True
                break
        if self.vis is not None:
            vis_out = {"events": self.vis.window_events, "truth": self.vis.window_truth}
            self.vis.close()
            self.vis = None
        self.phase = "return"
        t_home = self._walk(HOME_TARGET, _neutral_with_proximity(HOME_PROXIMITY))
        pushes = self.pushes[n_push0:]
        return {
            "t": round(self.ctrl.t - self.t0, 2), "t_arrive": t_arrive, "t_home": t_home,
            "arrived": t_arrive is not None,
            "d_stop": d_stop, "windows": [e.as_list() for e in win_ev], "aborted": aborted,
            "imu_observable": [e.quiet_frames >= MIN_QUIET_FRAMES for e in win_ev],
            "camera": [camera_code(c) for c in cam_seen[:len(win_ev)]] if self.camera is not None else [],
            "fell": self.fell, "trips": self.monitor.trips, "vision": vis_out,
            "truth": {"type": etype, "contacts": len(pushes), "max_push": max((p[1] for p in pushes), default=0.0),
                      "lunges": len(self.lunges) - n_lunge0}}


def _neutral_with_proximity(p: float):
    from apm_express.intent import ExpressionIntent
    from apm_express.styles import NEUTRAL
    return ExpressionIntent(**{**NEUTRAL.as_dict(), "proximity": p}).sanitized()
