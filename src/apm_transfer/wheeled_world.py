"""Programme K5: a second, physically different embodiment. A differential-drive wheeled robot in MuJoCo.

**The robot.**
- A 2.9 kg rigid chassis on two driven wheels (radius 0.04 m, track 0.22 m) and two frictionless casters.
- Velocity-controlled wheel motors; a chassis IMU (accelerometer, gyro).
- Full rigid-body contact physics. Pushes are real impulses on the chassis, and the robot's own wheels, friction and
  inertia decide how it responds.
- It cannot fall over the way NERVA can, but it can be shoved, slip, spin and tip.

**Its own layers** (nothing here is shared with NERVA's body):
- *Motion controller:* wheel-speed tracking of (v, ω) requests, with speed and acceleration limits.
- *Expression adapter:* H7's `apm_express.wheeled.WheeledAdapter`, mapping the same `ExpressionIntent` to (v, ω):
  stopping distance from proximity, cruise speed from tempo, pauses and retreats from hesitation.
- *Safety supervisor*, independent of APM: if the chassis is shoved above 0.35 m/s or tilts beyond 25°, it brakes to
  standstill for 0.5 s and counts a trip. Beyond 60° tilt it has tipped over (a fall).
- *Observation model:* horizontal acceleration deviation during quiet standing (efference-copy gate,
  self-calibrated detector), coded with the robot's **own** acc8 edges from its own exploration; plus the same
  declared `Camera` extension as NERVA's.

The entity model (`nerva_world.TYPES`, `impulse`) and the encounter protocol (settle, approach, 1 s skip, 4 s
windows, abort on detection) are identical to NERVA's, so the APM core is exercised unchanged.
`closed_loop_encounter` returns the same structures as `nerva_world.closed_loop_encounter`.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from apm_transfer.nerva_world import (
    ACC_NOISE,
    GYRO_NOISE,
    MAX_WINDOWS,
    POST_ARRIVAL_SKIP_S,
    SELF_MOTION_HOLD_S,
    TARGET,
    WINDOW_S,
    Camera,
    Evidence,
    camera_code,
    impulse,
    lunges,
)

R_WHEEL, TRACK = 0.04, 0.22
DT, SUBSTEPS = 0.02, 10
SHOVE_TRIP, TILT_TRIP, TILT_FALL, BRAKE_S = 0.35, math.radians(25), math.radians(60), 0.5

MJCF = """
<mujoco model="apm_wheeled">
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba=".82 .82 .82 1" friction="1 0.005 0.0001"/>
    <body name="base" pos="0 0 0.062">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="0.12 0.09 0.03" mass="2.5" rgba=".25 .45 .8 1"/>
      <geom name="mast" type="cylinder" size="0.015 0.07" pos="0.06 0 0.1" mass="0.08" rgba=".2 .2 .2 1"/>
      <geom name="head" type="box" size="0.03 0.04 0.025" pos="0.07 0 0.18" mass="0.05" rgba=".9 .9 .9 1"/>
      <geom name="caster_f" type="sphere" size="0.02" pos="0.095 0 -0.042" friction="0 0 0"
            condim="1" priority="1" mass="0.03"/>
      <geom name="caster_b" type="sphere" size="0.02" pos="-0.095 0 -0.042" friction="0 0 0"
            condim="1" priority="1" mass="0.03"/>
      <site name="imu" pos="0 0 0"/>
      <body name="wheel_l" pos="0 0.11 -0.022">
        <joint name="jl" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom type="cylinder" size="0.04 0.012" euler="90 0 0" mass="0.15" friction="1.2 0.01 0.0001"
              rgba=".1 .1 .1 1"/>
      </body>
      <body name="wheel_r" pos="0 -0.11 -0.022">
        <joint name="jr" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom type="cylinder" size="0.04 0.012" euler="90 0 0" mass="0.15" friction="1.2 0.01 0.0001"
              rgba=".1 .1 .1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="ml" joint="jl" kv="0.8" ctrlrange="-10 10" forcerange="-1.5 1.5"/>
    <velocity name="mr" joint="jr" kv="0.8" ctrlrange="-10 10" forcerange="-1.5 1.5"/>
  </actuator>
  <sensor>
    <accelerometer name="acc" site="imu"/>
    <gyro name="gyro" site="imu"/>
  </sensor>
</mujoco>
"""


@lru_cache(maxsize=1)
def _model_xml() -> str:
    return MJCF


class WheeledRobot:
    """Physics wheeled robot with a (v, ω) motion controller and an independent safety supervisor."""

    def __init__(self, seed: int, scene_extender=None):
        import mujoco

        spec = mujoco.MjSpec.from_string(_model_xml())
        if scene_extender is not None:
            scene_extender(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.mj = mujoco
        mujoco.mj_forward(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.dt = DT
        self.v_cmd = self.w_cmd = 0.0
        self.brake_until = -1.0
        self.trips = 0
        self._origin = np.zeros(2)
        self.acc_id = self.model.sensor("acc").adr[0]
        self.gyro_id = self.model.sensor("gyro").adr[0]

    def pose(self) -> tuple[float, float, float]:
        q = self.data.qpos
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), yaw

    def tilt(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        up_z = 1 - 2 * (x * x + y * y)
        return math.acos(float(np.clip(up_z, -1.0, 1.0)))

    def push(self, dvx: float, dvy: float) -> None:
        self.data.qvel[0] += dvx
        self.data.qvel[1] += dvy

    def apply(self, v: float, w: float) -> bool:
        """Request (v, ω); the supervisor may override. Returns True if the supervisor tripped this step."""
        speed = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
        tripped = False
        if speed > SHOVE_TRIP or self.tilt() > TILT_TRIP:
            if self.t >= self.brake_until:
                self.trips += 1
            self.brake_until = self.t + BRAKE_S
            tripped = True
        if self.t < self.brake_until:
            v, w = 0.0, 0.0
        # acceleration-limited tracking (the motion controller)
        self.v_cmd += float(np.clip(v - self.v_cmd, -0.4 * DT, 0.4 * DT))
        self.w_cmd += float(np.clip(w - self.w_cmd, -3.0 * DT, 3.0 * DT))
        wl = (self.v_cmd - self.w_cmd * TRACK / 2) / R_WHEEL
        wr = (self.v_cmd + self.w_cmd * TRACK / 2) / R_WHEEL
        self.data.ctrl[:] = [wl, wr]
        return tripped

    def fallen(self, row: dict) -> bool:
        return math.radians(row["tilt"]) > TILT_FALL or not row["finite"]

    def advance(self) -> dict:
        for _ in range(SUBSTEPS):
            self.mj.mj_step(self.model, self.data)
        self.t += DT
        acc = np.array(self.data.sensordata[self.acc_id:self.acc_id + 3], float)
        gyro = np.array(self.data.sensordata[self.gyro_id:self.gyro_id + 3], float)
        return {"acc": acc, "gyro": gyro, "tilt": math.degrees(self.tilt()),
                "finite": bool(np.all(np.isfinite(self.data.qpos)))}


class WheeledLogger:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.gyro: list[float] = []
        self.acc: list[np.ndarray] = []
        self.tilt: list[float] = []

    def read(self, row: dict) -> None:
        g = row["gyro"] + self.rng.uniform(-GYRO_NOISE, GYRO_NOISE, 3)
        a = row["acc"] + self.rng.uniform(-ACC_NOISE, ACC_NOISE, 3)
        self.gyro.append(float(np.linalg.norm(g)))
        self.acc.append(a[:2].copy())
        self.tilt.append(float(row["tilt"]))

    def summarise(self, start: int, baseline: slice, quiet: np.ndarray, stop: int) -> Evidence:
        acc = np.array(self.acc)
        base_acc = acc[baseline].mean(0)
        a = np.linalg.norm(acc[start:stop] - base_acc, axis=1)
        g = np.array(self.gyro[start:stop])
        mask = np.asarray(quiet[start:stop], bool)
        a, g = a[mask], g[mask]
        base_peak = float(np.max(np.linalg.norm(acc[baseline] - base_acc, axis=1)))
        peak = float(np.max(a)) if len(a) else 0.0
        energy = float(np.sum(np.clip(a - base_peak, 0.0, None) ** 2)) if len(a) else 0.0
        tilt = np.array(self.tilt)
        return Evidence(float(g.max()) if len(g) else 0.0, peak, 0.0, 0, 1.0,
                        float(tilt[start:stop].max() - tilt[baseline].mean()), bool(tilt[start:stop].max() > 60),
                        base_peak, energy, bool(peak > 1.5 * base_peak), int(mask.sum()))


def closed_loop_encounter(seed: int, etype: str, intent, channels: dict | None = None, settle_s: float = 2.0,
                          max_approach_s: float = 45.0, windows: int = 1, camera: Camera | None = None,
                          abort_on_detect: bool = True, scene_extender=None,
                          frame_hook=None, standoff: float = 0.0, robot_factory=None) -> tuple[Evidence, dict, dict]:
    """Same protocol and return structure as `nerva_world.closed_loop_encounter` with `windows` set.

    `robot_factory(seed, scene_extender)` may supply another (v, ω)-commanded body with the same interface
    (`humanoid_world.HumanoidRobot`); the default is this wheeled robot."""
    from apm_express.wheeled import WheeledAdapter

    rng = np.random.default_rng(seed)
    bot = (robot_factory or WheeledRobot)(int(rng.integers(2**31)), scene_extender)
    adapter = WheeledAdapter(seed=int(rng.integers(2**31)))
    log = WheeledLogger(np.random.default_rng(int(rng.integers(2**31))))
    cam_rng = np.random.default_rng([seed, 7])
    n_settle = int(round(settle_s / DT))
    for _ in range(n_settle):
        bot.apply(0.0, 0.0)
        log.read(bot.advance())
    n_settle = len(log.acc)
    horizon = max_approach_s + POST_ARRIVAL_SKIP_S + WINDOW_S * MAX_WINDOWS
    events = {n_settle + int(round(t / DT)): (reach, speed) for t, reach, speed in lunges(rng, etype, horizon)}
    aim = (TARGET[0] - standoff, TARGET[1])
    adapter.reset(bot.pose(), aim, intent)
    win_n, skip_n = int(round(WINDOW_S / DT)), int(round(POST_ARRIVAL_SKIP_S / DT))
    hold_n = int(round(SELF_MOTION_HOLD_S / DT))
    own = [False] * n_settle
    pushes: list[tuple[float, float, bool]] = []
    cam_seen: list[list[float]] = [[] for _ in range(windows)]
    win_ev: list[Evidence] = []
    k, k_arrive, t_arrive, fell, aborted = n_settle, None, None, False, False
    baseline = slice(max(0, n_settle - 50), n_settle)
    while True:
        x, y, yaw = bot.pose()
        dist = math.hypot(TARGET[0] - x, TARGET[1] - y)
        lunge = None
        if k in events:
            reach, speed = events[k]
            m = impulse(reach, speed, dist)
            lunge = (reach, m)
            if m > 0:
                away = math.atan2(y - TARGET[1], x - TARGET[0]) + float(rng.uniform(-0.35, 0.35))
                bot.push(m * math.cos(away), m * math.sin(away))
                pushes.append((round(bot.t, 2), round(m, 4), k_arrive is not None))
            if camera is not None and k_arrive is not None and k >= k_arrive + skip_n:
                w = (k - k_arrive - skip_n) // win_n
                bearing = math.atan2(math.sin(math.atan2(TARGET[1] - y, TARGET[0] - x) - yaw),
                                     math.cos(math.atan2(TARGET[1] - y, TARGET[0] - x) - yaw))
                if w < windows and abs(bearing) <= camera.fov and dist <= camera.range_m \
                        and cam_rng.random() < camera.p_detect:
                    cam_seen[w].append(round(reach + float(cam_rng.normal(0, camera.sigma)), 3))
        if k_arrive is None:
            v, w_ = adapter.request(bot.t, bot.pose(), aim, intent)
        else:
            v, w_ = 0.0, 0.0  # sensing pause: hold still
        tripped = bot.apply(v, w_)
        own.append(bool(tripped or abs(v) > 1e-6 or abs(w_) > 1e-6 or abs(bot.v_cmd) > 1e-3
                        or abs(bot.w_cmd) > 1e-3))
        row = bot.advance()
        log.read(row)
        if frame_hook is not None:
            frame_hook(bot, {"lunge": lunge, "dist": dist, "arrived": k_arrive is not None, "fallback": tripped})
        if bot.fallen(row):
            fell = True
            break
        if k_arrive is None and adapter.arrived:
            k_arrive, t_arrive = k + 1, round(bot.t - settle_s, 2)
        k += 1
        if k_arrive is not None and k > k_arrive + skip_n and (k - k_arrive - skip_n) % win_n == 0:
            w0 = k_arrive + skip_n + len(win_ev) * win_n
            mv = np.asarray(own, bool)
            q = np.ones(len(mv), bool)
            q[w0:] = ~np.array([mv[max(0, j - hold_n):j + 1].any() for j in range(w0, len(mv))], bool)
            win_ev.append(log.summarise(w0, baseline, q, k))
            if abort_on_detect and win_ev[-1].detected and len(win_ev) < windows:
                aborted = True
                break
            if len(win_ev) >= windows:
                break
        if k - n_settle >= int(round(horizon / DT)):
            break
    x, y, _ = bot.pose()
    ev = win_ev[0] if win_ev else Evidence(0, 0, 0, 0, 1.0, 0, fell, 0, 0, False, 0)
    outcome: dict = {"arrived": k_arrive is not None, "t_arrive": t_arrive,
                     "final_distance": round(math.hypot(TARGET[0] - x, TARGET[1] - y), 3), "trips": bot.trips,
                     "fallback_steps": 0, "fell": fell, "windows": [e.as_list() for e in win_ev], "aborted": aborted,
                     "dwell_s": round((k - k_arrive) * DT, 2) if k_arrive is not None else 0.0}
    if camera is not None:
        outcome["camera"] = [camera_code(c) for c in cam_seen[:len(win_ev)]]
        outcome["camera_reaches"] = cam_seen[:len(win_ev)]
    truth = {"type": etype, "contacts": len(pushes), "contacts_dwell": sum(p[2] for p in pushes),
             "max_push": max((p[1] for p in pushes), default=0.0), "pushes": pushes}
    return ev, outcome, truth
