"""Programme L Phase C: rendered-image perception of the entity's lunges.

It replaces K's abstract event sensor (`nerva_world.Camera`) with a camera that is rendered and perception code
that reads only images.

**World side.** This is simulation, not perception.
- **The entity is visible.** It is drawn as a person-scale column (radius 0.12 m, height 1.2 m) standing at the
  target. It is a render-only scene geometry, so physics is unchanged and the impulse model of K stays exactly as
  it is.
- **Each scheduled lunge moves it physically.**
  - The column slides toward the robot until its front face is `reach` from the target. It is clamped 5 cm short
    of the robot.
  - The move peaks at the lunge's time step, i.e. the step at which the impulse model applies any push.
  - It rises and falls in `distance / speed` seconds each way (0.15–1.0 s).
- **The eye is mounted on the robot.**
  - Placement: NERVA's head, along its face direction; the wheeled robot's mast head; the T1's head.
  - It is a depth camera pitched 10° down, with a 40° horizontal field of view (the abstract camera's, A5) at
    64 × 48 px.
  - It renders at 10 Hz during sensing windows.
  - Noise: Gaussian depth noise σ (1 + depth) and random dropouts.

**Perception side (`MotionPerception`).** Its inputs are:
- depth frames;
- the robot's own-motion flags (efference copy);
- the robot's *navigation estimate* of its distance to the target. This is the true distance plus a fixed odometry
  error, N(0, 3 cm) per encounter, i.e. what a navigation stack reports.

It never reads simulator state.

Per window:
1. **Background.** A per-pixel background is built from still frames, and rebuilt after the robot's own motion.
2. **Foreground.** The foreground is the valid pixels at least `tau` closer than the background, within `range_m`.
3. **Egomotion.** A frame where many pixels also moved *away* is egomotion (e.g. a push). It is discarded, and the
   background is rebuilt.
4. **A lunge event** is a run of frames with at least `min_px` foreground pixels.
   - Its reach estimate is the navigation distance minus the closest foreground depth (10th percentile) in the run.
   - The window's code is `nerva_world.camera_code` of the estimated reaches: the same 8 categories as K.

**Truth.** Ground truth (the lunge list with in-view flags) is recorded separately for metrics. Perception never
sees it.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ENTITY_R, ENTITY_H = 0.12, 1.2
CLEARANCE = 0.05
WARM_S = 0.5  # rendering starts this long before the first sensing window, to build the background
SURFACE_BAND = 0.35  # m around the navigation estimate of the target's distance


@dataclass
class VisionCamera:
    width: int = 64
    height: int = 48
    hfov_deg: float = 40.0
    pitch_deg: float = -10.0
    hz: float = 10.0
    depth_sigma: float = 0.01
    dropout: float = 0.02
    range_m: float = 2.0
    nav_sigma: float = 0.03
    tau: float = 0.05
    min_px: int = 8
    keep_frames: bool = False  # videos and figures only
    occluded: bool = False     # programme L Phase D: a static screen partly hides the entity in this encounter

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2 * math.atan(math.tan(math.radians(self.hfov_deg) / 2) * self.height / self.width))


def robot_view(robot) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """(model, data, eye position, unit forward direction, world offset of the pose frame) for any of the three
    bodies. This is the sensor's mounting, not a perception input."""
    if hasattr(robot, "sim"):  # NERVA (apm_express.biped.NervaBiped)
        m, d = robot.sim.model, robot.sim.data
        f = d.xmat[robot._head_body].reshape(3, 3) @ robot._face_local
        f = f / np.linalg.norm(f)
        return m, d, d.xpos[robot._head_body] + 0.05 * f, f, np.asarray(robot._origin, float)
    m, d = robot.model, robot.data
    if hasattr(robot, "imu_site"):  # Booster T1 (humanoid_world.HumanoidRobot)
        r = d.xmat[m.body("Trunk").id].reshape(3, 3)
        f = r[:, 0] / np.linalg.norm(r[:, 0])
        return m, d, d.xpos[m.body("H2").id] + 0.10 * f, f, np.zeros(2)
    r = d.xmat[m.body("base").id].reshape(3, 3)  # wheeled robot: the mast head
    f = r[:, 0] / np.linalg.norm(r[:, 0])
    return m, d, d.xpos[m.body("base").id] + r @ np.array([0.11, 0.0, 0.18]), f, np.zeros(2)


class LungeAnimator:
    """World side: the entity column's displacement toward the robot at each control step."""

    def __init__(self, events: dict[int, tuple[float, float]], dt: float):
        self.events, self.dt = events, dt

    def displacement(self, k: int, dist: float, front: float = 0.0) -> float:
        """Column displacement toward the robot; its face stops CLEARANCE short of the robot's front (`front` m
        ahead of the robot's origin): a lunge that reaches the robot looks like one that reaches it, no farther."""
        s = 0.0
        for ke, (reach, speed) in self.events.items():
            peak = float(np.clip(reach - ENTITY_R, 0.0, max(0.0, dist - front - ENTITY_R - CLEARANCE)))
            n = max(1, int(round(float(np.clip(peak / max(speed, 1e-3), 0.15, 1.0)) / self.dt)))
            j = k - ke
            if -n <= j <= n:
                s = max(s, peak * (1.0 - abs(j) / n))
        return s


@dataclass
class Event:
    frames: list[int] = field(default_factory=list)
    closest: float = math.inf
    quiet: int = 0  # consecutive frames without foreground


class MotionPerception:
    """Perception side: depth frames + own-motion flags + navigation distance -> lunge events with reach estimates."""

    N_BG = 3

    def __init__(self, cam: VisionCamera):
        self.cam = cam
        f = (cam.width / 2) / math.tan(math.radians(cam.hfov_deg) / 2)
        el = math.radians(cam.pitch_deg)
        up = (cam.height / 2 - 0.5 - np.arange(cam.height)) / f  # image-up tangent per row
        # horizontal forward distance per unit z-depth, per row, for a camera pitched by el
        self.h_per_z = (math.cos(el) - math.sin(el) * up)[:, None] * np.ones((1, cam.width))
        self.reset()

    def reset(self) -> None:
        self.bg: np.ndarray | None = None
        self.stack: list[np.ndarray] = []
        self.event: Event | None = None
        self.reaches: list[float] = []
        self.events: list[dict] = []
        self.i = 0
        self.eye_offset = 0.0

    def _close(self, d_nav: float) -> None:
        if self.event is not None and math.isfinite(self.event.closest):
            r = round(d_nav - self.eye_offset - self.event.closest, 3)
            self.reaches.append(r)
            self.events.append({"frames": self.event.frames, "reach_est": r})
        self.event = None

    @staticmethod
    def _coherent(mask: np.ndarray) -> np.ndarray:
        """Keep pixels with at least 2 of their 4 neighbours also set (removes isolated noise)."""
        n = np.zeros(mask.shape, int)
        n[1:] += mask[:-1]
        n[:-1] += mask[1:]
        n[:, 1:] += mask[:, :-1]
        n[:, :-1] += mask[:, 1:]
        return mask & (n >= 2)

    def frame(self, depth: np.ndarray, own_motion: bool, d_nav: float, eye_offset: float = 0.0) -> None:
        """One depth frame. `eye_offset`: how far the eye sits ahead of the body origin toward the target (the
        robot's knowledge of its own sensor mounting)."""
        c = self.cam
        self.i += 1
        self.eye_offset = eye_offset
        valid = (depth > 0.02) & (depth < c.range_m)
        if own_motion:
            self._close(d_nav)
            self.bg, self.stack = None, []
            return
        if self.bg is None:  # build: per-pixel median of the first still frames, dropouts ignored
            self.stack.append(np.where(depth > 0.02, depth, np.nan))
            if len(self.stack) >= self.N_BG:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-dropout pixels -> 0 (invalid) below
                    bg = np.nanmedian(np.array(self.stack), axis=0)
                self.bg = np.where(np.isfinite(bg), bg, 0.0)
                self.stack = []
            return
        thr = np.maximum(c.tau, 4.0 * c.depth_sigma * (1.0 + self.bg))  # the sensor's noise specification
        ok = valid & (self.bg > 0.02)
        closer = self._coherent(ok & (self.bg - depth > thr))
        farther = self._coherent(ok & (depth - self.bg > thr) & (self.bg < c.range_m))
        n_fg = int(closer.sum())
        if n_fg >= c.min_px and farther.sum() > 0.3 * n_fg:  # egomotion (e.g. pushed): discard, rebuild
            self._close(d_nav)
            self.bg, self.stack = None, [np.where(depth > 0.02, depth, np.nan)]
            return
        # motion *of the thing at the target*: foreground whose background lies near the target distance (from the
        # robot's navigation). Silhouette edges, whose background is the floor behind, do not count.
        near_target = np.abs(self.bg * self.h_per_z - (d_nav - eye_offset)) < SURFACE_BAND
        n_fg = int((closer & near_target).sum())
        if n_fg >= c.min_px:
            if self.event is None:
                self.event = Event()
            self.event.frames.append(self.i)
            self.event.quiet = 0
            h = depth[closer] * self.h_per_z[closer]
            self.event.closest = min(self.event.closest, float(np.percentile(h, 10)))
        else:
            if self.event is not None:
                self.event.quiet += 1
                if self.event.quiet < 2:  # one-frame gap tolerance
                    return
            self._close(d_nav)
            upd = ok & ~closer
            self.bg[upd] = 0.9 * self.bg[upd] + 0.1 * depth[upd]

    def end_window(self, d_nav: float) -> list[float]:
        self._close(d_nav)
        out = self.reaches
        self.reaches = []
        return out


class VisionSession:
    """One encounter's rendered camera. The world calls `step` after every control step and `window` at each
    sensing-window boundary."""

    def __init__(self, cam: VisionCamera, robot, events: dict[int, tuple[float, float]], dt: float,
                 target: tuple[float, float], seed: int, renderer: Any = None):
        """`renderer`: a depth renderer owned by the caller and reused across encounters. MuJoCo 3.3 destroys the
        GL context shared by every renderer in the thread when one is closed, which blanks any video renderer."""
        import mujoco

        self.mj, self.cam, self.robot, self.dt, self.target = mujoco, cam, robot, dt, target
        m, _, _, _, _ = robot_view(robot)
        m.vis.global_.fovy = cam.fovy_deg
        self.owns_renderer = renderer is None
        self.renderer = renderer if renderer is not None else mujoco.Renderer(m, cam.height, cam.width)
        self.renderer.enable_depth_rendering()
        self.mjcam = mujoco.MjvCamera()
        self.mjcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.anim = LungeAnimator(events, dt)
        self.events = events
        self.rng = np.random.default_rng([seed, 11])
        self.nav_err = float(self.rng.normal(0.0, cam.nav_sigma))
        self.every = max(1, int(round(1.0 / (cam.hz * dt))))
        self.perc = MotionPerception(cam)
        self.frames: list[np.ndarray] = []
        self.window_truth: list[list[dict]] = []
        self.window_events: list[list[dict]] = []
        self._truth: list[dict] = []
        self.frame_k: list[int] = []  # control step of each perception frame (for metrics only)
        self.last_depth: np.ndarray | None = None

    def _entity(self, k: int, dist: float, xy: tuple[float, float], origin: np.ndarray, front: float) -> np.ndarray:
        tx, ty = origin[0] + self.target[0], origin[1] + self.target[1]
        ux, uy = xy[0] + origin[0] - tx, xy[1] + origin[1] - ty
        n = math.hypot(ux, uy) or 1.0
        s = self.anim.displacement(k, dist, front)
        return np.array([tx + s * ux / n, ty + s * uy / n, ENTITY_H / 2])

    def eye_offset(self) -> float:
        """How far the eye sits ahead of the robot's origin toward the target (its sensor mounting)."""
        _, _, eye, _, origin = robot_view(self.robot)
        x, y, _ = self.robot.pose()
        to_t = np.array([self.target[0] - x, self.target[1] - y])
        to_t = to_t / (np.linalg.norm(to_t) or 1.0)
        return float(np.dot(eye[:2] - (origin + np.array([x, y])), to_t))

    def render(self, k: int, dist: float) -> np.ndarray:
        m, d, eye, fwd, origin = robot_view(self.robot)
        x, y, _ = self.robot.pose()
        centre = self._entity(k, dist, (x, y), origin, self.eye_offset())
        az = math.degrees(math.atan2(fwd[1], fwd[0]))
        el = math.degrees(math.asin(float(np.clip(fwd[2], -1, 1)))) + self.cam.pitch_deg
        look = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                         math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
        self.mjcam.lookat[:] = eye + look
        self.mjcam.distance, self.mjcam.azimuth, self.mjcam.elevation = 1.0, az, el
        r = self.renderer
        r.update_scene(d, camera=self.mjcam)
        scn = r.scene
        if scn.ngeom < scn.maxgeom:
            self.mj.mjv_initGeom(scn.geoms[scn.ngeom], int(self.mj.mjtGeom.mjGEOM_CYLINDER),
                                 np.array([ENTITY_R, ENTITY_H / 2, 0.0]), centre, np.eye(3).flatten(),
                                 np.array([0.85, 0.35, 0.2, 1.0], np.float32))
            scn.ngeom += 1
        if self.cam.occluded and scn.ngeom < scn.maxgeom:  # screen halfway, covering ~70% of the column's width
            tx, ty = origin[0] + self.target[0], origin[1] + self.target[1]
            v = np.array([tx - eye[0], ty - eye[1]])
            v = v / (np.linalg.norm(v) or 1.0)
            side = np.array([-v[1], v[0]])
            mid = np.array([eye[0], eye[1]]) + 0.5 * (np.array([tx, ty]) - eye[:2]) + 0.05 * side
            rot = np.array([[v[0], -v[1], 0.0], [v[1], v[0], 0.0], [0.0, 0.0, 1.0]])
            self.mj.mjv_initGeom(scn.geoms[scn.ngeom], int(self.mj.mjtGeom.mjGEOM_BOX),
                                 np.array([0.01, 0.10, ENTITY_H / 2]), np.array([mid[0], mid[1], ENTITY_H / 2]),
                                 rot.flatten(), np.array([0.3, 0.3, 0.3, 1.0], np.float32))
            scn.ngeom += 1
        depth = np.asarray(r.render(), float).copy()
        depth += self.rng.normal(0.0, 1.0, depth.shape) * self.cam.depth_sigma * (1.0 + depth)
        depth[self.rng.random(depth.shape) < self.cam.dropout] = 0.0
        self.last_depth = depth  # videos only
        return depth

    def in_view(self, dist: float) -> bool:
        """Truth only: is the entity's centre inside the horizontal field of view and range?"""
        _, _, eye, fwd, origin = robot_view(self.robot)
        tx, ty = origin[0] + self.target[0], origin[1] + self.target[1]
        b = math.atan2(ty - eye[1], tx - eye[0]) - math.atan2(fwd[1], fwd[0])
        b = math.atan2(math.sin(b), math.cos(b))
        return abs(b) <= math.radians(self.cam.hfov_deg) / 2 and dist <= self.cam.range_m

    def step(self, k: int, j_in_windows: int, dist: float, own_motion: bool) -> None:
        """Call after the control step k. j < 0: warm-up before the first window (background only; anything seen
        then is discarded at j = 0)."""
        if j_in_windows == 0:
            self.perc.end_window(dist + self.nav_err)
            self.perc.events = []
        if k in self.events and j_in_windows >= 0:
            reach, speed = self.events[k]
            front = self.eye_offset()
            seen = ENTITY_R + float(np.clip(reach - ENTITY_R, 0.0, max(0.0, dist - front - ENTITY_R - CLEARANCE)))
            self._truth.append({"k": k, "reach": round(reach, 3), "visible_reach": round(seen, 3),
                                "moves": bool(reach > ENTITY_R), "in_view": self.in_view(dist),
                                "dist": round(dist, 3)})
        if j_in_windows % self.every == 0:
            depth = self.render(k, dist)
            self.frame_k.append(k)
            if self.cam.keep_frames:
                self.frames.append(depth.astype(np.float16))
            self.perc.frame(depth, own_motion, dist + self.nav_err, self.eye_offset())

    def window(self, dist: float) -> list[float]:
        reaches = self.perc.end_window(dist + self.nav_err)
        self.window_events.append([{"k0": self.frame_k[e["frames"][0] - 1], "k1": self.frame_k[e["frames"][-1] - 1],
                                    "reach_est": e["reach_est"]} for e in self.perc.events])
        self.perc.events = []
        self.window_truth.append(self._truth)
        self._truth = []
        return reaches

    def close(self) -> None:
        if self.owns_renderer:
            self.renderer.close()
