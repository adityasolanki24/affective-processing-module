"""APM v1.0: encounters in realistic Isaac Sim scenes (needs the Isaac Sim 5.1 environment).

`IsaacWorld` holds one scene, one robot, its eye (a depth camera) and the people. `encounter(...)` runs the
validated encounter protocol of programmes K and L (`wheeled_world.closed_loop_encounter`), with three changes:
- **The entity is a physical person** (`agents.Person`). Lunges move a kinematic collider, and contacts are
  PhysX contacts.
- **The eye is an Isaac RTX depth camera** (distance to the image plane), rendered at 10 Hz during the sensing
  windows. Its frames go to the same `vision.MotionPerception`, which sees only pixels, the robot's own-motion
  flags, the navigation distance and its own camera specification.
- **The navigation distance** is the robot's odometry pose to the station's map position, plus a fixed
  per-encounter error.

What the robot senses becomes the same `outcome` dictionary as before (IMU windows and camera codes), so
`apm_transfer.runtime` body adapters and analyses work unchanged. `truth` (contacts, lunges, impulses) is for
evaluation only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apm_isaac.agents import Person, add_person, place
from apm_isaac.robots import DT

REPO = Path(__file__).resolve().parents[2]


class _WheeledDriver:
    """(v, w) bodies (Nova Carter, T1, the reference robot): `apm_express.wheeled.WheeledAdapter`, as in K/L."""

    idle = (0.0, 0.0)

    def __init__(self, robot, seed: int):
        from apm_express.wheeled import WheeledAdapter
        self.robot, self.ad = robot, WheeledAdapter(seed=seed)

    def reset(self, t: float, pose, aim, intent) -> None:
        self.ad.reset(pose, aim, intent)

    def request(self, t: float, pose, aim, intent):
        return self.ad.request(t, pose, aim, intent)

    @property
    def arrived(self) -> bool:
        return bool(self.ad.arrived)

    def send(self, cmd) -> bool:
        return bool(self.robot.apply(*cmd))

    def own_motion(self, cmd, tripped: bool) -> bool:
        v, w = cmd
        return bool(tripped or abs(v) > 1e-6 or abs(w) > 1e-6 or abs(self.robot.v_cmd) > 1e-3
                    or abs(self.robot.w_cmd) > 1e-3)


class _BipedDriver:
    """NERVA: H's `BipedExpressionAdapter` with the admitted channels, NERVA's SafetyMonitor inside the body, the
    adapter told of every fallback, exactly as in `nerva_world.closed_loop_encounter`."""

    def __init__(self, robot, seed: int, channels: dict):
        from apm_express.adapter import NEUTRAL_REQUEST, BipedChannels, BipedExpressionAdapter
        ch = BipedChannels(min_dwell_s=channels["min_dwell_s"], pitch=channels["pitch"], head=channels["head"],
                           head_pitch_range=tuple(channels["head_pitch_range"]),
                           head_yaw_range=tuple(channels["head_yaw_range"]), head_rate=channels["head_rate"],
                           head_while_moving=False)
        self.robot, self.ad, self.idle = robot, BipedExpressionAdapter(ch, seed=seed), NEUTRAL_REQUEST
        self.prev_head: np.ndarray = np.zeros(4)

    def reset(self, t: float, pose, aim, intent) -> None:
        self.ad.reset(t, pose, aim, intent)

    def request(self, t: float, pose, aim, intent):
        return self.ad.request(t, pose, aim, intent)

    @property
    def arrived(self) -> bool:
        return bool(self.ad._arrived)

    def send(self, cmd) -> bool:
        tripped = bool(self.robot.drive(cmd))
        if tripped:
            self.ad.notify_fallback(self.robot.t)
        return tripped

    def own_motion(self, cmd, tripped: bool) -> bool:
        head = np.asarray(cmd.head, float)
        moved = bool(tripped or cmd.command != "rest" or np.abs(head - self.prev_head).max() > 1e-3)
        self.prev_head = head
        return moved


@dataclass
class SceneSpec:
    name: str
    usd: str              # relative to the Isaac assets root, or an absolute path
    home: tuple[float, float, float]  # robot start x, y, yaw
    stations: dict[str, tuple[float, float]]  # where people stand (map positions)
    floor_z: float = 0.0  # the asset's floor height in metres after `scale`; the scene is shifted so the floor is z=0
    scale: float = 1.0    # 0.01 for centimetre assets (not rescaled automatically when referenced)
    held_out: bool = False


class IsaacWorld:
    def __init__(self, scene: SceneSpec, body: str, seed: int, vision_cfg: Any = None, video: bool = False,
                 characters: dict[str, str] | None = None, physics_dt: float = 0.002,
                 policy_dir: str | Path | None = None, nerva_channels: dict | None = None):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        from apm_transfer.vision import VisionCamera
        self.scene, self.body_name, self.seed = scene, body, seed
        self.rng = np.random.default_rng(seed)
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=DT)
        self.root = get_assets_root_path()
        usd = scene.usd if "://" in scene.usd or scene.usd.startswith("/work") else self.root + scene.usd
        add_reference_to_stage(usd, "/World/Scene")
        from pxr import Gf, UsdGeom
        xf = UsdGeom.XformCommonAPI(self.world.stage.GetPrimAtPath("/World/Scene"))
        xf.SetTranslate(Gf.Vec3d(0.0, 0.0, -scene.floor_z))
        if scene.scale != 1.0:
            xf.SetScale(Gf.Vec3f(scene.scale, scene.scale, scene.scale))
        self.world.scene.add_default_ground_plane(z_position=-0.001)
        hx, hy, hyaw = scene.home
        self.people: dict[str, Person] = {}
        from apm_isaac.agents import setup_animation
        graph = setup_animation(self.world.stage, self.root) if characters else None
        for sid, xy in scene.stations.items():
            ch = (characters or {}).get(sid)
            self.people[sid] = add_person(self.world.stage, sid, np.array(xy), ch, graph)
        if body == "nova_carter":  # the wheeled embodiment: NVIDIA Nova Carter (stock asset)
            from apm_isaac.robots import NovaCarterIsaac
            self.robot: Any = NovaCarterIsaac(self.world, "/World/Robot", (hx, hy), hyaw, seed, physics_dt)
        elif body in ("wheeled", "wheeled_reference"):  # reference only: the custom robot matched to MuJoCo
            from apm_isaac.robots import WheeledIsaac
            self.robot = WheeledIsaac(self.world, "/World/Robot", (hx, hy), hyaw, seed, physics_dt)
        elif body == "nerva":  # NERVA on its frozen neutral controller (M3 import fixes: apm_isaac.nerva)
            import os

            from apm_isaac.nerva import NervaIsaac
            self.robot = NervaIsaac(self.world, "/World/Robot", (hx, hy), hyaw, seed,
                                    Path(os.environ.get("NERVA_ROOT", "/work/NERVA")),
                                    Path(os.environ.get("ODP_ROOT", "/work/odp")),
                                    Path(os.environ.get("APM_CACHE", "/work/apm_cache")) / "nerva_usd",
                                    physics_dt, head_channel=bool((nerva_channels or {}).get("head", False)))
        elif body == "humanoid":  # Booster T1 on its Isaac Lab controller (M3: the MuJoCo controller does not transfer)
            from apm_isaac.robots import LabPolicyBody
            if policy_dir is None:
                raise ValueError("the humanoid needs policy_dir (an Isaac Lab export: policy.pt + env_cfg.json)")
            self.robot = LabPolicyBody.t1(self.world, "/World/Robot", (hx, hy), hyaw, seed, Path(policy_dir),
                                          physics_dt)
        elif body == "humanoid_mujoco_policy":  # reference only: the MuJoCo-trained T1 policy (falls in PhysX)
            from apm_isaac.robots import T1Isaac
            usd_t1 = self.root + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1.usd"
            self.robot = T1Isaac(self.world, "/World/Robot", (hx, hy), hyaw, seed, usd_t1, physics_dt)
        else:
            raise NotImplementedError(body)
        self.world.reset()
        if hasattr(self.robot, "post_reset"):
            self.robot.post_reset()
        self.vision_cfg = vision_cfg or VisionCamera()
        self.nerva_channels = nerva_channels
        self._eye_cam = self._make_eye()
        self.video_frames: list[np.ndarray] | None = [] if video else None
        self._video_target: tuple[float, float] = tuple(next(iter(scene.stations.values())))  # type: ignore[assignment]
        self._video_eye: np.ndarray | None = None
        self._video_cam = self._make_video_cam() if video else None
        self.contacts: list[tuple[float, str, float]] = []
        self._subscribe_contacts()
        self.t = 0.0
        self._home = (hx, hy, hyaw)
        _ = add_reference_to_stage  # (used above)

    def new_lifetime(self, seed: int) -> None:
        """Start a fresh lifetime in the same loaded scene: physics reset, the robot home and its controller state
        reset, a new random stream. Reloading the scene is not needed between lifetimes."""
        self.world.reset()
        self.seed, self.rng = seed, np.random.default_rng(seed)
        if hasattr(self.robot, "_seed"):
            self.robot._seed = seed
        self.robot._init_common(seed)
        if hasattr(self.robot, "monitor"):
            from apm_express.safety import SafetyMonitor
            self.robot.monitor = SafetyMonitor(self.robot.monitor.limits)
        if hasattr(self.robot, "post_reset"):
            self.robot.post_reset()
        for p in self.people.values():
            p.lunges, p.walk, p.disp = [], None, 0.0
        self.contacts.clear()
        self.t = 0.0
        if self.video_frames is not None:
            self.video_frames.clear()

    # ── sensors ──────────────────────────────────────────────────────────────
    def _make_eye(self):
        """The depth eye: a top-level camera posed before every frame from PhysX (`_pose_eye`). A camera parented
        to a robot link does not follow it: PhysX does not write link poses back to USD (M4 eye check)."""
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom, UsdPhysics
        c = self.vision_cfg
        cam = Camera(prim_path="/World/EyeCam", resolution=(c.width, c.height))
        cam.initialize()
        f = 1.0
        cam.set_focal_length(f)
        cam.set_horizontal_aperture(2 * f * math.tan(math.radians(c.hfov_deg) / 2))
        cam.set_clipping_range(float(getattr(self.robot, "eye_near", 0.02)), 30.0)
        cam.add_distance_to_image_plane_to_frame()
        # the eye's fixed mounting on its link, from the USD rest pose
        stage = self.world.stage
        eye = stage.GetPrimAtPath(self.robot.eye_prim)
        link = eye.GetParent()
        while link.IsValid() and not link.HasAPI(UsdPhysics.RigidBodyAPI):
            link = link.GetParent()
        names = list(self.robot.art._articulation_view._metadata.link_names)
        self._eye_link = names.index(link.GetName())
        cache = UsdGeom.XformCache()
        m = cache.GetLocalToWorldTransform(eye) * cache.GetLocalToWorldTransform(link).GetInverse()
        self._eye_local = np.array(m, float).T  # USD row-vector convention -> column convention
        return cam

    def eye_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation (x = view direction) of the depth eye, from the PhysX link pose.

        As programme L's camera (`vision.VisionSession`): roll-free, with the azimuth and elevation of the mounting's
        forward axis, plus the camera's own pitch (`VisionCamera.pitch_deg`)."""
        from apm_isaac.robots import quat_to_rot
        tf = np.asarray(self.robot.art._articulation_view._physics_view.get_link_transforms())[0][self._eye_link]
        M = np.eye(4)
        M[:3, :3] = quat_to_rot(np.array([tf[6], tf[3], tf[4], tf[5]]))
        M[:3, 3] = tf[:3]
        E = M @ self._eye_local
        f = E[:3, 0] / np.linalg.norm(E[:3, 0])
        az = math.atan2(f[1], f[0])
        el = math.asin(float(np.clip(f[2], -1.0, 1.0))) + math.radians(self.vision_cfg.pitch_deg)
        Rz = np.array([[math.cos(az), -math.sin(az), 0.0], [math.sin(az), math.cos(az), 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[math.cos(-el), 0.0, math.sin(-el)], [0.0, 1.0, 0.0], [-math.sin(-el), 0.0, math.cos(-el)]])
        return E[:3, 3], Rz @ Ry

    def _pose_eye(self) -> None:
        from apm_isaac.robots import rot_to_quat
        p, R = self.eye_pose()
        self._eye_cam.set_world_pose(position=p, orientation=rot_to_quat(R), camera_axes="world")

    def _make_video_cam(self):
        from isaacsim.sensors.camera import Camera
        cam = Camera(prim_path="/World/VideoCam", resolution=(1280, 720))
        cam.initialize()
        cam.set_focal_length(1.8)
        return cam

    def depth(self) -> np.ndarray:
        self._pose_eye()
        self.world.render()
        d = self._eye_cam.get_depth()
        if d is None or not np.size(d):
            return np.zeros((self.vision_cfg.height, self.vision_cfg.width))
        d = np.asarray(d, float).copy()
        d[~np.isfinite(d)] = 0.0
        c = self.vision_cfg  # the camera's own noise specification
        d += self.rng.normal(0.0, 1.0, d.shape) * c.depth_sigma * (1.0 + d)
        d[self.rng.random(d.shape) < c.dropout] = 0.0
        return d

    def _subscribe_contacts(self) -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, PhysxSchema
        for p in self.people.values():
            prim = self.world.stage.GetPrimAtPath(p.cap_path)  # type: ignore[attr-defined]
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr(0.0)

        def on_contact(headers, data):
            for h in headers:
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                if "/World/People/" in a0 + a1 and "/World/Robot" in a0 + a1:
                    imp = sum(np.linalg.norm(np.asarray(data[h.contact_data_offset + i].impulse))
                              for i in range(h.num_contact_data))
                    who = (a0 if "/World/People/" in a0 else a1).split("/")[3]
                    self.contacts.append((round(self.t, 3), who, float(imp)))
        self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

    def _video(self, info: str = "") -> None:
        if self._video_cam is None or self.video_frames is None:
            return
        # fixed side-on view of the interaction axis (home -> station), wide enough for robot and person
        x, y, _ = self.robot.pose()
        sx, sy = self._video_target
        mid = np.array([(x + sx) / 2, (y + sy) / 2, 0.55])
        if self._video_eye is None:
            self._video_eye = self._clear_view(mid, np.array([sx - x, sy - y]))
        eye = self._video_eye
        self._video_cam.set_world_pose(position=eye, orientation=_look_at(eye, mid))
        self.world.render()
        f = self._video_cam.get_rgba()
        if f is not None and np.size(f):
            self.video_frames.append(np.asarray(f[..., :3], np.uint8))

    def _clear_view(self, mid: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """A camera position looking at `mid` from the side of the interaction axis, with no collider in between
        (multi-hit PhysX raycasts; the scene's walls and shelving block views)."""
        import omni.physx
        q = omni.physx.get_physx_scene_query_interface()
        u = axis / (np.linalg.norm(axis) or 1.0)
        side = np.array([-u[1], u[0]])
        best, best_d = None, -1.0
        for dist in (3.2, 2.6, 2.0, 1.6):
            for sgn in (1.0, -1.0):
                for back in (-0.8, 0.0, 0.8):
                    eye = np.array([mid[0] + sgn * dist * side[0] + back * u[0],
                                    mid[1] + sgn * dist * side[1] + back * u[1], 1.6])
                    d = eye - mid
                    n = float(np.linalg.norm(d))
                    hit = q.raycast_closest(tuple(float(v) for v in mid), tuple(float(v) for v in d / n), n)
                    free = n if not hit["hit"] or "/World/People" in str(hit.get("rigidBody", "")) \
                        or "/World/Robot" in str(hit.get("rigidBody", "")) else float(hit["distance"])
                    if free >= n - 1e-3:
                        return eye
                    if free > best_d:
                        best, best_d = eye, free
        return best if best is not None else mid + np.array([0.0, 0.0, 1.0])

    # ── one control step with the people ─────────────────────────────────────
    def _driver(self, seed: int):
        if self.body_name == "nerva":
            if self.nerva_channels is None:
                raise ValueError("NERVA needs its admitted channels (H_admitted_interface.json 'interface')")
            return _BipedDriver(self.robot, seed, self.nerva_channels)
        return _WheeledDriver(self.robot, seed)

    def step(self, cmd, drv, render_video: bool = False) -> tuple[dict, bool]:
        x, y, _ = self.robot.pose()
        rxy = np.array([x, y])
        for p in self.people.values():
            pos, yaw = p.position(self.t, rxy)
            place(self.world.stage, p, pos, yaw)
        tripped = drv.send(cmd)
        row = self.robot.advance()
        self.t += DT
        if render_video and int(round(self.t / DT)) % 2 == 0:
            self._video()
        return row, tripped

    def encounter(self, station: str, etype: str, intent, windows: int, standoff: float = 0.0,
                  settle_s: float = 1.0, max_approach_s: float = 45.0, abort_on_detect: bool = True,
                  video: bool = False, approach_from: float | None = None,
                  return_home: bool = True) -> tuple[dict, dict]:
        """One encounter from the robot's current pose; afterwards it walks home (a continuous lifetime).

        Experiments use `approach_from=A, return_home=False`: the robot starts at rest A metres from the station
        on the home-station line, facing it, and does not walk back. Encounters then take the same time
        regardless of where the last one ended."""
        from apm_express.intent import ExpressionIntent
        from apm_express.styles import NEUTRAL
        from apm_transfer.nerva_world import (
            MAX_WINDOWS,
            MIN_QUIET_FRAMES,
            POST_ARRIVAL_SKIP_S,
            SELF_MOTION_HOLD_S,
            WINDOW_S,
            camera_code,
        )
        from apm_transfer.vision import MotionPerception
        from apm_transfer.wheeled_world import WheeledLogger
        person = self.people[station]
        sx, sy = person.xy
        self._video_target, self._video_eye = (sx, sy), None
        if approach_from is not None:
            hx0, hy0, _ = self._home
            dv = np.array([hx0 - sx, hy0 - sy])
            dv = dv / (np.linalg.norm(dv) or 1.0)
            self.robot.teleport(sx + approach_from * dv[0], sy + approach_from * dv[1], math.atan2(-dv[1], -dv[0]))
        horizon = max_approach_s + POST_ARRIVAL_SKIP_S + WINDOW_S * MAX_WINDOWS + 20.0
        erng = np.random.default_rng(int(self.rng.integers(2**31)))
        person.schedule(erng, etype, self.t, horizon)
        n_contacts0 = len(self.contacts)
        trips0 = self.robot.trips  # the robot persists across a lifetime; trips are counted per encounter
        adapter = self._driver(int(self.rng.integers(2**31)))
        log = WheeledLogger(np.random.default_rng(int(self.rng.integers(2**31))))
        perc = MotionPerception(self.vision_cfg, articulated_target=True)  # people move their limbs
        nav_err = float(self.rng.normal(0.0, self.vision_cfg.nav_sigma))
        # The camera works in the abstract encounter's frame (agents.GAP0): the robot knows a person's standard body
        # radius, so it takes the target at the station distance - RADIUS + GAP0. A reach estimate is then the
        # person's surface displacement + GAP0, i.e. the abstract reach the camera codes were defined on.
        from apm_isaac.agents import GAP0 as _G0
        from apm_isaac.agents import RADIUS as _R
        surf = _R - _G0
        n_settle = int(round(settle_s / DT))
        for _ in range(n_settle):
            row, _ = self.step(adapter.idle, adapter, video)
            log.read(row)
        x0, y0, _ = self.robot.pose()
        u = np.array([sx - x0, sy - y0])
        u = u / (np.linalg.norm(u) or 1.0)
        from apm_isaac.agents import GAP0, RADIUS
        person.front = self.robot.front
        back = standoff + RADIUS + self.robot.front - GAP0  # physical geometry mapping (agents.GAP0)
        aim = (sx - back * u[0], sy - back * u[1])
        adapter.reset(self.t, self.robot.pose(), aim, intent)
        win_n, skip_n = int(round(WINDOW_S / DT)), int(round(POST_ARRIVAL_SKIP_S / DT))
        hold_n = int(round(SELF_MOTION_HOLD_S / DT))
        every = max(1, int(round(1.0 / (self.vision_cfg.hz * DT))))
        own = [False] * n_settle
        k, k_arrive, t_arrive, fell, aborted = n_settle, None, None, False, False
        win_ev: list = []
        cam_seen: list[list[float]] = [[] for _ in range(windows)]
        hold_d: list[float] = []
        baseline = slice(max(0, n_settle - 50), n_settle)
        t_start = self.t
        while True:
            x, y, yaw = self.robot.pose()
            dist = math.hypot(sx - x, sy - y)
            if k_arrive is not None and k_arrive + skip_n <= k < k_arrive + skip_n + win_n:
                hold_d.append(dist)
            cmd = adapter.request(self.t, self.robot.pose(), aim, intent) if k_arrive is None else adapter.idle
            row, tripped = self.step(cmd, adapter, video)
            log.read(row)
            own.append(adapter.own_motion(cmd, tripped))
            if k_arrive is not None:
                j = k - (k_arrive + skip_n)
                if j == 0:
                    perc.end_window(dist + nav_err - surf)
                    perc.events = []
                if -int(0.5 / DT) <= j < windows * win_n and j % every == 0:
                    perc.frame(self.depth(), own[-1], dist + nav_err - surf, self.robot.eye_offset)
            if self.robot.fallen(row):
                fell = True
                break
            if k_arrive is None and adapter.arrived:
                k_arrive, t_arrive = k + 1, round(self.t - t_start - settle_s, 2)
            k += 1
            if k_arrive is not None and k > k_arrive + skip_n and (k - k_arrive - skip_n) % win_n == 0:
                w0 = k_arrive + skip_n + len(win_ev) * win_n
                mv = np.asarray(own, bool)
                q = np.ones(len(mv), bool)
                q[w0:] = ~np.array([mv[max(0, jj - hold_n):jj + 1].any() for jj in range(w0, len(mv))], bool)
                win_ev.append(log.summarise(w0, baseline, q, k))
                cam_seen[len(win_ev) - 1] = perc.end_window(dist + nav_err - surf)
                if abort_on_detect and win_ev[-1].detected and len(win_ev) < windows:
                    aborted = True
                    break
                if len(win_ev) >= windows:
                    break
            if self.t - t_start > horizon:
                break
        x, y, _ = self.robot.pose()
        d_stop = round(math.hypot(sx - x, sy - y), 3)
        person.lunges = []
        # return home (part of the continuous lifetime; not sensed for this encounter)
        hx, hy, _ = self._home
        neutral = ExpressionIntent(**{**NEUTRAL.as_dict(), "proximity": 0.5}).sanitized()
        adapter.reset(self.t, self.robot.pose(), (hx, hy), neutral)
        t_back = self.t
        while return_home and not fell and self.t - t_back < max_approach_s:
            row, _ = self.step(adapter.request(self.t, self.robot.pose(), (hx, hy), neutral), adapter, video)
            if self.robot.fallen(row):
                fell = True
            if adapter.arrived:
                break
        contacts = self.contacts[n_contacts0:]
        outcome = {"arrived": k_arrive is not None, "t_arrive": t_arrive, "final_distance": d_stop,
                   "trips": self.robot.trips - trips0, "fell": fell, "windows": [e.as_list() for e in win_ev],
                   "aborted": aborted, "camera": [camera_code(c) for c in cam_seen[:len(win_ev)]],
                   "camera_reaches": cam_seen[:len(win_ev)],
                   "hold_max_dist": round(max(hold_d), 4) if len(hold_d) == win_n else None,
                   "hold_min_dist": round(min(hold_d), 4) if hold_d else None,
                   "quiet_frames": [e.quiet_frames >= MIN_QUIET_FRAMES for e in win_ev]}
        truth = {"type": etype, "contacts": len({round(c[0], 1) for c in contacts}),
                 "contact_impulse_max": max((c[2] for c in contacts), default=0.0), "station": station}
        return outcome, truth


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-convention camera quaternion (x forward, z up) looking from eye to target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    yaw = math.atan2(f[1], f[0])
    pitch = -math.asin(float(np.clip(f[2], -1, 1)))
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])
