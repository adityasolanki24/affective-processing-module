"""APM v1.0: the three bodies in Isaac Sim (PhysX). Importable only inside the Isaac Sim 5.1 environment.

Every body has the same interface as the MuJoCo bodies in `apm_transfer`:
- `apply(v, w) -> tripped`: request a body velocity; the body's own supervisor may override it;
- `advance() -> row`: one 0.02 s control step. `row` has acc, gyro, tilt, finite, z;
- `pose()`, `tilt()`, `fallen(row)`, `push(dvx, dvy)`;
- the eye: `eye_prim` and `eye_offset`, the camera mount.

So the robot-independent pieces (expression adapters, IMU evidence logger, motion perception, runtime) run
unchanged. The IMU is computed from the articulation root: acceleration from the finite difference of its velocity
plus gravity, and angular velocity, both in the body frame. That is what a body-fixed IMU measures, with noise added
by the logger.

**Bodies:**
- `WheeledIsaac`: the APM differential-drive robot (2.9 kg, 0.24 m chassis), built directly in USD physics with
  the same geometry and masses as the MuJoCo model:
  - wheel joints with velocity drives (damping 0.8 N·m·s, 1.5 N·m limit), as MuJoCo's velocity actuators;
  - frictionless casters;
  - acceleration-limited (v, ω) tracking and a shove/tilt supervisor, as `wheeled_world.WheeledRobot`.
- `T1Isaac`: the stock Booster T1 USD shipped with Isaac Sim, driven by the MuJoCo-trained walking policy (NumPy
  copy) with matched joint order, PD gains, armature and home pose. It has a tilt supervisor (25° → zero command
  for 1 s).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

DT = 0.02
REPO = Path(__file__).resolve().parents[2]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    qa = np.asarray(q, float)
    return qa / np.linalg.norm(qa)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def physx_like_isaac_lab(stage, root_prefix: str, depen: float = 1.0) -> None:
    """The articulation settings Isaac Lab trains with: depenetration velocity capped, 8/4 solver iterations,
    self-collision off."""
    from pxr import PhysxSchema, UsdPhysics
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root_prefix):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(depen)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            api.CreateEnabledSelfCollisionsAttr(False)
            api.CreateSolverPositionIterationCountAttr(8)
            api.CreateSolverVelocityIterationCountAttr(4)


class _IsaacBody:
    """Shared IMU, pose and supervisor bookkeeping. Subclasses set `self.art` (SingleArticulation) and `self.world`."""

    art: Any
    world: Any
    physics_dt: float

    def _init_common(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.t, self.dt = 0.0, DT
        self.v_cmd = self.w_cmd = 0.0
        self.brake_until, self.trips = -1.0, 0
        self._v_prev: np.ndarray | None = None

    def pose(self) -> tuple[float, float, float]:
        p, q = self.art.get_world_pose()
        R = quat_to_rot(np.asarray(q))
        return float(p[0]), float(p[1]), math.atan2(R[1, 0], R[0, 0])

    def rot(self) -> np.ndarray:
        return quat_to_rot(np.asarray(self.art.get_world_pose()[1]))

    def tilt(self) -> float:
        return math.acos(float(np.clip(self.rot()[2, 2], -1.0, 1.0)))

    def teleport(self, x: float, y: float, yaw: float) -> None:
        """Experiments only: put the robot at (x, y, yaw) at rest, keeping its height and joint configuration (used
        to start each encounter from its approach point instead of walking back; videos walk)."""
        z = float(np.asarray(self.art.get_world_pose()[0])[2])
        self.art.set_world_pose(position=np.array([x, y, z]), orientation=yaw_quat(yaw))
        self.art.set_linear_velocity(np.zeros(3))
        self.art.set_angular_velocity(np.zeros(3))
        self.v_cmd = self.w_cmd = 0.0
        self._v_prev = None

    def push(self, dvx: float, dvy: float) -> None:
        v = np.asarray(self.art.get_linear_velocity(), float)
        self.art.set_linear_velocity(v + np.array([dvx, dvy, 0.0]))

    def _imu(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.rot()
        v = np.asarray(self.art.get_linear_velocity(), float)
        a_world = np.zeros(3) if self._v_prev is None else (v - self._v_prev) / DT
        self._v_prev = v
        acc = R.T @ (a_world + np.array([0.0, 0.0, 9.81]))
        gyro = R.T @ np.asarray(self.art.get_angular_velocity(), float)
        return acc, gyro

    def _row(self) -> dict:
        acc, gyro = self._imu()
        p = np.asarray(self.art.get_world_pose()[0], float)
        return {"acc": acc, "gyro": gyro, "tilt": math.degrees(self.tilt()),
                "finite": bool(np.all(np.isfinite(p))), "z": float(p[2])}


class WheeledIsaac(_IsaacBody):
    R_WHEEL, TRACK = 0.04, 0.22
    SHOVE_TRIP, TILT_TRIP, TILT_FALL, BRAKE_S = 0.35, math.radians(25), math.radians(60), 0.5
    eye_offset = 0.11
    front = 0.12  # chassis extent ahead of the origin

    def __init__(self, world, path: str, pos: tuple[float, float], yaw: float, seed: int, physics_dt: float = 0.002):
        from isaacsim.core.prims import SingleArticulation
        self.world, self.path, self.physics_dt = world, path, physics_dt
        self._build(world.stage, path, pos, yaw)
        physx_like_isaac_lab(world.stage, path, depen=0.3)
        self.art = world.scene.add(SingleArticulation(path, name=path.split("/")[-1]))
        self.eye_prim = f"{path}/base/eye"
        self._init_common(seed)

    @staticmethod
    def _build(stage, path: str, pos: tuple[float, float], yaw: float) -> None:
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

        def body(name: str, parent: str, xyz, mass: float):
            p = f"{parent}/{name}"
            x = UsdGeom.Xform.Define(stage, p)
            x.AddTranslateOp().Set(Gf.Vec3d(*xyz))
            UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
            return p

        def shape(kind, p: str, size, xyz=(0, 0, 0), rot=None, color=(0.25, 0.45, 0.8), mat=None):
            g = {"box": UsdGeom.Cube, "cyl": UsdGeom.Cylinder, "sph": UsdGeom.Sphere}[kind].Define(stage, p)
            xf = UsdGeom.XformCommonAPI(g)
            xf.SetTranslate(Gf.Vec3d(*xyz))
            if rot is not None:
                xf.SetRotate(Gf.Vec3f(*rot))
            if kind == "box":
                g.CreateSizeAttr(1.0)
                xf.SetScale(Gf.Vec3f(*[2 * s for s in size]))
            elif kind == "cyl":
                g.CreateRadiusAttr(size[0])
                g.CreateHeightAttr(2 * size[1])
                g.CreateAxisAttr("Z")
            else:
                g.CreateRadiusAttr(size[0])
            g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            if mat is not None:
                UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                    "physics")
            return g

        def material(name: str, mu: float):
            m = UsdShade.Material.Define(stage, f"{path}/mat_{name}")
            api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
            api.CreateStaticFrictionAttr(mu)
            api.CreateDynamicFrictionAttr(mu)
            api.CreateRestitutionAttr(0.0)
            px = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
            px.CreateFrictionCombineModeAttr("min" if mu == 0 else "average")
            return m

        root = UsdGeom.Xform.Define(stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], 0.0))
        root.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        no_fric, tyre = material("caster", 0.0), material("tyre", 1.2)
        base = body("base", path, (0, 0, 0.062), 2.69)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(base))
        shape("box", f"{base}/chassis", (0.12, 0.09, 0.03))
        shape("cyl", f"{base}/mast", (0.015, 0.07), (0.06, 0, 0.1), color=(0.2, 0.2, 0.2))
        shape("box", f"{base}/head", (0.03, 0.04, 0.025), (0.07, 0, 0.18), color=(0.9, 0.9, 0.9))
        shape("sph", f"{base}/caster_f", (0.02,), (0.095, 0, -0.042), mat=no_fric)
        shape("sph", f"{base}/caster_b", (0.02,), (-0.095, 0, -0.042), mat=no_fric)
        UsdGeom.Xform.Define(stage, f"{base}/eye").AddTranslateOp().Set(Gf.Vec3d(0.11, 0, 0.18))
        for name, y in (("wheel_l", 0.11), ("wheel_r", -0.11)):
            w = body(name, path, (0, y, 0.04), 0.15)
            shape("cyl", f"{w}/tyre", (0.04, 0.012), rot=(90, 0, 0), color=(0.1, 0.1, 0.1), mat=tyre)
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/j_{name}")
            j.CreateBody0Rel().SetTargets([Sdf.Path(base)])
            j.CreateBody1Rel().SetTargets([Sdf.Path(w)])
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0, y, -0.022))
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(0.0)
            d.CreateDampingAttr(0.8 * 180 / math.pi)  # USD angular drive damping is per degree
            d.CreateMaxForceAttr(1.5)
            d.CreateTargetVelocityAttr(0.0)
        cam_body = stage.GetPrimAtPath(base)
        assert cam_body.IsValid()

    def apply(self, v: float, w: float) -> bool:
        from isaacsim.core.utils.types import ArticulationAction
        lin = np.asarray(self.art.get_linear_velocity(), float)
        tripped = False
        if float(np.hypot(lin[0], lin[1])) > self.SHOVE_TRIP or self.tilt() > self.TILT_TRIP:
            if self.t >= self.brake_until:
                self.trips += 1
            self.brake_until = self.t + self.BRAKE_S
            tripped = True
        if self.t < self.brake_until:
            v, w = 0.0, 0.0
        self.v_cmd += float(np.clip(v - self.v_cmd, -0.4 * DT, 0.4 * DT))
        self.w_cmd += float(np.clip(w - self.w_cmd, -3.0 * DT, 3.0 * DT))
        wl = (self.v_cmd - self.w_cmd * self.TRACK / 2) / self.R_WHEEL
        wr = (self.v_cmd + self.w_cmd * self.TRACK / 2) / self.R_WHEEL
        names = list(self.art.dof_names)
        vel = np.zeros(len(names))
        vel[names.index("j_wheel_l")], vel[names.index("j_wheel_r")] = wl, wr
        self.art.apply_action(ArticulationAction(joint_velocities=vel))
        return tripped

    def advance(self) -> dict:
        for _ in range(int(round(DT / self.physics_dt))):
            self.world.step(render=False)
        self.t += DT
        return self._row()

    def fallen(self, row: dict) -> bool:
        return math.radians(row["tilt"]) > self.TILT_FALL or not row["finite"]


class NovaCarterIsaac(_IsaacBody):
    """NVIDIA Nova Carter, the stock Isaac Sim asset: a commercial differential-drive AMR. It is APM v1.0's wheeled
    embodiment (user decision 2026-10-19). The custom `WheeledIsaac` remains only as a MuJoCo cross-physics
    reference.

    - **Asset.** Physics, wheel drives and casters are the asset's own. Nothing is retuned.
    - **Geometry measured from the asset** after the first reset: wheel radius (axle height), track (wheel
      separation), front extent (bounding box ahead of the base origin), mass.
    - **Motion.** The same (v, w) interface and acceleration limits as the reference robot, executed as wheel
      velocity targets on the asset's wheel drives.
    - **Safety.** The same supervisor: base speed > 0.35 m/s or tilt > 25 deg -> zero command for 0.5 s and a
      counted trip.
    - **Eye.** APM's depth eye is mounted at the front Hawk stereo camera's forward position and height, on the
      centre line (chassis frame, facing forward)."""

    SHOVE_TRIP, TILT_TRIP, TILT_FALL, BRAKE_S = 0.35, math.radians(25), math.radians(60), 0.5
    CANDIDATES = ("/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd", "/Isaac/Robots/Carter/nova_carter_sensors.usd",
                  "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd")
    R_WHEEL, TRACK = 0.14, 0.413  # replaced by measured values in post_reset
    front, eye_offset = 0.35, 0.25

    def __init__(self, world, path: str, pos: tuple[float, float], yaw: float, seed: int,
                 physics_dt: float = 0.002):
        import omni.client
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        self.world, self.path, self.physics_dt = world, path, physics_dt
        root = get_assets_root_path()
        self.usd = next((root + c for c in self.CANDIDATES if omni.client.stat(root + c)[0] == omni.client.Result.OK),
                        None)
        if self.usd is None:
            raise FileNotFoundError("Nova Carter asset not found under " + root)
        add_reference_to_stage(self.usd, path)
        top = world.stage.GetPrimAtPath(path)
        prims = list(Usd.PrimRange(top, Usd.TraverseInstanceProxies()))
        bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        chassis = next((p for p in bodies if "chassis" in p.GetName().lower()), bodies[0] if bodies else top)
        # geometry at the asset's own origin and heading, before placement
        cache = UsdGeom.XformCache()
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(top).ComputeAlignedRange()
        c0 = cache.GetLocalToWorldTransform(chassis).ExtractTranslation()
        self.front = float(bound.GetMax()[0] - c0[0])
        self.height = float(bound.GetMax()[2] - bound.GetMin()[2])
        art_root = next((str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)), path)
        # contact handling only: overlap with a person's kinematic collider is resolved at <= 0.3 m/s (as on the
        # reference robot), so a lunge transmits its contact velocity, not a depenetration spike
        from pxr import PhysxSchema
        for p in bodies:
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr(0.3)
        self.art = world.scene.add(SingleArticulation(art_root, name=path.split("/")[-1]))
        self.start = (pos, yaw)
        # the eye: an Xform on the chassis link at the front Hawk camera's position, axes = chassis axes
        hawk = next((p for p in prims if "front_hawk" in p.GetName().lower()), None)
        # the Hawk's lens: its first camera prim (the module's root sits behind the housing)
        lens = next((q for q in Usd.PrimRange(hawk, Usd.TraverseInstanceProxies()) if q.IsA(UsdGeom.Camera)),
                    hawk) if hawk is not None else None
        c_w = cache.GetLocalToWorldTransform(chassis)
        if lens is not None:
            local = c_w.GetInverse().Transform(cache.GetLocalToWorldTransform(lens).ExtractTranslation())
        else:
            local = Gf.Vec3d(0.3, 0.0, 0.35)
        # on the centre line at the lens height, and clear of the robot's own front face (M4 eye check: from the
        # module root the depth eye saw its own housing at 6 cm)
        local = Gf.Vec3d(max(float(local[0]), self.front) + 0.02, 0.0, local[2])
        eye = UsdGeom.Xform.Define(world.stage, f"{chassis.GetPath()}/apm_eye")
        eye.AddTranslateOp().Set(local)
        self.eye_prim = str(eye.GetPath())
        self.eye_local = [round(float(v), 4) for v in local]
        self.eye_from_hawk = lens is not None and lens is not hawk
        self.chassis_name = chassis.GetName()
        self.measured: dict | None = None
        self._init_common(seed)

    def post_reset(self) -> None:
        """Find the wheel joints and measure the asset's geometry (once)."""
        names = list(self.art.dof_names)
        self.wl = next(i for i, n in enumerate(names) if "wheel" in n.lower() and "left" in n.lower())
        self.wr = next(i for i, n in enumerate(names) if "wheel" in n.lower() and "right" in n.lower())
        self.dof_names = names
        (x0, y0), yaw0 = self.start
        self.art.set_world_pose(position=np.array([x0, y0, float(self.art.get_world_pose()[0][2])]),
                                orientation=yaw_quat(yaw0))
        self.art.set_linear_velocity(np.zeros(3))
        self.art.set_angular_velocity(np.zeros(3))
        if self.measured is not None:
            return
        links = list(self.art._articulation_view._metadata.link_names)
        tf = np.asarray(self.art._articulation_view._physics_view.get_link_transforms())[0]
        wl = next((i for i, n in enumerate(links) if "wheel" in n.lower() and "left" in n.lower()), None)
        wr = next((i for i, n in enumerate(links) if "wheel" in n.lower() and "right" in n.lower()), None)
        if wl is not None and wr is not None:
            self.TRACK = float(np.linalg.norm(tf[wl, :2] - tf[wr, :2]))
            self.R_WHEEL = float((tf[wl, 2] + tf[wr, 2]) / 2)
        self.eye_offset = float(self.eye_local[0])  # the articulation root is the chassis link
        mass = float(np.asarray(self.art._articulation_view.get_body_masses())[0].sum())
        self.measured = {"usd": self.usd, "wheel_radius": round(self.R_WHEEL, 4), "track": round(self.TRACK, 4),
                         "front": round(self.front, 3), "eye_local": self.eye_local,
                         "eye_from_front_hawk": self.eye_from_hawk, "eye_offset": round(self.eye_offset, 3),
                         "mass_kg": round(mass, 2), "height_m": round(self.height, 3), "dofs": names,
                         "links": links}

    def apply(self, v: float, w: float) -> bool:
        from isaacsim.core.utils.types import ArticulationAction
        lin = np.asarray(self.art.get_linear_velocity(), float)
        tripped = False
        if float(np.hypot(lin[0], lin[1])) > self.SHOVE_TRIP or self.tilt() > self.TILT_TRIP:
            if self.t >= self.brake_until:
                self.trips += 1
            self.brake_until = self.t + self.BRAKE_S
            tripped = True
        if self.t < self.brake_until:
            v, w = 0.0, 0.0
        self.v_cmd += float(np.clip(v - self.v_cmd, -0.4 * DT, 0.4 * DT))
        self.w_cmd += float(np.clip(w - self.w_cmd, -3.0 * DT, 3.0 * DT))
        wl = (self.v_cmd - self.w_cmd * self.TRACK / 2) / self.R_WHEEL
        wr = (self.v_cmd + self.w_cmd * self.TRACK / 2) / self.R_WHEEL
        self.art.apply_action(ArticulationAction(joint_velocities=np.array([wl, wr]),
                                                 joint_indices=np.array([self.wl, self.wr])))
        return tripped

    def advance(self) -> dict:
        for _ in range(int(round(DT / self.physics_dt))):
            self.world.step(render=False)
        self.t += DT
        return self._row()

    def fallen(self, row: dict) -> bool:
        return math.radians(row["tilt"]) > self.TILT_FALL or not row["finite"]


class T1Isaac(_IsaacBody):
    """Booster T1 (stock Isaac Sim USD) on the MuJoCo-trained walking policy."""

    TILT_TRIP, TILT_FALL, Z_FALL, BRAKE_S = math.radians(25), math.radians(50), 0.45, 1.0
    eye_offset = 0.16
    front = 0.15  # feet/torso extent ahead of the base origin

    def __init__(self, world, path: str, pos: tuple[float, float], yaw: float, seed: int, usd: str,
                 physics_dt: float = 0.002):
        import json

        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdPhysics
        self.world, self.physics_dt = world, physics_dt
        ref = json.loads((REPO / "experiments" / "M_release" / "configs" / "t1_mujoco_ref.json").read_text())
        self.ref = ref
        add_reference_to_stage(usd, path)
        root = path
        for prim in world.stage.Traverse():
            if str(prim.GetPath()).startswith(path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                root = str(prim.GetPath())
                break
        self.art = world.scene.add(SingleArticulation(root, name=path.split("/")[-1]))
        self.start = (pos, yaw)
        z = np.load(REPO / "experiments" / "K_recovery" / "humanoid" / "trained" / "t1_policy.npz")
        self.mean, self.std = z["mean"], z["std"]
        n = len([k for k in z.files if k.endswith("_kernel")])
        self.W = [(z[f"hidden_{i}_kernel"], z[f"hidden_{i}_bias"]) for i in range(n)]
        self.eye_prim: str | None = None  # set after reset (the H2 link path)
        self._init_common(seed)

    def post_reset(self) -> None:
        names = list(self.art.dof_names)
        mj = [x["joint"] for x in self.ref["actuators"]]
        self.idx = np.array([names.index(j) for j in mj])
        self.home = np.array([x["home"] for x in self.ref["actuators"]])
        kp, kd = np.zeros(len(names)), np.zeros(len(names))
        kp[self.idx] = [x["kp"] for x in self.ref["actuators"]]
        kd[self.idx] = [x["damping"] for x in self.ref["actuators"]]
        self.art.get_articulation_controller().set_gains(kps=kp, kds=kd)
        q = np.zeros(len(names))
        q[self.idx] = self.home
        self.art.set_joint_positions(q)
        self.art.set_joint_velocities(np.zeros(len(names)))
        (x, y), yaw = self.start
        self.art.set_world_pose(position=np.array([x, y, self.ref["home_base_pos"][2]]), orientation=yaw_quat(yaw))
        self.last = np.zeros(23)
        self.phase = np.array([0.0, math.pi])
        for prim in self.world.stage.Traverse():
            if prim.GetName() == "H2" and str(prim.GetPath()).startswith(str(self.art.prim_path).rsplit("/", 1)[0]):
                self.eye_prim = str(prim.GetPath())
                break

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / self.std
        for i, (w, b) in enumerate(self.W):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = x / (1.0 + np.exp(-x))
        return np.tanh(x[: x.shape[0] // 2])

    def apply(self, v: float, w: float) -> bool:
        tripped = False
        if self.tilt() > self.TILT_TRIP:
            if self.t >= self.brake_until:
                self.trips += 1
            self.brake_until = self.t + self.BRAKE_S
            tripped = True
        if self.t < self.brake_until:
            v, w = 0.0, 0.0
        self.v_cmd += float(np.clip(v - self.v_cmd, -1.0 * DT, 1.0 * DT))
        self.w_cmd += float(np.clip(w - self.w_cmd, -3.0 * DT, 3.0 * DT))
        return tripped

    def advance(self) -> dict:
        from isaacsim.core.utils.types import ArticulationAction
        R = self.rot()
        lin = R.T @ np.asarray(self.art.get_linear_velocity(), float)
        gyro = R.T @ np.asarray(self.art.get_angular_velocity(), float)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        q = np.asarray(self.art.get_joint_positions(), float)[self.idx]
        dq = np.asarray(self.art.get_joint_velocities(), float)[self.idx]
        cmd = np.array([self.v_cmd, 0.0, self.w_cmd])
        ph = self.phase if np.linalg.norm(cmd) > 0.01 else np.ones(2) * math.pi
        obs = np.concatenate([lin, gyro, grav, cmd, q - self.home, dq, self.last, np.cos(ph), np.sin(ph)])
        act = self._policy(obs)
        target = np.asarray(self.art.get_joint_positions(), float).copy()
        target[self.idx] = self.home + act
        self.art.apply_action(ArticulationAction(joint_positions=target))
        for _ in range(int(round(DT / self.physics_dt))):
            self.world.step(render=False)
        self.last = act
        nxt = np.fmod(self.phase + 2 * math.pi * DT * 1.5 + math.pi, 2 * math.pi) - math.pi
        self.phase = nxt if np.linalg.norm(cmd) > 0.01 else np.ones(2) * math.pi
        self.t += DT
        return self._row()

    def fallen(self, row: dict) -> bool:
        return math.radians(row["tilt"]) > self.TILT_FALL or row["z"] < self.Z_FALL or not row["finite"]


# T1 PD gains (kp, kd) as trained (`apm_isaac.lab.t1.T1_CFG`)
T1_GAINS = {".*_Hip_.*": (100.0, 3.0), ".*_Knee_Pitch": (150.0, 4.0), "Waist": (100.0, 3.0),
            ".*_Ankle_.*": (40.0, 2.0), ".*_Shoulder_.*": (40.0, 2.0), ".*_Elbow_.*": (40.0, 2.0),
            "AAHead_yaw": (20.0, 1.0), "Head_pitch": (20.0, 1.0)}


class LabPolicyBody(_IsaacBody):
    """A legged body driven by a policy trained in Isaac Lab (this repository's tasks), deployed with exactly the
    training observation: base linear and angular velocity, projected gravity, (vx, vy, ω) command, joint position
    and velocity offsets, last action. Control 50 Hz (physics `sim_dt`, `decimation` from `env_cfg.json`).
    Targets: default + scale × action. A tilt supervisor (25° → zero command for 1 s) is the body's own."""

    TILT_TRIP, TILT_FALL, Z_FALL, BRAKE_S = math.radians(25), math.radians(50), 0.45, 1.0
    eye_offset = 0.16
    front = 0.15  # feet/torso extent ahead of the base origin
    eye_link, trunk_link = "H2", "Trunk"
    eye_ahead = 0.10   # m ahead of the head link
    eye_near = 0.05    # m: the depth eye's near clip (it sits just outside the head)

    def __init__(self, world, path: str, usd: str, pos: tuple[float, float], yaw: float, seed: int, policy_dir: Path,
                 base_z: float, physics_dt: float = 0.005):
        import json

        import torch
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdPhysics
        self.world, self.physics_dt = world, physics_dt
        self.cfg = json.loads((policy_dir / "env_cfg.json").read_text())
        # the controller's measured speed response (m3_lab_battery.py --calibrate), inverted on forward commands
        vm = policy_dir / "velocity_map.json"
        self.vmap = json.loads(vm.read_text()) if vm.exists() else None
        self.policy = torch.jit.load(str(policy_dir / "policy.pt"), map_location="cpu").eval()
        self.torch = torch
        add_reference_to_stage(usd, path)
        root = path
        for prim in world.stage.Traverse():
            if str(prim.GetPath()).startswith(path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                root = str(prim.GetPath())
                break
        self.art = world.scene.add(SingleArticulation(root, name=path.split("/")[-1]))
        self.start, self.base_z = (pos, yaw), base_z
        self.eye_prim: str | None = None
        self.decim = int(round(DT / physics_dt))
        self._init_common(seed)

    @classmethod
    def t1(cls, world, path: str, pos: tuple[float, float], yaw: float, seed: int, policy_dir: Path,
           physics_dt: float = 0.005) -> LabPolicyBody:
        """The Booster T1 with the physics it was trained in (`apm_isaac.lab.t1`): stock T1_locomotion.usd, the
        trained PD gains and armature, self-collision off, 8/4 solver iterations."""
        from isaacsim.storage.native import get_assets_root_path
        from pxr import PhysxSchema, UsdPhysics
        usd = get_assets_root_path() + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd"
        body = cls(world, path, usd, pos, yaw, seed, policy_dir, 0.70, physics_dt)
        physx_like_isaac_lab(world.stage, path)
        for prim in world.stage.Traverse():
            if str(prim.GetPath()).startswith(path) and prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(0.01)
        body.gains = T1_GAINS
        return body

    gains: dict[str, tuple[float, float]] | None = None

    def post_reset(self, gains: dict[str, tuple[float, float]] | None = None) -> None:
        import re
        gains = gains or self.gains
        names = list(self.art.dof_names)
        self.idx = np.array([names.index(j) for j in self.cfg["joint_names"]])
        self.default = np.asarray(self.cfg["default_joint_pos"], float)
        if gains:
            kp, kd = np.zeros(len(names)), np.zeros(len(names))
            for i, n in enumerate(names):
                for pat, (p, d) in gains.items():
                    if re.fullmatch(pat, n):
                        kp[i], kd[i] = p, d
            self.art.get_articulation_controller().set_gains(kps=kp, kds=kd)
        q = np.zeros(len(names))
        q[self.idx] = self.default
        self.art.set_joint_positions(q)
        self.art.set_joint_velocities(np.zeros(len(names)))
        (x, y), yaw = self.start
        self.art.set_world_pose(position=np.array([x, y, self.base_z]), orientation=yaw_quat(yaw))
        self.last = np.zeros(len(self.idx))
        if self.eye_prim is None:
            self._make_eye_frame()

    def _make_eye_frame(self) -> None:
        """The depth eye: 10 cm ahead of the head link, with the trunk's axes (x forward, z up) at the asset's rest
        pose, so it is level and faces forward (the head link's own frame is rolled). Computed from the USD rest
        transforms (PhysX link poses update only on a step)."""
        from pxr import Gf, UsdGeom
        root = str(self.art.prim_path).rsplit("/", 1)[0]
        prims = [p for p in self.world.stage.Traverse() if str(p.GetPath()).startswith(root)]
        head = next(p for p in prims if p.GetName() == self.eye_link)
        trunk = next((p for p in prims if p.GetName() == self.trunk_link), head)
        cache = UsdGeom.XformCache()
        m_h, m_t = cache.GetLocalToWorldTransform(head), cache.GetLocalToWorldTransform(trunk)
        r_t = m_t.ExtractRotationMatrix()
        fwd = Gf.Vec3d(r_t.GetRow(0))  # trunk x axis in world (row-vector convention)
        p_eye = m_h.ExtractTranslation() + fwd * self.eye_ahead
        m_e = Gf.Matrix4d(1.0)
        m_e.SetRotateOnly(r_t)
        m_e.SetTranslateOnly(p_eye)
        eye = UsdGeom.Xform.Define(self.world.stage, f"{head.GetPath()}/apm_eye")
        eye.AddTransformOp().Set(m_e * m_h.GetInverse())
        self.eye_prim = str(eye.GetPath())
        # the body's extent ahead of the trunk origin along its forward axis, at the asset's rest pose
        from pxr import Usd
        top = self.world.stage.GetPrimAtPath(root)
        bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bbc.ComputeWorldBound(top).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        t0 = np.array(m_t.ExtractTranslation())
        f = np.array(fwd)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        self.front = float(max(((corners - t0) @ f).max(), 0.05))
        self.eye_offset = float((np.array(p_eye) - t0) @ f)

    def apply(self, v: float, w: float) -> bool:
        tripped = False
        if self.tilt() > self.TILT_TRIP:
            if self.t >= self.brake_until:
                self.trips += 1
            self.brake_until = self.t + self.BRAKE_S
            tripped = True
        if self.t < self.brake_until:
            v, w = 0.0, 0.0
        self.v_cmd += float(np.clip(v - self.v_cmd, -1.0 * DT, 1.0 * DT))
        self.w_cmd += float(np.clip(w - self.w_cmd, -3.0 * DT, 3.0 * DT))
        return tripped

    def command_for(self, v: float) -> float:
        """Forward command that yields speed v (the calibrated map); identity without a map or for v <= 0."""
        if self.vmap is None or v <= 1e-6:
            return v
        cmd, meas = np.asarray(self.vmap["command"]), np.asarray(self.vmap["measured"])
        ok = meas > 0.02  # commands that make progress
        if not ok.any():
            return v
        cmd, meas = np.concatenate([[0.0], cmd[ok]]), np.concatenate([[0.0], meas[ok]])  # anchored at rest
        return float(np.interp(v, meas, cmd, right=cmd[-1]))

    def observation(self) -> np.ndarray:
        R = self.rot()
        lin = R.T @ np.asarray(self.art.get_linear_velocity(), float)
        ang = R.T @ np.asarray(self.art.get_angular_velocity(), float)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        q = np.asarray(self.art.get_joint_positions(), float)[self.idx]
        dq = np.asarray(self.art.get_joint_velocities(), float)[self.idx]
        cmd = np.array([self.command_for(self.v_cmd), 0.0, self.w_cmd])
        return np.concatenate([lin, ang, grav, cmd, q - self.default, dq, self.last]).astype(np.float32)

    def advance(self) -> dict:
        from isaacsim.core.utils.types import ArticulationAction
        with self.torch.no_grad():
            act = self.policy(self.torch.from_numpy(self.observation())[None]).numpy()[0]
        target = np.asarray(self.art.get_joint_positions(), float).copy()
        target[self.idx] = self.default + self.cfg["action_scale"] * act
        self.art.apply_action(ArticulationAction(joint_positions=target))
        for _ in range(self.decim):
            self.world.step(render=False)
        self.last = act.astype(float)
        self.t += DT
        return self._row()

    def fallen(self, row: dict) -> bool:
        return math.radians(row["tilt"]) > self.TILT_FALL or row["z"] < self.Z_FALL or not row["finite"]
