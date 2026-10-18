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


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


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

    def __init__(self, world, path: str, pos: tuple[float, float], yaw: float, seed: int, physics_dt: float = 0.002):
        from isaacsim.core.prims import SingleArticulation
        self.world, self.path, self.physics_dt = world, path, physics_dt
        self._build(world.stage, path, pos, yaw)
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


class T1Isaac(_IsaacBody):
    """Booster T1 (stock Isaac Sim USD) on the MuJoCo-trained walking policy."""

    TILT_TRIP, TILT_FALL, Z_FALL, BRAKE_S = math.radians(25), math.radians(50), 0.45, 1.0
    eye_offset = 0.16

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
