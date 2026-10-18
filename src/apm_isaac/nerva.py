"""APM v1.0: NERVA (Open Duck Mini v2) in Isaac Sim, on its frozen neutral controller (needs Isaac Sim 5.1).

**Import** (`prepare_usd`, `match_mujoco`). The MJCF NERVA validated against (`open_duck_mini_v2_backlash.xml`, Open
Duck Playground at NERVA's pinned commit) is converted with Isaac Lab's `MjcfConverter`. M3 found and fixed four
import faults:
- **Two top-level `<default>` blocks**, which the importer cannot read. They are merged (semantics unchanged).
- **Two joints per leg body** (the actuated joint and a ±0.5° backlash joint on the same axis). PhysX has one joint
  per link, so the 10 backlash joints are removed. In Isaac Sim the legs have no backlash: a documented physics
  difference.
- **Inertials.** The floating "base" body has no inertial in the MJCF; PhysX gave it 1 kg (3.1 kg robot instead of
  2.1). Every link now gets MuJoCo's compiled mass, centre of mass and principal inertia.
- **Joint frames.** The importer rotated several joints about the wrong axis: joints reached their targets, yet the
  legs ended up 15 cm out. Every revolute joint's frames, axis and limits are rebuilt from MuJoCo's compiled model.
  Forward kinematics then match MuJoCo to 0.0 mm and 0.02° (home pose, and each joint at ±0.5 rad).
- **Gains.** The importer's leg drive gains were per-degree copies of MuJoCo's per-radian kp (57× too stiff).
  MuJoCo's kp 17.11 / damping 0.56 / ±3.23 N·m are written to PhysX in SI units and read back.

**Controller** (`NervaIsaac`). NERVA's frozen neutral ONNX policy (SHA-256 checked), with NERVA's own control step
(`nerva/sim/open_duck.py`):
- the 101-value observation, with NERVA's training observation noise;
- the `NeutralPhaseClock(27)`;
- targets = default + 0.25 × action, rate-limited to 5.24 rad/s;
- head offsets added to the head motor targets afterwards (motor targets only);
- 50 Hz control, 500 Hz physics.

The observation comes from Isaac's own sensing:
- **gyro and accelerometer** at NERVA's IMU site: specific force from the site velocity over the last physics step;
- **foot contacts** from PhysX contact reports on the two foot links.

The robot is driven through H's admitted interface (`apm_express.adapter.MotionRequest`: the seven trained
commands plus head offsets). Every request first passes NERVA's `SafetyMonitor` (neutral fallback), exactly as in
`nerva_world`.

**Eye.** As in programme L (`vision.robot_view`): 5 cm ahead of the head body along the face direction (the world +x
of the home pose, in the head frame), level with the head's up axis.
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from apm_isaac.robots import DT, _IsaacBody, quat_to_rot, yaw_quat

REPO = Path(__file__).resolve().parents[2]
REF_PATH = REPO / "experiments" / "M_release" / "configs" / "nerva_mujoco_ref.json"
HEAD = slice(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll in actuator order


def ref() -> dict:
    return json.loads(REF_PATH.read_text())


def prepare_usd(odp: Path, out_dir: Path, force: bool = False) -> tuple[str, Path, dict]:
    """MJCF → USD with the M3 edits (merged defaults, backlash joints removed). Returns (usd, edited MJCF, report)."""
    import omni.kit.app
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.mjcf")
    omni.kit.app.get_app().update()
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    xml = Path(odp) / "playground" / "open_duck_mini_v2" / "xmls" / "open_duck_mini_v2_backlash.xml"
    tree = ET.parse(xml)
    tops = tree.getroot().findall("default")
    for extra in tops[1:]:
        for child in list(extra):
            tops[0].append(child)
        tree.getroot().remove(extra)
    removed = []
    for body in tree.getroot().iter("body"):
        for jn in list(body.findall("joint")):
            if (jn.get("name") or "").endswith("_backlash"):
                body.remove(jn)
                removed.append(jn.get("name"))
    edited = xml.with_name("open_duck_mini_v2_isaac.xml")  # next to the original: mesh paths still resolve
    tree.write(edited)
    conv = MjcfConverter(MjcfConverterCfg(asset_path=str(edited), usd_dir=str(Path(out_dir).resolve()),
                                          usd_file_name="open_duck.usd", fix_base=False, import_sites=True,
                                          force_usd_conversion=force, make_instanceable=False))
    return (str(Path(conv.usd_path).resolve()), edited,
            {"defaults_merged": len(tops), "backlash_removed": removed, "mjcf": str(edited)})


def _quat_mul(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _x_to(axis) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating +x onto `axis`."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    c = float(np.clip(u[0], -1.0, 1.0))
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.cross([1.0, 0.0, 0.0], u)
    ax /= np.linalg.norm(ax)
    h = math.acos(c) / 2
    return np.array([math.cos(h), *(math.sin(h) * ax)])


def match_mujoco(stage, root: str, mjcf: Path) -> dict:
    """Copy MuJoCo's compiled inertials and joint frames/limits onto the converted robot under `root`."""
    import mujoco
    from pxr import Gf, Usd, UsdPhysics
    mj = mujoco.MjModel.from_xml_path(str(mjcf))
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()))
    bodies = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(1, mj.nbody)}
    inertials, joints = [], []
    for p in prims:
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in bodies:
            i = bodies[p.GetName()]
            m = UsdPhysics.MassAPI.Apply(p)
            m.CreateMassAttr(max(float(mj.body_mass[i]), 1e-3))
            m.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in mj.body_ipos[i])))
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*(max(float(v), 1e-7) for v in mj.body_inertia[i])))
            m.CreatePrincipalAxesAttr(Gf.Quatf(*(float(v) for v in mj.body_iquat[i])))
            inertials.append(p.GetName())
        if p.IsA(UsdPhysics.RevoluteJoint):
            jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, p.GetName())
            if jid < 0:
                continue
            b = int(mj.jnt_bodyid[jid])
            parent = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(mj.body_parentid[b]))
            rj = UsdPhysics.RevoluteJoint(p)
            b1 = [str(t) for t in rj.GetBody1Rel().GetTargets()]
            swapped = bool(b1 and b1[0].rsplit("/", 1)[-1] == parent)
            qa = _x_to(mj.jnt_axis[jid])
            pos_c = np.asarray(mj.jnt_pos[jid], float)
            R_b = quat_to_rot(np.asarray(mj.body_quat[b], float))
            pos_p = np.asarray(mj.body_pos[b], float) + R_b @ pos_c
            q_p = _quat_mul(mj.body_quat[b], qa)
            f_par = (Gf.Vec3f(*pos_p.tolist()), Gf.Quatf(*q_p.tolist()))
            f_chi = (Gf.Vec3f(*pos_c.tolist()), Gf.Quatf(*qa.tolist()))
            (p0, r0), (p1, r1) = (f_chi, f_par) if swapped else (f_par, f_chi)
            rj.CreateAxisAttr("X")
            rj.CreateLocalPos0Attr().Set(p0)
            rj.CreateLocalRot0Attr().Set(r0)
            rj.CreateLocalPos1Attr().Set(p1)
            rj.CreateLocalRot1Attr().Set(r1)
            lo, hi = (float(np.degrees(v)) for v in mj.jnt_range[jid])
            if swapped:
                lo, hi = -hi, -lo
            if mj.jnt_limited[jid]:
                rj.CreateLowerLimitAttr().Set(lo)
                rj.CreateUpperLimitAttr().Set(hi)
            joints.append(p.GetName())
    return {"inertials_from_mujoco": inertials, "joint_frames_from_mujoco": joints,
            "mujoco_total_mass": round(float(mj.body_mass[1:].sum()), 4)}


def head_eye_frame(mjcf: Path, home_qpos) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, 3x3 rotation) of L's eye in the head body frame: 5 cm along the face direction, x = face,
    z = the head's up direction at the home pose made orthogonal to x. Third: the eye in the base frame at home."""
    import mujoco
    mj = mujoco.MjModel.from_xml_path(str(mjcf))
    md = mujoco.MjData(mj)
    md.qpos[:7] = home_qpos[:7]
    mujoco.mj_kinematics(mj, md)
    hb = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "head_assembly")
    r_head = md.xmat[hb].reshape(3, 3)
    base_yaw = math.atan2(2 * (home_qpos[3] * home_qpos[6]), 1 - 2 * home_qpos[6] ** 2)
    face = r_head.T @ np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
    face /= np.linalg.norm(face)
    up = r_head.T @ np.array([0.0, 0.0, 1.0])
    up = up - face * float(up @ face)
    up /= np.linalg.norm(up)
    eye_base = md.xpos[hb] + r_head @ (0.05 * face) - np.asarray(home_qpos[:3], float)
    return 0.05 * face, np.column_stack([face, np.cross(up, face), up]), eye_base


class NervaIsaac(_IsaacBody):
    TILT_FALL_DEG = 45.0  # K/L: a fall is tilt > 45 deg
    eye_near = 0.02       # m: the eye sits 1 cm beyond the head shell
    eye_offset = 0.08     # replaced by the measured mounting in post_reset
    front = 0.10          # replaced by the measured extent

    def __init__(self, world, path: str, pos: tuple[float, float], yaw: float, seed: int, nerva_root: Path,
                 odp: Path, cache_dir: Path, physics_dt: float = 0.002, head_channel: bool = False,
                 obs_noise: bool = True, init_joint_noise: float = 0.02, limits=None):
        import onnxruntime as ort
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        from apm_express.safety import MonitorLimits, SafetyMonitor
        self.world, self.path, self.physics_dt = world, path, physics_dt
        self.ref = R = ref()
        if abs(physics_dt - R["timestep"]) > 1e-9:
            raise ValueError("NERVA runs at its validated 500 Hz physics")
        import sys
        if str(nerva_root) not in sys.path:
            sys.path.insert(0, str(nerva_root))
        from nerva.motor_contract import NeutralPhaseClock
        self._clock_cls = NeutralPhaseClock
        pol = Path(nerva_root) / R["policy_rel"]
        if hashlib.sha256(pol.read_bytes()).hexdigest() != R["policy_sha256"]:
            raise ValueError("NERVA policy hash mismatch")
        self.sess = ort.InferenceSession(str(pol), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.usd, self.mjcf, self.import_report = prepare_usd(Path(odp), Path(cache_dir))
        add_reference_to_stage(self.usd, path)
        top = world.stage.GetPrimAtPath(path)
        self.import_report.update(match_mujoco(world.stage, path, self.mjcf))
        prims = list(Usd.PrimRange(top, Usd.TraverseInstanceProxies()))
        art_root = next(str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        # geometry at the asset origin: extent ahead of the base origin (feet/head)
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(top).ComputeAlignedRange()
        self.front = float(max(bound.GetMax()[0], 0.05))
        # foot contacts from PhysX contact reports
        found = {p.GetName(): str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        self.feet = [found[n] for n in R["feet"]]  # left, right (the observation's order)
        for f in self.feet:
            PhysxSchema.PhysxContactReportAPI.Apply(world.stage.GetPrimAtPath(f)).CreateThresholdAttr(0.0)
        self._touch = {f: 0 for f in self.feet}
        # the eye on the head body
        head = next(p for p in prims if p.GetName() == "head_assembly")
        e_pos, e_rot, self.eye_base = head_eye_frame(self.mjcf, R["home_qpos"])
        # L's eye (5 cm along the face) lies inside the head shell in Isaac's render (M4 eye check): move it to 1 cm
        # beyond the head geometry's extent along the face direction (measured from the asset at rest)
        cache = UsdGeom.XformCache()
        m_h = np.array(cache.GetLocalToWorldTransform(head), float).T  # column convention
        rng_h = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            head).ComputeAlignedRange()
        lo, hi = np.array(rng_h.GetMin()), np.array(rng_h.GetMax())
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        local = (np.linalg.inv(m_h) @ corners.T).T[:, :3]
        face = e_rot[:, 0]
        self.head_extent = float(max((local @ face).max(), 0.05))
        shift = (self.head_extent + 0.01) - 0.05
        e_pos = e_pos + face * shift
        self.eye_base = self.eye_base + np.array([shift, 0.0, 0.0])  # the face points along +x at the home pose
        eye = UsdGeom.Xform.Define(world.stage, f"{head.GetPath()}/apm_eye")
        m = Gf.Matrix4d(1.0)
        m.SetRotateOnly(Gf.Matrix3d(*e_rot.T.flatten().tolist()))  # USD row-vector convention
        m.SetTranslateOnly(Gf.Vec3d(*e_pos.tolist()))
        eye.AddTransformOp().Set(m)
        self.eye_prim = str(eye.GetPath())
        self.art = world.scene.add(SingleArticulation(art_root, name=path.split("/")[-1]))
        self.start = (pos, yaw)
        self.head_channel, self.obs_noise, self.init_joint_noise = head_channel, obs_noise, init_joint_noise
        self.monitor = SafetyMonitor(limits or MonitorLimits())
        self.decim = int(R["decimation"])
        self._seed = seed
        self._subscribe_contacts()
        self._init_common(seed)

    def _subscribe_contacts(self) -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        feet = set(self.feet)

        def on_contact(headers, data):
            for h in headers:
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                for f in feet & {a0, a1}:
                    other = a1 if f == a0 else a0
                    if other.startswith(self.path):
                        continue  # self-contact
                    kind = str(h.type)
                    if "LOST" in kind:
                        self._touch[f] = max(0, self._touch[f] - 1)
                    elif "FOUND" in kind:
                        self._touch[f] += 1
        self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

    # ── reset ────────────────────────────────────────────────────────────────
    def post_reset(self) -> None:
        R = self.ref
        names = list(self.art.dof_names)
        self.idx = np.array([names.index(x["joint"]) for x in R["actuators"]])
        self.default = np.array([x["home_ctrl"] for x in R["actuators"]])
        kp, kd, fmax = np.zeros(len(names)), np.zeros(len(names)), np.full(len(names), 1e3)
        for i, x in zip(self.idx, R["actuators"], strict=True):
            kp[i], kd[i], fmax[i] = x["kp"], x["damping"], abs(x["forcerange"][1])
        av = self.art._articulation_view
        av.set_gains(kps=kp[None], kds=kd[None])
        av.set_max_efforts(fmax[None])
        self.joint_lo = np.array([x["range"][0] for x in R["actuators"]])
        self.joint_hi = np.array([x["range"][1] for x in R["actuators"]])
        self.force_limit = np.array([abs(x["forcerange"][1]) for x in R["actuators"]])
        rng = np.random.default_rng([self._seed, 11])
        self.noise_rng = np.random.default_rng([self._seed, 12])
        q = np.zeros(len(names))
        q[self.idx] = self.default + rng.uniform(-self.init_joint_noise, self.init_joint_noise, len(self.idx))
        self.art.set_joint_positions(q)
        self.art.set_joint_velocities(np.zeros(len(names)))
        (x, y), yaw = self.start
        self.art.set_world_pose(position=np.array([x, y, R["home_qpos"][2]]), orientation=yaw_quat(yaw))
        self.clock = self._clock_cls(R["phase_period"])
        self.phase = self.clock.reset()
        self.last = [np.zeros(14)] * 3
        self.targets = self.default.copy()       # last applied motor targets (with head offsets)
        self.prev_targets = self.default.copy()  # rate-limit reference (without head offsets)
        self.command: np.ndarray = np.zeros(7)
        self.head_offset = np.zeros(4)
        self.request = None
        self._was_tripped = False
        self._touch = {f: 0 for f in self.feet}  # contact reports restart after a reset
        links = list(av._metadata.link_names)
        self.imu_link = links.index(R["imu_site_body"]) if R["imu_site_body"] in links else 0
        self.imu_r = np.asarray(R["imu_site_pos"], float)
        self._v_site_prev: np.ndarray | None = None
        self.eye_offset = float(self.eye_base[0])
        self.measured = {"usd": self.usd, "import": self.import_report, "front": round(self.front, 3),
                         "head_extent_along_face": round(self.head_extent, 3),
                         "eye_offset": round(self.eye_offset, 3),
                         "eye_height": round(float(self.eye_base[2] + R["home_qpos"][2]), 3),
                         "feet": self.feet, "mass_kg": round(float(np.asarray(av.get_body_masses())[0].sum()), 3)}

    def teleport(self, x: float, y: float, yaw: float) -> None:
        super().teleport(x, y, yaw)
        self._v_site_prev = None

    # ── sensing ──────────────────────────────────────────────────────────────
    def _site_velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(site linear velocity in world, body angular velocity in world, body rotation) at NERVA's IMU site."""
        pv = self.art._articulation_view._physics_view
        vel = np.asarray(pv.get_link_velocities())[0][self.imu_link]
        tf = np.asarray(pv.get_link_transforms())[0][self.imu_link]
        Rb = quat_to_rot(np.array([tf[6], tf[3], tf[4], tf[5]]))
        w = vel[3:6]
        return vel[:3] + np.cross(w, Rb @ self.imu_r), w, Rb

    def _imu(self) -> tuple[np.ndarray, np.ndarray]:
        v, w, Rb = self._site_velocity()
        a_w = np.zeros(3) if self._v_site_prev is None else (v - self._v_site_prev) / self.physics_dt
        acc = Rb.T @ (a_w + np.array([0.0, 0.0, 9.81]))
        return acc, Rb.T @ w

    def contacts(self) -> list[float]:
        return [float(self._touch[f] > 0) for f in self.feet]

    def physical(self) -> dict:
        jp = np.asarray(self.art.get_joint_positions(), float)[self.idx]
        try:
            tau = np.asarray(self.art.get_measured_joint_efforts(), float)[self.idx]
        except Exception:  # noqa: BLE001
            tau = np.zeros(len(self.idx))
        return {"tilt_deg": math.degrees(self.tilt()),
                "angular_speed": float(np.linalg.norm(np.asarray(self.art.get_angular_velocity(), float))),
                "joint_pos": jp, "joint_lo": self.joint_lo, "joint_hi": self.joint_hi, "actuator_force": tau,
                "force_limit": self.force_limit}

    # ── the controller interface ─────────────────────────────────────────────
    def drive(self, request) -> bool:
        """One control step's request (H's MotionRequest), through NERVA's SafetyMonitor. Returns True while the
        neutral fallback is active (a trip)."""
        from apm_express.adapter import COMMANDS, NEUTRAL_REQUEST
        tripped = bool(self.monitor.update(self.t, **self.physical()))
        if tripped:
            request = NEUTRAL_REQUEST
        if request.e_pitch != 0.0:
            raise ValueError("the neutral policy has no pitch input")
        head = np.asarray(request.head, float)
        if not self.head_channel and np.any(head != 0.0):
            raise ValueError("head channel disabled")
        step = 1.5 * DT  # NervaBiped(max_head_rate=1.5)
        self.head_offset = np.clip(head, self.head_offset - step, self.head_offset + step)
        vx, vy, wz = COMMANDS[request.command]
        self.command = np.array([vx, vy, wz, 0.0, 0.0, 0.0, 0.0])  # head slots stay zero for the neutral policy
        self.request = request
        self.v_cmd, self.w_cmd = vx, wz
        if tripped and not self._was_tripped:
            self.trips += 1
        self._was_tripped = tripped
        return tripped

    def apply(self, v: float, w: float) -> bool:
        """(v, w) bodies' interface, for scripted tests: nearest trained command."""
        from apm_express.adapter import MotionRequest
        cmd = ("forward" if v > 0.02 else "backward" if v < -0.02 else "turn_left" if w > 0.2
               else "turn_right" if w < -0.2 else "rest")
        return self.drive(MotionRequest(cmd, 0.0, (0.0, 0.0, 0.0, 0.0)))

    def advance(self) -> dict:
        from isaacsim.core.utils.types import ArticulationAction
        R = self.ref
        tick = self.clock.tick(self.command)
        self.phase, cmd = tick["phase"], np.asarray(tick["command"], float)
        acc, gyro = self._imu()
        acc = acc.copy()
        acc[0] += R["accel_x_offset"]
        jp = np.asarray(self.art.get_joint_positions(), float)[self.idx]
        jv = np.asarray(self.art.get_joint_velocities(), float)[self.idx]
        obs = np.concatenate([gyro, acc, cmd, jp - self.default, jv * R["dof_vel_scale"], self.last[0],
                              self.last[1], self.last[2], self.targets, self.contacts(), self.phase])
        if self.obs_noise:
            obs = obs + self.noise_rng.uniform(-1.0, 1.0, obs.shape) * np.asarray(R["obs_noise_scale"])
        action = self.sess.run(None, {self.inp: obs.astype(np.float32)[None]})[0][0].astype(float)
        self.last = [action, self.last[0], self.last[1]]
        targets = self.default + action * R["action_scale"]
        md = R["max_motor_velocity"] * DT
        targets = np.clip(targets, self.prev_targets - md, self.prev_targets + md)
        self.prev_targets = targets.copy()
        targets[HEAD] = targets[HEAD] + self.head_offset
        self.targets = targets
        full = np.asarray(self.art.get_joint_positions(), float).copy()
        full[self.idx] = targets
        self.art.apply_action(ArticulationAction(joint_positions=full))
        v_start = self._site_velocity()[0]
        for k in range(self.decim):
            if k == self.decim - 1:
                self._v_site_prev = self._site_velocity()[0]
            self.world.step(render=False)
        self.t += DT
        row = self._row()
        # the evidence IMU (logger): specific force averaged over the control period, an IMU's bandwidth. The
        # single-step value above (the policy's accelerometer) is dominated by PhysX contact impulses (5-8 m/s2 at
        # rest; K's NERVA IMU edges are 0.4-0.85).
        v_end, w, Rb = self._site_velocity()
        row["acc"] = Rb.T @ ((v_end - v_start) / DT + np.array([0.0, 0.0, 9.81]))
        row["contacts"] = self.contacts()
        return row

    def fallen(self, row: dict) -> bool:
        return row["tilt"] > self.TILT_FALL_DEG or not row["finite"]

