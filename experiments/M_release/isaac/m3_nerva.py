"""M3: NERVA (Open Duck Mini v2) in Isaac Sim. MJCF → USD, then NERVA's validated neutral policy in PhysX.

    $ISAAC_PY experiments/M_release/isaac/m3_nerva.py --out out/m3_nerva     (after setup_nerva_inputs.sh)

**Conversion.** Isaac Lab's `MjcfConverter` converts the backlash model NERVA validated against
(`open_duck_mini_v2_backlash.xml`, Open Duck Playground at NERVA's pinned commit) after two edits:
- its two top-level `<default>` blocks are merged (the importer reads one);
- its 10 backlash joints are removed (PhysX has one joint per link; see below).
The USD is written to `out/` for reuse.

**Matched to MuJoCo** (`configs/nerva_mujoco_ref.json`):
- every link's mass, centre of mass and principal inertia, copied from MuJoCo's compiled model;
- actuated joints: kp 17.11, damping 0.56, ±3.23 N·m (written to PhysX in SI units and read back), armature 0.027;
- the home pose;
- control: 50 Hz, physics 500 Hz.
`--diag` compares forward kinematics with MuJoCo (home pose and ±0.3 rad on each joint) and holds the home pose
without the policy.

**The policy** is NERVA's frozen neutral ONNX, its SHA-256 checked. It sees the exact 101-value observation:
- gyro;
- the raw accelerometer, with +1.3 added to x as upstream does;
- the 7-slot command;
- joint offsets;
- joint velocities × 0.05;
- the last three actions;
- motor targets;
- both foot contacts;
- the phase from NERVA's own `NeutralPhaseClock(27)`.

Targets are the default pose + 0.25 × action, clipped to 5.24 rad/s. No observation noise is added.

**Battery:** stand, walk forward (0.074 m/s), turn, stop/start, and small pushes. A fall is a tilt over 45°.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--odp", default="/work/odp")
ap.add_argument("--nerva", default="/work/NERVA")
ap.add_argument("--keep-backlash", action="store_true", help="convert the backlash joints too (broken kinematics)")
ap.add_argument("--keep-importer-axes", action="store_true", help="keep the importer's joint frames (wrong axes)")
ap.add_argument("--keep-importer-mass", action="store_true", help="keep the importer's inertials (NERVA5: 3.1 kg)")
ap.add_argument("--diag", action="store_true", help="masses, drive gains, spawn geometry and a passive hold")
ap.add_argument("--physx-lab", action="store_true", help="Isaac Lab articulation settings (see robots.py)")
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")
app.update()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, a.nerva)
from nerva.motor_contract import NeutralPhaseClock  # noqa: E402

REF = json.loads((REPO / "experiments" / "M_release" / "configs" / "nerva_mujoco_ref.json").read_text())
a.out.mkdir(parents=True, exist_ok=True)
rep: dict = {}
xml = Path(a.odp) / "playground" / "open_duck_mini_v2" / "xmls" / "open_duck_mini_v2_backlash.xml"
# The importer handles one top-level <default>; this MJCF has two. Merge the second's classes into the first
# (semantics unchanged) and write the result next to the original so mesh paths still resolve.
import xml.etree.ElementTree as ET  # noqa: E402

tree = ET.parse(xml)
tops = tree.getroot().findall("default")
for extra in tops[1:]:
    for child in list(extra):
        tops[0].append(child)
    tree.getroot().remove(extra)
# PhysX allows one joint per link. NERVA's MJCF puts two joints in each leg body (the actuated joint and a ±0.5°
# backlash joint on the same axis), which the importer turns into 2-DOF joints. The joint-frame rebuild below handles
# single revolute joints only, so the 10 backlash joints are removed: in Isaac Sim the legs have no backlash, a
# documented physics difference from the MuJoCo reference. (The kinematic error found in M3 was the importer's joint
# axes, not the backlash: see the joint-frame rebuild.)
rep["backlash_removed"] = []
if not a.keep_backlash:
    for body in tree.getroot().iter("body"):
        for jn in list(body.findall("joint")):
            if (jn.get("name") or "").endswith("_backlash"):
                body.remove(jn)
                rep["backlash_removed"].append(jn.get("name"))
print("backlash_removed", rep["backlash_removed"], flush=True)
merged = xml.with_name("open_duck_mini_v2_isaac.xml")
tree.write(merged)
rep["mjcf_top_level_defaults_merged"] = len(tops)
conv = MjcfConverter(MjcfConverterCfg(asset_path=str(merged), usd_dir=str((a.out / "usd").resolve()),
                                      usd_file_name="open_duck.usd",
                                      fix_base=False, import_sites=True, force_usd_conversion=True,
                                      make_instanceable=False))
usd = str(Path(conv.usd_path).resolve())  # absolute: the converter's configuration sublayers resolve from it
rep["usd"] = usd
print("converted", usd, flush=True)
pol_path = Path(a.nerva) / REF["policy_rel"]
assert hashlib.sha256(pol_path.read_bytes()).hexdigest() == REF["policy_sha256"], "policy hash mismatch"
sess = ort.InferenceSession(str(pol_path), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

world = World(stage_units_in_meters=1.0, physics_dt=REF["timestep"], rendering_dt=0.02)
world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
add_reference_to_stage(usd, "/World/Duck")
stage = world.stage
from pxr import Usd  # noqa: E402

prims = list(Usd.PrimRange(stage.GetPrimAtPath("/World/Duck"), Usd.TraverseInstanceProxies()))
root = next((str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)
             or "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()), None)
if root is None:
    for p in prims[:60]:
        print("prim", p.GetPath(), p.GetTypeName(), list(p.GetAppliedSchemas())[:4], flush=True)
    root = "/World/Duck"
rep["articulation_root"] = root
# drives: MuJoCo actuator gains, armature; passive backlash joints keep their damping/armature
by_name = {p.GetName(): p for p in prims if p.IsA(UsdPhysics.RevoluteJoint)}
for x in REF["actuators"]:
    p = by_name.get(x["joint"])
    if p is None:
        continue
    d = UsdPhysics.DriveAPI.Apply(p, "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(x["kp"] * math.pi / 180)   # USD angular drives are per degree
    d.CreateDampingAttr(x["damping"] * math.pi / 180)
    d.CreateMaxForceAttr(abs(x["forcerange"][1]))
    PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(x["armature"])
for x in REF["backlash"]:
    p = by_name.get(x["joint"])
    if p is not None:
        PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(x["armature"])
# Inertials: the importer's masses differ from MuJoCo's compiled model (e.g. the floating "base" body, which has no
# inertial in the MJCF, gets a default 1 kg in PhysX). Every link gets MuJoCo's compiled mass, centre of mass and
# principal inertia, so both engines simulate the same rigid bodies.
rep["inertials_from_mujoco"] = not a.keep_importer_mass
if not a.keep_importer_mass:
    import mujoco
    from pxr import Gf
    mj = mujoco.MjModel.from_xml_path(str(merged))
    mjb = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(1, mj.nbody)}
    rep["mujoco_total_mass"] = round(float(mj.body_mass[1:].sum()), 4)
    changed = []
    for p in prims:
        if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetName() not in mjb:
            continue
        i = mjb[p.GetName()]
        m = UsdPhysics.MassAPI.Apply(p)
        m.CreateMassAttr(max(float(mj.body_mass[i]), 1e-3))
        m.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in mj.body_ipos[i])))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*(max(float(v), 1e-7) for v in mj.body_inertia[i])))
        w, x, y, z = (float(v) for v in mj.body_iquat[i])
        m.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z))
        changed.append((p.GetName(), round(float(mj.body_mass[i]), 4)))
    rep["inertials"] = changed
    print("mujoco_total_mass", rep["mujoco_total_mass"], "links set", changed, flush=True)
# Joint frames: the importer rotates several NERVA joints about the wrong axis (M3 FK check: joints reach their
# targets, yet +0.5 and -0.5 rad give the same link error, i.e. a perpendicular axis). Every revolute joint's frames
# are rebuilt from MuJoCo's compiled model: joint position and axis in the child body, the same frame in the parent
# at the zero configuration, USD axis X along MuJoCo's axis, and MuJoCo's limits (degrees in USD).
rep["joint_frames_from_mujoco"] = not a.keep_importer_axes
if not a.keep_importer_axes:
    import mujoco
    from pxr import Gf

    mjj = mujoco.MjModel.from_xml_path(str(merged))

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def quat_rot(q, v):
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return R @ np.asarray(v, float)

    def x_to(axis):
        u = np.asarray(axis, float) / np.linalg.norm(axis)
        c = float(np.clip(u[0], -1.0, 1.0))
        if c > 1 - 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if c < -1 + 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0])
        ax = np.cross([1.0, 0.0, 0.0], u)
        ax /= np.linalg.norm(ax)
        h = np.arccos(c) / 2
        return np.array([np.cos(h), *(np.sin(h) * ax)])

    fixed = []
    for jp in prims:
        if not jp.IsA(UsdPhysics.RevoluteJoint):
            continue
        jid = mujoco.mj_name2id(mjj, mujoco.mjtObj.mjOBJ_JOINT, jp.GetName())
        if jid < 0:
            continue
        b = int(mjj.jnt_bodyid[jid])
        par = int(mjj.body_parentid[b])
        child_name = mujoco.mj_id2name(mjj, mujoco.mjtObj.mjOBJ_BODY, b)
        parent_name = mujoco.mj_id2name(mjj, mujoco.mjtObj.mjOBJ_BODY, par)
        rj = UsdPhysics.RevoluteJoint(jp)
        b0 = [str(t) for t in rj.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in rj.GetBody1Rel().GetTargets()]
        swapped = bool(b1 and b1[0].rsplit("/", 1)[-1] == parent_name)
        qa = x_to(mjj.jnt_axis[jid])
        pos_c = np.asarray(mjj.jnt_pos[jid], float)
        pos_p = np.asarray(mjj.body_pos[b], float) + quat_rot(mjj.body_quat[b], pos_c)
        q_p = quat_mul(mjj.body_quat[b], qa)
        f_parent = (Gf.Vec3f(*pos_p.tolist()), Gf.Quatf(*q_p.tolist()))
        f_child = (Gf.Vec3f(*pos_c.tolist()), Gf.Quatf(*qa.tolist()))
        (p0, r0), (p1, r1) = (f_child, f_parent) if swapped else (f_parent, f_child)
        rj.CreateAxisAttr("X")
        rj.CreateLocalPos0Attr().Set(p0)
        rj.CreateLocalRot0Attr().Set(r0)
        rj.CreateLocalPos1Attr().Set(p1)
        rj.CreateLocalRot1Attr().Set(r1)
        lo, hi = (float(np.degrees(v)) for v in mjj.jnt_range[jid])
        if swapped:  # body order reversed: the same motion has the opposite sign
            lo, hi = -hi, -lo
        if mjj.jnt_limited[jid]:
            rj.CreateLowerLimitAttr().Set(lo)
            rj.CreateUpperLimitAttr().Set(hi)
        fixed.append({"joint": jp.GetName(), "parent": parent_name, "child": child_name, "swapped": swapped,
                      "body0": b0[0].rsplit("/", 1)[-1] if b0 else None})
    rep["joint_frames"] = fixed
    print("joint_frames_rebuilt", len(fixed), "swapped", [f["joint"] for f in fixed if f["swapped"]], flush=True)
if a.physx_lab:
    sys.path.insert(0, str(REPO / "src"))
    from apm_isaac.robots import physx_like_isaac_lab
    physx_like_isaac_lab(world.stage, "/World/Duck")
robot = world.scene.add(SingleArticulation(root, name="duck"))
world.reset()
names = list(robot.dof_names)
rep["isaac_dofs"] = names
act = [x["joint"] for x in REF["actuators"]]
rep["missing"] = [j for j in act if j not in names]
print("dofs", len(names), "missing", rep["missing"], flush=True)
if rep["missing"]:
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    app.close()
    sys.exit(0)
idx = np.array([names.index(j) for j in act])
default = np.array([x["home_ctrl"] for x in REF["actuators"]])
lnames = list(robot._articulation_view._metadata.link_names)
feet = [lnames.index(f) for f in REF["feet"] if f in lnames]
rep["feet_links"] = [lnames[i] for i in feet]


def set_gains() -> None:
    """MuJoCo's gains in SI units, written to PhysX directly after each reset (USD drive attributes from the
    importer turned out to be per-degree copies of MuJoCo's per-radian kp on the legs). Read back for the report."""
    kp, kd, fmax = np.zeros(len(names)), np.zeros(len(names)), np.full(len(names), 1e3)
    for x in REF["actuators"]:
        i = names.index(x["joint"])
        kp[i], kd[i], fmax[i] = x["kp"], x["damping"], abs(x["forcerange"][1])
    for x in REF["backlash"]:
        if x["joint"] in names:
            kd[names.index(x["joint"])] = x.get("damping", 0.0)
    av = robot._articulation_view
    av.set_gains(kps=kp[None], kds=kd[None])
    av.set_max_efforts(fmax[None])
    pv = av._physics_view
    rep["gains_readback"] = {"kp": np.round(np.asarray(pv.get_dof_stiffnesses())[0], 3).tolist(),
                             "kd": np.round(np.asarray(pv.get_dof_dampings())[0], 3).tolist()}


def fk_check() -> None:
    """Link positions relative to the base: PhysX vs MuJoCo forward kinematics, at the home pose and with each
    actuated joint perturbed by +0.3 rad in turn (finds joints whose axis or zero differs). PhysX link poses update
    only on a step, so each pose is written with gravity off and read after one step."""
    import mujoco
    mj = mujoco.MjModel.from_xml_path(str(merged))
    md = mujoco.MjData(mj)
    hq = REF["home_qpos"]
    world.get_physics_context().set_gravity(0.0)
    base = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, lnames[0])

    def compare(q_act: np.ndarray, q_mj: np.ndarray | None = None) -> tuple[float, list]:
        md.qpos[:] = 0.0
        md.qpos[:7] = hq[:7]
        for x, v in zip(REF["actuators"], q_act if q_mj is None else q_mj, strict=True):
            md.qpos[mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, x["joint"])]] = v
        mujoco.mj_kinematics(mj, md)
        world.reset()
        q = np.zeros(len(names))
        q[idx] = q_act
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(len(names)))
        robot.set_world_pose(position=np.array([0.0, 0.0, 1.0]), orientation=np.array(hq[3:7]))
        full = q.copy()
        for _ in range(100):  # gravity is off: the PD drives settle on the targets in 0.2 s
            robot.apply_action(ArticulationAction(joint_positions=full))
            robot.set_linear_velocity(np.zeros(3))
            robot.set_angular_velocity(np.zeros(3))
            world.step(render=False)
        q_read = np.asarray(robot.get_joint_positions(), float)[idx]
        compare.q_read = q_read  # type: ignore[attr-defined]
        md.qpos[:] = 0.0  # MuJoCo at the joint angles PhysX actually reached
        md.qpos[:7] = hq[:7]
        for x, v in zip(REF["actuators"], q_read if q_mj is None else -q_read, strict=True):
            md.qpos[mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, x["joint"])]] = v
        mujoco.mj_kinematics(mj, md)
        R0 = rot(np.asarray(robot.get_world_pose()[1]))
        tf = np.asarray(robot._articulation_view._physics_view.get_link_transforms())[0]
        rows = []
        for i, n in enumerate(lnames):
            jb = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n)
            if jb < 0:
                continue
            d_isaac = R0.T @ (tf[i, :3] - tf[0, :3])
            d_mj = md.xmat[base].reshape(3, 3).T @ (md.xpos[jb] - md.xpos[base])
            R_i = R0.T @ rot(tf[i, [6, 3, 4, 5]])  # PhysX link quaternion is (x, y, z, w)
            R_m = md.xmat[base].reshape(3, 3).T @ md.xmat[jb].reshape(3, 3)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(R_i.T @ R_m) - 1) / 2, -1.0, 1.0))))
            rows.append((n, np.round(d_isaac, 4).tolist(), np.round(d_mj, 4).tolist(),
                         round(float(np.linalg.norm(d_isaac - d_mj)), 4), round(ang, 2)))
        compare.ang = max(r[4] for r in rows)  # type: ignore[attr-defined]
        return max(r[3] for r in rows), rows

    err, rows = compare(default)
    rep["fk_home"] = rows
    print("fk_home_max_err_m", err, "max_orientation_err_deg", compare.ang, flush=True)
    for r in rows:
        if r[3] > 0.002 or r[4] > 1.0:
            print("  fk", r, flush=True)
    err0, _ = compare(np.zeros(len(idx)))
    print("fk_zero_max_err_m", err0, flush=True)
    rep["fk_joint"] = {}
    for k, x in enumerate(REF["actuators"]):  # one joint at a time from the zero pose (which matches exactly)
        qi = np.zeros(len(idx))
        qi[k] = 0.5
        e_same, rr = compare(qi)
        e_flip, _ = compare(qi, -qi)
        rep["fk_joint"][x["joint"]] = {"same_sign": e_same, "opposite_sign": e_flip}
        print("fk_joint", x["joint"], "isaac +0.5 vs mujoco +0.5", e_same, "vs mujoco -0.5", e_flip,
              "orient_err_deg", compare.ang, "worst", max(rr, key=lambda r: r[3])[:1],
              "q_read", np.round(compare.q_read, 3).tolist(), flush=True)
    world.get_physics_context().set_gravity(-9.81)



def rot(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def diagnose() -> None:
    """Why does the policy fall? Physical model as PhysX sees it, then the home pose held without the policy."""
    fk_check()
    world.reset()
    set_gains()
    pv = robot._articulation_view._physics_view
    q0 = np.zeros(len(names))
    q0[idx] = default
    robot.set_joint_positions(q0)
    hq = REF["home_qpos"]
    robot.set_world_pose(position=np.array(hq[:3]), orientation=np.array(hq[3:7]))
    robot.set_joint_velocities(np.zeros(len(names)))
    world.step(render=False)
    try:
        masses = np.asarray(robot._articulation_view.get_body_masses())[0]
        print("total_mass", round(float(masses.sum()), 3), "links", len(lnames), flush=True)
        for n, m in zip(lnames, masses, strict=False):
            print("  link", n, round(float(m), 4), flush=True)
    except Exception as e:  # noqa: BLE001
        print("masses n/a", repr(e)[:120], flush=True)
    for k, nm in (("stiff", "get_dof_stiffnesses"), ("damp", "get_dof_dampings"), ("maxf", "get_dof_max_forces"),
                  ("arm", "get_dof_armatures"), ("limits", "get_dof_limits")):
        try:
            v = np.asarray(getattr(pv, nm)())[0]  # CPU pipeline
            print("dof", k, [(names[i], np.round(v[i], 4).tolist()) for i in range(len(names))], flush=True)
        except Exception as e:  # noqa: BLE001
            print("dof", k, "n/a", repr(e)[:100], flush=True)
    tf = np.asarray(pv.get_link_transforms())[0]
    p, q = robot.get_world_pose()
    print("root_pose", np.round(p, 4).tolist(), np.round(q, 4).tolist(), "home_qpos", hq[:7], flush=True)
    print("link_z", [(n, round(float(tf[i, 2]), 4)) for i, n in enumerate(lnames)], flush=True)
    print("joint_pos_after_set", np.round(np.asarray(robot.get_joint_positions())[idx] - default, 4).tolist(),
          flush=True)
    for label, hold in (("hold_default", True), ("limp", False)):
        world.reset()
        set_gains()
        pv = robot._articulation_view._physics_view
        robot.set_joint_positions(q0)
        robot.set_world_pose(position=np.array(hq[:3]), orientation=np.array(hq[3:7]))
        for k in range(60):
            full = np.asarray(robot.get_joint_positions(), float).copy()
            if hold:
                full[idx] = default
                robot.apply_action(ArticulationAction(joint_positions=full))
            else:
                robot.apply_action(ArticulationAction(joint_efforts=np.zeros(len(names))))
            for _k in range(REF["decimation"]):
                world.step(render=False)
            if k % 3 == 0 or k < 6:
                p, q = robot.get_world_pose()
                R = rot(q)
                tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
                tf = np.asarray(pv.get_link_transforms())[0]
                jp = np.asarray(robot.get_joint_positions(), float)
                print(label, f"t={0.02 * (k + 1):.2f} z={p[2]:.4f} tilt={tilt:.1f} feet_z",
                      [round(float(tf[i, 2]), 4) for i in feet], "q_err_max",
                      round(float(np.abs(jp[idx] - default).max()), 3), "backlash",
                      np.round(np.delete(jp, idx), 3).tolist(), flush=True)


def run(name: str, script: list, pushes: list | None = None) -> dict:
    world.reset()
    set_gains()
    q0 = np.zeros(len(names))
    q0[idx] = default
    robot.set_joint_positions(q0)
    hq = REF["home_qpos"]
    robot.set_world_pose(position=np.array(hq[:3]), orientation=np.array(hq[3:7]))
    clock = NeutralPhaseClock(REF["phase_period"])
    phase = clock.reset()
    last = [np.zeros(14)] * 3
    targets = default.copy()
    prev_targets = default.copy()
    v_prev = None
    t, fell, tilts, track = 0.0, False, [], []
    for cmd3, dur in script:
        for _ in range(int(dur / 0.02)):
            command = np.array([*cmd3, 0.0, 0.0, 0.0, 0.0])
            tick = clock.tick(command)
            phase, command = tick["phase"], tick["command"]
            p, q = robot.get_world_pose()
            R = rot(q)
            v = np.asarray(robot.get_linear_velocity(), float)
            a_w = np.zeros(3) if v_prev is None else (v - v_prev) / 0.02
            v_prev = v
            gyro = R.T @ np.asarray(robot.get_angular_velocity(), float)
            acc = R.T @ (a_w + np.array([0, 0, 9.81]))
            acc[0] += REF["accel_x_offset"]
            jp = np.asarray(robot.get_joint_positions(), float)[idx]
            jv = np.asarray(robot.get_joint_velocities(), float)[idx]
            tf = np.asarray(robot._articulation_view._physics_view.get_link_transforms())[0]
            contacts = [float(tf[i, 2] < 0.035) for i in feet] or [1.0, 1.0]
            obs = np.concatenate([gyro, acc, command, jp - default, jv * REF["dof_vel_scale"], last[0], last[1],
                                  last[2], targets, contacts, phase]).astype(np.float32)
            action = sess.run(None, {inp: obs[None]})[0][0]
            last = [action, last[0], last[1]]
            targets = default + action * REF["action_scale"]
            md = REF["max_motor_velocity"] * 0.02
            targets = np.clip(targets, prev_targets - md, prev_targets + md)
            prev_targets = targets.copy()
            full = np.asarray(robot.get_joint_positions(), float).copy()
            full[idx] = targets
            robot.apply_action(ArticulationAction(joint_positions=full))
            for pu in pushes or []:
                if abs(t - pu[0]) < 0.01:
                    robot.set_linear_velocity(v + np.array([pu[1], pu[2], 0.0]))
            for _k in range(REF["decimation"]):
                world.step(render=False)
            t += 0.02
            tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
            tilts.append(tilt)
            if tilt > 45:
                fell = True
                break
            if t > 2 and any(cmd3):
                yaw = math.atan2(R[1, 0], R[0, 0])
                track.append(v[0] * math.cos(yaw) + v[1] * math.sin(yaw) - cmd3[0])
        if fell:
            break
    res = {"fell": fell, "t": round(t, 2), "max_tilt": round(max(tilts), 1),
           "v_err": round(float(np.mean(track)), 4) if track else None}
    print(name, res, flush=True)
    return res


if a.diag:
    diagnose()
    app.close()
    sys.exit(0)
rep["battery"] = {
    "stand": run("stand", [((0.0, 0.0, 0.0), 6.0)]),
    "walk": run("walk", [((0.0, 0.0, 0.0), 1.0), ((0.074, 0.0, 0.0), 10.0)]),
    "turn": run("turn", [((0.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 0.6), 8.0)]),
    "stop_start": run("stop_start", [((0.074, 0, 0), 3.0), ((0, 0, 0), 2.0), ((0.074, 0, 0), 3.0), ((0, 0, 0), 2.0)]),
    "push_0.05": run("push_0.05", [((0.074, 0.0, 0.0), 10.0)], pushes=[(3.0, 0.05, 0.0), (6.0, 0.0, 0.05)]),
}
(a.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
app.close()
