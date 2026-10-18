"""M3: NERVA (Open Duck Mini v2) in Isaac Sim. MJCF → USD, then NERVA's validated neutral policy in PhysX.

    $ISAAC_PY experiments/M_release/isaac/m3_nerva.py --out out/m3_nerva     (after setup_nerva_inputs.sh)

**Conversion.** Isaac Lab's `MjcfConverter` converts the backlash model NERVA validated against
(`open_duck_mini_v2_backlash.xml`, Open Duck Playground at NERVA's pinned commit). The USD is written to `out/` for
reuse.

**Matched to MuJoCo** (`configs/nerva_mujoco_ref.json`):
- actuated joints: kp 17.11, damping 0.56, armature 0.027, ±3.23 N·m;
- the 10 passive backlash joints: their damping and armature;
- the home pose;
- control: 50 Hz, physics 500 Hz.

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
conv = MjcfConverter(MjcfConverterCfg(asset_path=str(xml), usd_dir=str(a.out / "usd"), usd_file_name="open_duck.usd",
                                      fix_base=False, import_sites=True, force_usd_conversion=True,
                                      make_instanceable=False))
usd = conv.usd_path
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
root = next((str(p.GetPath()) for p in stage.Traverse()
             if str(p.GetPath()).startswith("/World/Duck") and p.HasAPI(UsdPhysics.ArticulationRootAPI)), "/World/Duck")
rep["articulation_root"] = root
# drives: MuJoCo actuator gains, armature; passive backlash joints keep their damping/armature
by_name = {p.GetName(): p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/Duck")
           and p.IsA(UsdPhysics.RevoluteJoint)}
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


def rot(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def run(name: str, script: list, pushes: list | None = None) -> dict:
    world.reset()
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


rep["battery"] = {
    "stand": run("stand", [((0.0, 0.0, 0.0), 6.0)]),
    "walk": run("walk", [((0.0, 0.0, 0.0), 1.0), ((0.074, 0.0, 0.0), 10.0)]),
    "turn": run("turn", [((0.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 0.6), 8.0)]),
    "stop_start": run("stop_start", [((0.074, 0, 0), 3.0), ((0, 0, 0), 2.0), ((0.074, 0, 0), 3.0), ((0, 0, 0), 2.0)]),
    "push_0.05": run("push_0.05", [((0.074, 0.0, 0.0), 10.0)], pushes=[(3.0, 0.05, 0.0), (6.0, 0.0, 0.05)]),
}
(a.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
app.close()
