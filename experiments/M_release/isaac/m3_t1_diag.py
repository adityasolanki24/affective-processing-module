"""M3 diagnostic: why does the T1 fall within 0.6 s in Isaac even when standing still?

    $ISAAC_PY experiments/M_release/isaac/m3_t1_diag.py --out out/t1diag

It writes:
- the USD's own articulation facts: root path, link masses, and each joint's drive type, stiffness, damping,
  max force and limits;
- a hold test with no policy: PD targets at the MuJoCo home pose for 3 s, under four gain settings: the USD's own
  gains, the MuJoCo gains set through `set_gains`, the MuJoCo gains × 180/π (a unit check), and the MuJoCo gains
  written straight onto the USD drives. It reports base height, tilt and joint tracking error over time;
- the base height after reset compared with the MuJoCo home height, and the root link's frame offset.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
launcher = AppLauncher({"headless": True})
app = launcher.app

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REF = json.loads((REPO / "experiments" / "M_release" / "configs" / "t1_mujoco_ref.json").read_text())
usd = get_assets_root_path() + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd"
a.out.mkdir(parents=True, exist_ok=True)
rep: dict = {}
world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=0.02)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd, "/World/T1")
stage = world.stage
root = next(str(p.GetPath()) for p in stage.Traverse()
            if str(p.GetPath()).startswith("/World/T1") and p.HasAPI(UsdPhysics.ArticulationRootAPI))
rep["articulation_root"] = root
drives = {}
masses = {}
for p in stage.Traverse():
    if not str(p.GetPath()).startswith("/World/T1"):
        continue
    if p.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(p).GetMassAttr().Get()
        if m:
            masses[p.GetName()] = float(m)
    if p.IsA(UsdPhysics.RevoluteJoint):
        d = UsdPhysics.DriveAPI.Get(p, "angular")
        lo = UsdPhysics.RevoluteJoint(p).GetLowerLimitAttr().Get()
        hi = UsdPhysics.RevoluteJoint(p).GetUpperLimitAttr().Get()
        drives[p.GetName()] = {"type": d.GetTypeAttr().Get() if d else None,
                               "stiffness": d.GetStiffnessAttr().Get() if d else None,
                               "damping": d.GetDampingAttr().Get() if d else None,
                               "max_force": d.GetMaxForceAttr().Get() if d else None, "limits_deg": [lo, hi]}
rep["masses_total"] = sum(masses.values())
rep["n_mass_prims"] = len(masses)
rep["drives"] = drives
robot = world.scene.add(SingleArticulation(root, name="t1"))
world.reset()
names = list(robot.dof_names)
idx = np.array([names.index(x["joint"]) for x in REF["actuators"]])
home = np.array([x["home"] for x in REF["actuators"]])
kp = np.array([x["kp"] for x in REF["actuators"]])
kd = np.array([x["damping"] for x in REF["actuators"]])
rep["isaac_default_gains"] = {k: v.tolist() for k, v in zip(("kps", "kds"),
                                                            robot.get_articulation_controller().get_gains(),
                                                            strict=False)}


def hold(label: str, gains: str) -> dict:
    world.reset()
    ctrl = robot.get_articulation_controller()
    n = len(names)
    if gains != "usd":
        k = np.zeros(n)
        d = np.zeros(n)
        scale = 180 / math.pi if gains == "scaled" else 1.0
        k[idx], d[idx] = kp * scale, kd * scale
        ctrl.set_gains(kps=k, kds=d)
    q = np.zeros(n)
    q[idx] = home
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros(n))
    robot.set_world_pose(position=np.array(REF["home_base_pos"]), orientation=np.array(REF["home_base_quat"]))
    z0 = float(robot.get_world_pose()[0][2])
    traj = []
    for i in range(1500):
        robot.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=False)
        if i % 100 == 0:
            p, qq = robot.get_world_pose()
            w, x, y, z = qq
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
            err = float(np.abs(np.asarray(robot.get_joint_positions())[idx] - home).max())
            traj.append({"t": round(i * 0.002, 2), "z": round(float(p[2]), 3), "tilt": round(tilt, 1),
                         "max_joint_err": round(err, 3)})
    res = {"z_after_reset": z0, "traj": traj}
    print(label, res, flush=True)
    return res


from pxr import PhysxSchema, Usd, UsdGeom  # noqa: E402


def feet_lowest() -> float:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    lows = []
    for p in stage.Traverse():
        n = str(p.GetPath())
        if n.startswith("/World/T1") and ("foot" in n.lower() or "ankle" in n.lower()):
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty():
                lows.append(r.GetMin()[2])
    return float(min(lows)) if lows else float("nan")

world.reset()
rep["feet_lowest_z_at_reset_pose"] = feet_lowest()
print("feet lowest z", rep["feet_lowest_z_at_reset_pose"], flush=True)
art = PhysxSchema.PhysxArticulationAPI.Get(stage, root)
rep["self_collisions_enabled"] = art.GetEnabledSelfCollisionsAttr().Get() if art else None
print("self collisions", rep["self_collisions_enabled"], flush=True)
rep["hold_mujoco_gains"] = hold("mujoco", "mujoco")
REF["home_base_pos"][2] += 0.05
rep["hold_raised_5cm"] = hold("raised", "mujoco")
if art:
    art.CreateEnabledSelfCollisionsAttr(False)
rep["hold_raised_no_selfcoll"] = hold("raised_noself", "mujoco")
REF["home_base_pos"][2] -= 0.05
rep["hold_noself"] = hold("noself", "mujoco")
# links: where is the root link relative to the feet? (frame offset vs MuJoCo's Trunk)
world.reset()
rep["root_pose_after_reset"] = [np.asarray(v).tolist() for v in robot.get_world_pose()]
(a.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps({k: v for k, v in rep.items() if k not in ("drives",)}, default=str)[:3000], flush=True)
print("drives sample", list(drives.items())[:6], flush=True)
app.close()
sys.exit(0)
