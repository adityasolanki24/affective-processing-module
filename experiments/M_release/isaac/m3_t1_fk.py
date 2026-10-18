"""M3 diagnostic: does the Isaac T1 USD share the MuJoCo model's joint conventions?

    $ISAAC_PY experiments/M_release/isaac/m3_t1_fk.py --out out/t1fk

With gravity off and the root fixed in place, it sets the zero pose, the home pose and each single-joint +0.3 rad
perturbation. It reads every link's position relative to the trunk and compares it with MuJoCo's forward kinematics
(`configs/t1_mujoco_ref.json` → `fk_rel_trunk`).

Per joint:
- `sign`: the sign of dot(Isaac link displacement, MuJoCo link displacement), summed over links;
- `ratio`: |Isaac| / |MuJoCo| displacement.

It also reports the zero-pose and home-pose link-position errors, so a frame or offset difference shows up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True}).app

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REF = json.loads((REPO / "experiments" / "M_release" / "configs" / "t1_mujoco_ref.json").read_text())
FK = REF["fk_rel_trunk"]
a.out.mkdir(parents=True, exist_ok=True)
world = World(stage_units_in_meters=1.0, physics_dt=0.002)
world.get_physics_context().set_gravity(0.0)
add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd",
                       "/World/T1")
stage = world.stage
root = next(str(p.GetPath()) for p in stage.Traverse()
            if str(p.GetPath()).startswith("/World/T1") and p.HasAPI(UsdPhysics.ArticulationRootAPI))
robot = world.scene.add(SingleArticulation(root, name="t1"))
link_paths = {}
for p in stage.Traverse():
    n = p.GetName()
    if (str(p.GetPath()).startswith("/World/T1") and p.HasAPI(UsdPhysics.RigidBodyAPI)
            and n in FK["zero"] | {"Trunk": 0}):
        link_paths[n] = str(p.GetPath())
world.reset()
names = list(robot.dof_names)
idx = {x["joint"]: names.index(x["joint"]) for x in REF["actuators"]}


def rel(q_by_joint: dict) -> dict:
    q = np.zeros(len(names))
    for j, v in q_by_joint.items():
        q[idx[j]] = v
    for _ in range(3):
        robot.set_world_pose(position=np.array([0, 0, 1.5]), orientation=np.array([1.0, 0, 0, 0]))
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(len(names)))
        world.step(render=False)
    view = robot._articulation_view._physics_view  # PhysX tensor view: the simulated link poses
    tf = np.asarray(view.get_link_transforms())[0]
    lnames = list(robot._articulation_view._metadata.link_names)
    by = {n: tf[i, :3] for i, n in enumerate(lnames) if n in link_paths}
    p0 = by["Trunk"]
    return {k: (v - p0) for k, v in by.items() if k != "Trunk"}


out: dict = {"links_matched": sorted(link_paths)}
zero = rel({})
home = rel({x["joint"]: x["home"] for x in REF["actuators"]})
out["zero_err_max"] = float(max(np.linalg.norm(zero[k] - np.array(FK["zero"][k])) for k in zero))
out["home_err_max"] = float(max(np.linalg.norm(home[k] - np.array(FK["home"][k])) for k in home))
out["zero_err_by_link"] = {k: round(float(np.linalg.norm(zero[k] - np.array(FK["zero"][k]))), 4) for k in zero}
out["joints"] = {}
for x in REF["actuators"]:
    j = x["joint"]
    pert = rel({j: 0.3})
    num = den_i = den_m = 0.0
    for k in pert:
        di = pert[k] - zero[k]
        dm = np.array(FK["perturb"][j][k]) - np.array(FK["zero"][k])
        num += float(di @ dm)
        den_i += float(di @ di)
        den_m += float(dm @ dm)
    out["joints"][j] = {"sign": int(np.sign(num)) if abs(num) > 1e-8 else 0,
                        "ratio": round(float(np.sqrt(den_i / max(den_m, 1e-12))), 3),
                        "cos": round(num / max(np.sqrt(den_i * den_m), 1e-12), 3)}
print(json.dumps({k: v for k, v in out.items() if k != "zero_err_by_link"}, indent=0), flush=True)
(a.out / "fk.json").write_text(json.dumps(out, indent=1))
app.close()
