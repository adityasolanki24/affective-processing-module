"""M3 diagnostic: drop tests for the T1 holding its home pose (no policy).

Variants:
- `drop30`: spawned 0.30 m higher;
- `drop30_noself`: also with articulation self-collision disabled before the first reset;
- `noself`: at the MuJoCo home height, self-collision disabled.

Every 0.1 s it reports base height, tilt and the lowest foot-link height, plus the resting height once settled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True}).app

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REF = json.loads((REPO / "experiments" / "M_release" / "configs" / "t1_mujoco_ref.json").read_text())
world = World(stage_units_in_meters=1.0, physics_dt=0.002)
world.scene.add_default_ground_plane()
add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd",
                       "/World/T1")
stage = world.stage
root = next(str(p.GetPath()) for p in stage.Traverse()
            if str(p.GetPath()).startswith("/World/T1") and p.HasAPI(UsdPhysics.ArticulationRootAPI))
if "noself" in a.variant:
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root)).CreateEnabledSelfCollisionsAttr(False)
robot = world.scene.add(SingleArticulation(root, name="t1"))
world.reset()
names = list(robot.dof_names)
idx = np.array([names.index(x["joint"]) for x in REF["actuators"]])
k = np.zeros(len(names))
d = np.zeros(len(names))
k[idx] = [x["kp"] for x in REF["actuators"]]
d[idx] = [x["damping"] for x in REF["actuators"]]
robot.get_articulation_controller().set_gains(kps=k, kds=d)
q = np.zeros(len(names))
q[idx] = [x["home"] for x in REF["actuators"]]
z0 = REF["home_base_pos"][2] + (0.30 if "drop30" in a.variant else 0.0)
robot.set_world_pose(position=np.array([0, 0, z0]), orientation=np.array([1.0, 0, 0, 0]))
robot.set_joint_positions(q)
lnames = list(robot._articulation_view._metadata.link_names)
feet = [i for i, n in enumerate(lnames) if "foot" in n.lower()]
traj = []
for i in range(2000):
    robot.apply_action(ArticulationAction(joint_positions=q))
    world.step(render=False)
    if i % 50 == 0:
        p, qq = robot.get_world_pose()
        w, x, y, z = qq
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        tf = np.asarray(robot._articulation_view._physics_view.get_link_transforms())[0]
        traj.append((round(i * 0.002, 2), round(float(p[2]), 3), round(tilt, 1), round(float(tf[feet, 2].min()), 3)))
print(a.variant, "feet links", [lnames[i] for i in feet], flush=True)
print(a.variant, " ".join(f"t{t}:z{z}/tilt{ti}/foot{fz}" for t, z, ti, fz in traj[:20]), flush=True)
print(a.variant, "final", traj[-1], flush=True)
app.close()
