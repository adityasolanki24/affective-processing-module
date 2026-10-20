"""Nova Carter contact check: the asset's real collision extent and what a person's lunge does to it.

    $ISAAC_PY experiments/M_release/isaac/carter_contact_check.py --out out/carter_contact [--seeds 6]

1. Geometry: the bounding range of every collision prim in the chassis frame, the whole asset's bound for each
   purpose, and the `front` the world uses (NovaCarterIsaac.front).
2. Probe: a person's kinematic capsule is moved toward the standing robot 2 mm per control step, from straight
   ahead and from 30 and 60 degrees off its heading; the distance at which the robot first moves is its real
   extent in that direction.
3. Encounters: the robot stands where an encounter's approach stops it (radial gap 0.10 m by `front`) and a
   person lunges as in `world.encounter` (agents.Person, "pushy" and "gentle" types) for 30 s; the robot's peak
   base speed, physical contact reports and any ejection (> 3 m/s with no command) are recorded per seed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--front", type=float, default=None, help="override the robot's front extent (fix check)")
ap.add_argument("--footprint", action="store_true", help="lunge limits from the collision footprint (as world.step)")
a = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.storage.native")

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from apm_isaac import exit_on_error, hard_exit  # noqa: E402
from apm_isaac.agents import RADIUS, Person, add_person, place  # noqa: E402
from apm_isaac.robots import DT, NovaCarterIsaac  # noqa: E402

exit_on_error()
a.out.mkdir(parents=True, exist_ok=True)
world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=DT)
world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
robot = NovaCarterIsaac(world, "/World/Robot", (0.0, 0.0), 0.0, 1, 0.002)
person = add_person(world.stage, "P", (5.0, 0.0), None)
# physical contact truth, as in IsaacWorld._subscribe_contacts: PhysX contact reports person <-> robot
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema  # noqa: E402

PhysxSchema.PhysxContactReportAPI.Apply(world.stage.GetPrimAtPath(person.cap_path)).CreateThresholdAttr(0.0)
touches = [0]


def on_contact(headers, data):
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        if "/World/People/" in a0 + a1 and "/World/Robot" in a0 + a1:
            touches[0] += 1


_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
world.reset()
robot.post_reset()
if a.front is not None:
    robot.front = a.front
for _ in range(25):
    robot.apply(0.0, 0.0)
    robot.advance()

# ── 1. geometry ──
stage = world.stage
top = stage.GetPrimAtPath("/World/Robot")
prims = list(Usd.PrimRange(top, Usd.TraverseInstanceProxies()))
cache = UsdGeom.XformCache()
chassis = next(p for p in prims if p.GetName() == robot.chassis_name)
c_inv = cache.GetLocalToWorldTransform(chassis).GetInverse()
geo: dict = {"front_used": robot.front, "chassis": robot.chassis_name, "colliders": [], "purpose_bounds": {}}
for purposes in (["default"], ["default", "render"], ["default", "proxy"], ["default", "guide"]):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes).ComputeWorldBound(top).ComputeAlignedRange()
    geo["purpose_bounds"]["+".join(purposes)] = [list(map(float, r.GetMin())), list(map(float, r.GetMax()))]
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
for p in prims:
    if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Imageable):
        b = bbc.ComputeUntransformedBound(p)
        m = cache.GetLocalToWorldTransform(p) * c_inv
        rng = b.ComputeAlignedRange()
        corners = [m.Transform(rng.GetCorner(i)) for i in range(8)] if not rng.IsEmpty() else []
        if corners:
            lo = np.min(np.array(corners), axis=0)
            hi = np.max(np.array(corners), axis=0)
            geo["colliders"].append({"prim": str(p.GetPath()).replace("/World/Robot", ""),
                                     "lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist()})
if geo["colliders"]:
    allc = np.array([c["lo"] for c in geo["colliders"]] + [c["hi"] for c in geo["colliders"]])
    geo["collision_extent_chassis_frame"] = {"x": [float(allc[:, 0].min()), float(allc[:, 0].max())],
                                             "y": [float(allc[:, 1].min()), float(allc[:, 1].max())]}
geo["footprint"] = [round(v, 3) for v in robot.footprint]
print("GEOMETRY", json.dumps(geo)[:3000], flush=True)


def speed() -> float:
    return float(np.linalg.norm(np.asarray(robot.art.get_linear_velocity(), float)[:2]))


def reset_robot() -> None:
    place(stage, person, np.array([20.0, 0.0]), 0.0)  # never leave the person where the robot is reset
    for _ in range(3):
        world.step(render=False)
    world.reset()
    robot.post_reset()
    if a.front is not None:
        robot.front = a.front
    for _ in range(25):
        robot.apply(0.0, 0.0)
        robot.advance()


# ── 2. probe ──
probe = {}
for deg in (0, 30, 60, 90, 180):
    reset_robot()
    x0, y0, yaw0 = robot.pose()
    ang = yaw0 + math.radians(deg)
    u = np.array([math.cos(ang), math.sin(ang)])
    d, touched = 1.6, None
    while d > 0.1:
        place(stage, person, np.array([x0, y0]) + d * u, 0.0)
        robot.apply(0.0, 0.0)
        robot.advance()
        x, y, _ = robot.pose()
        if math.hypot(x - x0, y - y0) > 0.003 or speed() > 0.02:
            touched = d
            break
        d -= 0.002
    probe[deg] = {"contact_root_to_person_centre_m": None if touched is None else round(touched, 3),
                  "extent_m": None if touched is None else round(touched - RADIUS, 3)}
    print("PROBE", deg, probe[deg], flush=True)
place(stage, person, np.array([20.0, 0.0]), 0.0)

# ── 3. encounters ──
enc = []
for etype in ("pushy", "gentle"):
    for s in range(a.seeds):
        reset_robot()
        rng = np.random.default_rng(1000 + s)
        x0, y0, yaw0 = robot.pose()
        off = math.radians(float(rng.uniform(-45, 45)))  # people off the robot's heading, too
        u = np.array([math.cos(yaw0 + off), math.sin(yaw0 + off)])
        # stopped as an encounter's approach stops it: 0.10 m from the footprint (or from `front`)
        station = np.array([x0, y0]) + (robot.front + RADIUS + 0.10) * u
        if a.footprint:
            station = station + (0.10 - robot.surface_gap(station, RADIUS)) * u
        p = Person("P", "/World/People/P", station.copy())
        p.cap_path, p.vis_path, p.animated = person.cap_path, None, False  # type: ignore[attr-defined]
        p._rb, p._vis, p._char = person._rb, None, None  # type: ignore[attr-defined]
        p.front = robot.front
        t0 = robot.t
        p.schedule(rng, etype, t0, 30.0)
        touches[0] = 0
        peak, jumps, prev = 0.0, [], robot.pose()
        while robot.t < t0 + 30.0:
            x, y, _ = robot.pose()
            gap = robot.surface_gap(p.xy, RADIUS) if a.footprint else None
            pos, yaw = p.position(robot.t, np.array([x, y]), gap)
            place(stage, p, pos, yaw)
            robot.apply(0.0, 0.0)
            robot.advance()
            sp = speed()
            peak = max(peak, sp)
            cur = robot.pose()
            if sp > 3.0:  # world.EJECTION_M_S
                jumps.append({"t": round(robot.t - t0, 2), "speed": round(sp, 2), "disp": round(p.disp, 3),
                              "pose": [round(v, 3) for v in cur]})
            prev = cur
        x, y, _ = prev
        enc.append({"type": etype, "seed": s, "lunges": len(p.lunges), "peak_speed_m_s": round(peak, 3),
                    "ejections": len(jumps), "first_jumps": jumps[:3], "contact_reports": touches[0],
                    "net_displacement_m": round(math.hypot(x - x0, y - y0), 3)})
        print("ENCOUNTER", json.dumps(enc[-1]), flush=True)
(a.out / "carter_contact.json").write_text(json.dumps({"geometry": geo, "probe": probe, "encounters": enc},
                                                       indent=1))
print("CARTER CHECK DONE", flush=True)
hard_exit(0)
