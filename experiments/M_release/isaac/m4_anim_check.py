"""M4 check: does a stock character play its idle clip (`agents.Animator`)? Renders frames 0.4 s apart and lists
the stock animation clips.

    $ISAAC_PY experiments/M_release/isaac/m4_anim_check.py --out out/anim
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True, "enable_cameras": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
for ext in ("omni.anim.skelJoint", "omni.anim.graph.core", "omni.anim.timeline", "omni.kit.usd.layers"):
    try:
        print("ext", ext, enable_extension(ext), flush=True)
    except Exception as e:  # noqa: BLE001
        print("ext", ext, "failed", repr(e)[:80], flush=True)
import numpy as np  # noqa: E402
import omni.client  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Sdf, UsdLux  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac.agents import add_person, find_character  # noqa: E402
from apm_isaac.world import _look_at  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
res, entries = omni.client.list(f"{root}/Isaac/People/Animations")
print("clips", sorted(e.relative_path for e in entries), flush=True)
world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02)
world.scene.add_default_ground_plane()
UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Dome")).CreateIntensityAttr(1000.0)
ch = find_character(root, "male_adult_construction_05_new")
print("character", ch, flush=True)
from pxr import Usd  # noqa: E402

probe = world.stage.DefinePrim("/World/Probe")
probe.GetReferences().AddReference(ch)
for q in list(Usd.PrimRange(probe, Usd.TraverseInstanceProxies()))[:40]:
    print("  prim", q.GetPath(), q.GetTypeName(), "instance" if q.IsInstance() else "",
          "proxy" if q.IsInstanceProxy() else "", flush=True)
world.stage.RemovePrim("/World/Probe")
p = add_person(world.stage, "P1", np.array([0.0, -0.6]), ch)
an = p.animator  # type: ignore[attr-defined]
# second character: the clip's pose at 1 s baked into its mesh points on the CPU (no GPU skinning needed)
p2 = add_person(world.stage, "P2", np.array([0.0, 0.6]), ch)
an2 = p2.animator  # type: ignore[attr-defined]
from pxr import Gf, UsdSkel  # noqa: E402

if an2 and an2.ok:
    an2.update(1.0)
    roots = [q for q in Usd.PrimRange(world.stage.GetPrimAtPath("/World/People/P2/character")) if q.IsA(UsdSkel.Root)]
    print("bake roots", [str(r.GetPath()) for r in roots], flush=True)
    for r in roots:
        try:
            print("bake", UsdSkel.BakeSkinning(UsdSkel.Root(r), Gf.Interval(0.0, 0.0)), flush=True)
        except Exception as e:  # noqa: BLE001
            print("bake failed", repr(e)[:200], flush=True)
if an and an.ok:
    sk = next(q for q in Usd.PrimRange(world.stage.GetPrimAtPath("/World/People/P1/character"))
              if q.IsA(UsdSkel.Skeleton))
    sj = list(UsdSkel.Skeleton(sk).GetJointsAttr().Get() or [])
    cj = list(an.src.GetJointsAttr().Get() or [])
    print("skeleton joints", len(sj), sj[:4], "clip joints", len(cj), cj[:4], "common", len(set(sj) & set(cj)),
          "common leaf names", len({j.split("/")[-1] for j in sj} & {j.split("/")[-1] for j in cj}), flush=True)


def span(path: str) -> float:
    from pxr import UsdGeom
    xs = []
    for q in Usd.PrimRange(world.stage.GetPrimAtPath(path)):
        if q.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(q).GetPointsAttr().Get()
            if pts:
                arr = np.asarray(pts)
                xs.append((arr[:, 0].min(), arr[:, 0].max(), arr[:, 1].min(), arr[:, 1].max()))
    if not xs:
        return -1.0
    a_ = np.asarray(xs)
    return round(float(max(a_[:, 1].max() - a_[:, 0].min(), a_[:, 3].max() - a_[:, 2].min())), 3)


print("mesh span (T-pose ~1.7 m, arms down ~0.6 m): P1", span("/World/People/P1/character"), "P2 baked",
      span("/World/People/P2/character"), flush=True)
import omni.timeline  # noqa: E402

tl = omni.timeline.get_timeline_interface()
print("animator ok", an.ok if an else None, an.why if an else None,
      "span", (an.t0, an.t1, an.tcps) if an and an.ok else None, flush=True)
cam = Camera(prim_path="/World/Cam", resolution=(640, 480))
cam.initialize()
cam.set_focal_length(1.8)
eye = np.array([0.0, 3.5, 1.3])
cam.set_world_pose(position=eye, orientation=_look_at(eye, np.array([0.0, 0.0, 0.9])))
world.reset()
frames = []
for k in range(160):
    t = k * 0.02
    if an:
        an.update(t)
    world.step(render=True)
    if k % 40 == 0:
        print("timeline", round(tl.get_current_time(), 3), tl.is_playing(), flush=True)
    if k % 20 == 19:
        f = cam.get_rgba()
        if f is not None and np.size(f):
            frames.append(np.asarray(f[..., :3], np.uint8))
print("frames", len(frames), "pixel change between frames",
      [round(float(np.abs(frames[i + 1].astype(int) - frames[i].astype(int)).mean()), 3)
       for i in range(len(frames) - 1)], flush=True)
if frames:
    Image.fromarray(np.concatenate(frames, axis=1)).save(a.out / "anim_strip.png")
app.close()
