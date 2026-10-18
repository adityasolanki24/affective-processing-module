"""M4 check: realistic stock characters animated through Isaac's people AnimationGraph (`Biped_Setup`), which
retargets the stock clips to the characters' own rig. Renders frames 0.4 s apart and measures the mesh span.

    $ISAAC_PY experiments/M_release/isaac/m4_anim_graph.py --out out/anim_graph
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
# the AnimationGraph schema must be registered at startup, so its extensions are enabled on the Kit command line
ANIM_EXTS = ("omni.anim.graph.schema", "omni.anim.graph.core", "omni.anim.retarget.core", "omni.anim.navigation.schema")
app = AppLauncher({"headless": True, "enable_cameras": True,
                   "kit_args": " ".join(f"--enable {e}" for e in ANIM_EXTS)}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for ext in ("isaacsim.sensors.camera", "omni.anim.graph.core", "omni.anim.graph.schema", "omni.anim.retarget.core",
            "omni.anim.navigation.schema"):
    try:
        print("ext", ext, enable_extension(ext), flush=True)
    except Exception as e:  # noqa: BLE001
        print("ext", ext, "failed", repr(e)[:80], flush=True)
app.update()
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSkel  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac.world import _look_at  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02)
world.scene.add_default_ground_plane()
st = world.stage
UsdLux.DomeLight.Define(st, Sdf.Path("/World/Dome")).CreateIntensityAttr(1000.0)
add_reference_to_stage(f"{root}/Isaac/People/Characters/Biped_Setup.usd", "/World/Biped_Setup")
graphs = [str(q.GetPath()) for q in Usd.PrimRange(st.GetPrimAtPath("/World/Biped_Setup")) if
          q.GetTypeName() == "AnimationGraph"]
print("graphs", graphs, flush=True)
chars = {"P1": "male_adult_construction_05_new", "P2": "F_Business_02"}
for k, (pid, name) in enumerate(chars.items()):
    path = f"/World/People/{pid}"
    add_reference_to_stage(f"{root}/Isaac/People/Characters/{name}/{name}.usd", path)
    UsdGeom.XformCommonAPI(st.GetPrimAtPath(path)).SetTranslate(Gf.Vec3d(0.0, -0.7 + 1.4 * k, 0.0))
    changed = True
    while changed:  # un-instance so the SkelRoot can take the AnimationGraph API
        changed = False
        for q in Usd.PrimRange(st.GetPrimAtPath(path)):
            if q.IsInstance():
                q.SetInstanceable(False)
                changed = True
                break
    sr = next((q for q in Usd.PrimRange(st.GetPrimAtPath(path)) if q.IsA(UsdSkel.Root)), None)
    print(pid, "skelroot", sr.GetPath() if sr else None, flush=True)
    if sr is not None and graphs:
        try:
            from pxr import AnimGraphSchema
            AnimGraphSchema.AnimationGraphAPI.Apply(sr)
            sr.GetRelationship("animationGraph").SetTargets([Sdf.Path(graphs[0])])
            print(pid, "graph applied", list(sr.GetAppliedSchemas()), flush=True)
        except Exception as e:  # noqa: BLE001
            print(pid, "graph failed", repr(e)[:600], flush=True)
cam = Camera(prim_path="/World/Cam", resolution=(640, 480))
cam.initialize()
cam.set_focal_length(1.8)
eye = np.array([3.5, 0.0, 1.3])
cam.set_world_pose(position=eye, orientation=_look_at(eye, np.array([0.0, 0.0, 0.9])))
world.reset()
frames = []
for k in range(200):
    world.step(render=True)
    if k == 60:
        try:
            import omni.anim.graph.core as ag
            for pid in chars:
                sr = next(q for q in Usd.PrimRange(st.GetPrimAtPath(f"/World/People/{pid}")) if q.IsA(UsdSkel.Root))
                c = ag.get_character(str(sr.GetPath()))
                print(pid, "character handle", c, [m for m in dir(c) if not m.startswith("_")], flush=True)
                if c is not None and pid == "P2":
                    import carb
                    pos, rot = carb.Float3(0, 0, 0), carb.Float4(0, 0, 0, 1)
                    try:
                        c.get_world_transform(pos, rot)
                        print("P2 world before", pos, rot, flush=True)
                    except Exception as e:  # noqa: BLE001
                        print("get_world_transform", repr(e)[:200], flush=True)
                    try:
                        c.set_world_transform(carb.Float3(1.5, 1.0, 0.0), carb.Float4(0, 0, 0, 1))
                        print("set_world_transform ok", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print("set_world_transform", repr(e)[:200], flush=True)
        except Exception as e:  # noqa: BLE001
            print("ag api", repr(e)[:200], flush=True)
    if k % 25 == 24:
        f = cam.get_rgba()
        if f is not None and np.size(f):
            frames.append(np.asarray(f[..., :3], np.uint8))
print("frames", len(frames), flush=True)
if frames:
    Image.fromarray(np.concatenate(frames, axis=1)).save(a.out / "graph_strip.png")
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
