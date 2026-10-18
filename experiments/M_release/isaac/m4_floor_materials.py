"""M4 debug: the physics materials (friction, restitution) of each scene's floor colliders, as PhysX will use them.

    $ISAAC_PY experiments/M_release/isaac/m4_floor_materials.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True}).app
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import exit_on_error, hard_exit  # noqa: E402

exit_on_error()
import omni.physx  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdPhysics, UsdShade  # noqa: E402

root = get_assets_root_path()
cfg = json.loads((REPO / "experiments" / "M_release" / "configs" / "scenes.json").read_text())
for name in ("home", "warehouse", "hospital"):
    create_new_stage()
    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(root + cfg[name]["usd"], "/World/Scene")
    world.reset()
    st = world.stage
    hx, hy = cfg[name]["home"][:2]
    hit = omni.physx.get_physx_scene_query_interface().raycast_closest((hx, hy, cfg[name]["floor_z"] + 2.0),
                                                                        (0.0, 0.0, -1.0), 5.0)
    out = {"scene": name, "floor_hit": str(hit.get("collision")) if hit["hit"] else None}
    if hit["hit"]:
        prim = st.GetPrimAtPath(str(hit["collision"]))
        mats = []
        p = prim
        while p and p.IsValid() and str(p.GetPath()) != "/":
            b = UsdShade.MaterialBindingAPI(p).GetDirectBinding("physics").GetMaterial()
            if b:
                m = UsdPhysics.MaterialAPI(b.GetPrim())
                mats.append({"at": str(p.GetPath()), "material": str(b.GetPath()),
                             "static": m.GetStaticFrictionAttr().Get(), "dynamic": m.GetDynamicFrictionAttr().Get(),
                             "restitution": m.GetRestitutionAttr().Get()})
                break
            p = p.GetParent()
        out["physics_material"] = mats or "none bound (PhysX default: static 0.5, dynamic 0.5)"
    print("floor", json.dumps(out, default=str), flush=True)
    world.clear_instance()
hard_exit(0)
