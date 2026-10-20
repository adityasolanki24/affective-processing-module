"""M3 debug: the converted NERVA USD's prim tree under a body link and a foot link (where visuals and colliders are).

    $ISAAC_PY experiments/M_release/isaac/m3_nerva_tree.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True}).app
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import exit_on_error, hard_exit  # noqa: E402

exit_on_error()
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from apm_isaac.nerva import prepare_usd  # noqa: E402

usd, mjcf, rep = prepare_usd(Path("/work/odp"), Path("/work/apm_cache/nerva_usd"))
world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd, "/World/Duck")
st = world.stage
for name in ("trunk_assembly", "foot_assembly"):
    link = next(p for p in Usd.PrimRange(st.GetPrimAtPath("/World/Duck"), Usd.TraverseInstanceProxies())
                if p.GetName() == name)
    print("LINK", link.GetPath(), "instance" if link.IsInstance() else "", flush=True)
    for q in list(Usd.PrimRange(link, Usd.TraverseInstanceProxies()))[:40]:
        print("  ", q.GetPath(), q.GetTypeName(), "INST" if q.IsInstance() else "",
              "PROXY" if q.IsInstanceProxy() else "", "COLL" if q.HasAPI(UsdPhysics.CollisionAPI) else "",
              list(q.GetAppliedSchemas())[:3], flush=True)
hard_exit(0)
