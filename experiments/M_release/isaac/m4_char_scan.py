"""M4: which stock characters share the stock clips' skeleton (`Root/Pelvis/...`)? Only those can play the clips
without retargeting.

    $ISAAC_PY experiments/M_release/isaac/m4_char_scan.py
"""
from __future__ import annotations

from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True}).app
import omni.client  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Usd, UsdSkel  # noqa: E402

root = get_assets_root_path()
clip = Usd.Stage.Open(f"{root}/Isaac/People/Animations/stand_idle_loop.skelanim.usd")
cj = next(UsdSkel.Animation(q).GetJointsAttr().Get() for q in clip.Traverse() if q.IsA(UsdSkel.Animation))
cset = set(cj)
_, entries = omni.client.list(f"{root}/Isaac/People/Characters")
for e in sorted(entries, key=lambda e: e.relative_path):
    name = e.relative_path
    _, files = omni.client.list(f"{root}/Isaac/People/Characters/{name}")
    usds = sorted(f.relative_path for f in files if f.relative_path.endswith(".usd"))
    if not usds:
        continue
    try:
        st = Usd.Stage.Open(f"{root}/Isaac/People/Characters/{name}/{usds[0]}")
        sk = next((q for q in st.Traverse(Usd.TraverseInstanceProxies()) if q.IsA(UsdSkel.Skeleton)), None)
        sj = list(UsdSkel.Skeleton(sk).GetJointsAttr().Get() or []) if sk else []
        print("char", name, usds[0], "joints", len(sj), sj[:2], "common_with_clip", len(cset & set(sj)), flush=True)
    except Exception as ex:  # noqa: BLE001
        print("char", name, "error", repr(ex)[:100], flush=True)
app.close()
