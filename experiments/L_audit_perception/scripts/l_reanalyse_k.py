"""L Phase A: reanalyse programme K's saved lifetimes under the audit corrections, without new simulation.

    python l_reanalyse_k.py --out results/A_reanalysis_K.json

Modes per dataset:
- `published`: legacy confirmation + test-partition safe-mean target. Must reproduce K's numbers.
- `A1`: confirmed onset (2 consecutive, both observed) + test-partition target.
- `A1+A2task`: confirmed onset + the fixed task-derived target, stopping distance ≤ 0.30 + 0.05 m (handoff).

The validation-calibrated target needs new validation-partition lifetimes; it is part of the corrective test
(`PROTOCOL.md`), not this reanalysis.

Datasets (raw lifetimes under `runs/K/cloud/`, hashed in K's meta files): K3-X, K3-X2 (NERVA), K5-X, K5-X2
(wheeled), K6-X (humanoid).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "K_recovery" / "scripts"))
import k_analyze as KA  # noqa: E402

K = REPO / "experiments" / "K_recovery"
DATASETS = {
    "K3-X (NERVA, frozen memory)": ("runs/K/cloud/K3Xr/out/K/K3_X.jsonl", K / "configs" / "K_types.json"),
    "K3-X2 (NERVA, fixed memory)": ("runs/K/cloud/K3X2r/out/K/K3_X2.jsonl", K / "configs" / "K_types.json"),
    "K5-X (wheeled, frozen)": ("runs/K/cloud/K5X/out/K/K5_X.jsonl", K / "configs" / "K_wheeled_types.json"),
    "K5-X2 (wheeled, fixed)": ("runs/K/cloud/K5X2/out/K/K5_X2.jsonl", K / "configs" / "K_wheeled_types.json"),
    "K6-X (humanoid, fixed)": ("runs/K/cloud/K6X/out/K/K6_X.jsonl", K / "configs" / "K_humanoid_types.json"),
}
MODES = {"published": (True, "test-safe-mean"), "A1": (False, "test-safe-mean"), "A1+A2task": (False, "task")}
SYS = ("DT", "DT_VOI", "DT_CAM", "DT_CAM_VOI")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    out: dict = {}
    for name, (path, tm) in DATASETS.items():
        mcfg = json.loads(Path(tm).read_text())
        r = np.asarray(mcfg["model"]["r"], float)
        anchors = {t: float(np.asarray(v) @ r) for t, v in mcfg["oracle"].items()}
        data = KA.load([REPO / path])
        out[name] = {}
        for mode, (legacy, target) in MODES.items():
            KA.LEGACY_LAST, KA.RECOVERY_TARGET = legacy, target
            res = KA.analyse(data, r, anchors)
            crit = KA.criteria(res, tuple(s for s in SYS if s in res["systems"]))
            out[name][mode] = {s: {"pushy_to_calm": res["systems"][s]["revision_pushy_to_calm"]["median"],
                                   "recovery": (res["systems"][s].get("behaviour_recovery") or {}).get("median"),
                                   "recovered": (res["systems"][s].get("behaviour_recovery") or {}).get("revised"),
                                   "calm_to_pushy": res["systems"][s]["revision_calm_to_pushy"]["median"],
                                   "milestone": crit[s]["recovery_milestone_met"]}
                               for s in SYS if s in res["systems"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for name, modes in out.items():
        print(name)
        for s in SYS:
            if s in modes["published"]:
                print(f"  {s:11}", " | ".join(
                    f"{m}: {o[s]['pushy_to_calm']}/{o[s]['recovery']}({o[s]['recovered']})/{o[s]['calm_to_pushy']} "
                    f"{'MET' if o[s]['milestone'] else 'no'}" for m, o in modes.items()))


if __name__ == "__main__":
    main()
