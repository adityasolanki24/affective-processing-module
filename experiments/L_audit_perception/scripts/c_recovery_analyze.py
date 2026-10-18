"""L protocol §C.3: recovery with IMU only, the abstract camera, rendered vision, IMU + vision and the oracle.

    python c_recovery_analyze.py --body nerva --runs <K-model X.jsonl> <vision-model X.jsonl>
        --vision-types configs/LC_nerva_vision_types.json --targets configs/LC_recovery_targets.json
        --out results/C_recovery_nerva.json

The same criteria as §B.3 (`l_b_analyze.py`). The predicted-risk audit uses the type model that each system ran
with: K's model for DT_VOI, DT_CAM_VOI and ORACLE, and the body's vision model for DT_VIS_VOI and DT_IMUVIS_VOI.
Paired differences against DT_CAM_VOI (the abstract camera) are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l_b_analyze as LB  # noqa: E402

KA = LB.KA
VISION = ("DT_VIS_VOI", "DT_IMUVIS_VOI", "DT_IMUVIS")
JUDGED = ("DT_VOI", "DT_CAM_VOI", "DT_VIS_VOI", "DT_IMUVIS_VOI")
BODY = {"nerva": "LB1", "wheeled": "LB2"}


def recovery_items(data: dict, s: str) -> dict:
    return {i: KA.recovery_time(rows, None) for i, rows in data[s].get("change_B", {}).items()}


def paired(data: dict, a: str, b: str) -> dict:
    """a − b over paired change_B lifetimes: recovery time (censored at the horizon) and contacts per pushy B1."""
    ra, rb = recovery_items(data, a), recovery_items(data, b)
    keys = sorted(set(ra) & set(rb))
    rec = KA.boot([min(ra[i][0], 14.0) - min(rb[i][0], 14.0) for i in keys])

    def cpm(s: str, h: str, i: int) -> float:
        v = [x["contacts"] for x in data[s][h][i] if x["entity"] == "B1" and x["type"] == "pushy"]
        return float(np.mean(v)) if v else np.nan
    con = KA.boot([cpm(a, h, i) - cpm(b, h, i) for h in ("risky_B", "change_B", "late_B")
                   for i in sorted(set(data[a].get(h, {})) & set(data[b].get(h, {})))])
    return {"recovery_meetings": rec, "contacts_per_pushy_B1": con}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", choices=tuple(BODY), required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--vision-types", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _, types_name, time_name = LB.BODIES[BODY[a.body]]
    k_types = LB.K / "configs" / types_name
    time_model = (LB.REPO / "experiments" / "J_validation_transfer" / "configs" / "J6_model.json"
                  if time_name == "J" else LB.K / "configs" / time_name)
    data = KA.load(a.runs)
    targets = json.loads(a.targets.read_text())["targets"][a.body]
    res = {}
    for label, model_path, systems in (("k_model", k_types, [s for s in data if s not in VISION]),
                                       ("vision_model", a.vision_types, [s for s in data if s in VISION])):
        if not systems:
            continue
        sub = {s: data[s] for s in systems}
        mcfg = json.loads(model_path.read_text())
        r = np.asarray(mcfg["model"]["r"], float)
        anchors = {t: float(np.asarray(v) @ r) for t, v in mcfg["oracle"].items()}
        KA.LEGACY_LAST, KA.RECOVERY_TARGET = False, "task"
        rt = KA.analyse(sub, r, anchors)
        KA.RECOVERY_TARGET = targets
        rv = KA.analyse(sub, r, anchors)
        cap = LB.cap_audit_with(sub, model_path, time_model, vision=VISION)
        res[label] = {"task": rt, "V": rv, "cap": cap}
    crit, summary = {}, {}
    for label, o in res.items():
        for s in o["task"]["systems"]:
            ct = KA.criteria(o["task"], (s,)).get(s)
            cv = KA.criteria(o["V"], (s,)).get(s)
            st, sv = o["task"]["systems"][s], o["V"]["systems"][s]
            summary[s] = {"model": label,
                          "pushy_to_calm": (st.get("revision_pushy_to_calm") or {}).get("median"),
                          "recovery_task": (st.get("behaviour_recovery") or {}).get("median"),
                          "recovered_task": (st.get("behaviour_recovery") or {}).get("revised"),
                          "recovery_V": (sv.get("behaviour_recovery") or {}).get("median"),
                          "calm_to_pushy": (st.get("revision_calm_to_pushy") or {}).get("median"),
                          "contacts_per_pushy_B1": st["contacts_per_pushy_B1"], "falls": st["falls"],
                          "arrival_rate": st["arrival_rate"], "cap_over": o["cap"].get(s, {}).get("over_cap")}
            if s in JUDGED and ct is not None and cv is not None:
                c = {"P1": ct["P1_revision_pushy_to_calm"], "P2_task": ct["P2_behaviour_recovery"],
                     "P2_V": cv["P2_behaviour_recovery"], "P3": ct["P3_revision_calm_to_pushy"],
                     "P4": ct["P4_safety"] and o["cap"].get(s, {}).get("over_cap", 0) == 0, "P5": ct["P5_task"]}
                c["milestone"] = all(c.values())
                crit[s] = c
    pairs = {f"{s}_minus_DT_CAM_VOI": paired(data, s, "DT_CAM_VOI") for s in ("DT_VIS_VOI", "DT_IMUVIS_VOI", "DT_VOI")
             if s in data and "DT_CAM_VOI" in data}
    out = {"body": a.body, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.runs},
           "summary": summary, "criteria": crit, "paired_vs_abstract_camera": pairs,
           "questions": {"Q1_imu_plus_vision_milestone": crit.get("DT_IMUVIS_VOI", {}).get("milestone"),
                         "Q2_vision_only_milestone": crit.get("DT_VIS_VOI", {}).get("milestone")},
           "analysis": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for s, m in summary.items():
        print(f"{s:14} {m} {'MET' if crit.get(s, {}).get('milestone') else ''}")
    for k, v in pairs.items():
        print(k, {kk: (vv["estimate"], vv["ci95"]) for kk, vv in v.items()})
    print(out["questions"])


if __name__ == "__main__":
    main()
