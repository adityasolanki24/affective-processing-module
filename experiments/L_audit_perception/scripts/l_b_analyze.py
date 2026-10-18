"""L protocol §B.3-B.4: the corrective recovery test and the verdicts on programme K's claims.

    python l_b_analyze.py --lb1 <X.jsonl> --lb2 <X.jsonl> --lb3 <X.jsonl> --lb4 results/LB4_tasks.json
        --targets configs/LB_recovery_targets.json --out results/B_corrective.json

LB4 is analysed first, by `k3b_analyze.py --measured-success`.

Per body:
- K's analysis twice, with the task criterion (P2-task) and with the V-calibrated targets (P2-V);
- a recomputation of every executed approach's predicted risk from the logged belief (P4 cap compliance);
- the task outcomes (interaction reached, hold completed, holds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
K = REPO / "experiments" / "K_recovery"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(K / "scripts"))
import k3b_analyze as K3B  # noqa: E402
import k_analyze as KA  # noqa: E402

BODIES = {"LB1": ("nerva", "K_types.json", "J"), "LB2": ("wheeled", "K_wheeled_types.json", "K_wheeled_time.json"),
          "LB3": ("humanoid", "K_humanoid_types.json", "K_humanoid_time.json")}
CANDIDATES = ("DT_VOI", "DT_CAM_VOI")
JUDGED = ("DT_VOI", "DT_CAM_VOI", "DT", "DT_CAM")
CAPPED = ("DT_STATIC", "DT", "DT_VOI", "DT_UNC", "DT_CAM", "DT_CAM_VOI")  # ORACLE acts on the true type; L0 has no cap
CAM = ("DT_CAM", "DT_CAM_VOI", "DT_CAM_VOI_K")
TOL = 1e-4  # beliefs are logged to 5 decimals


def cap_audit(data: dict, types: Path, time_model: Path) -> dict:
    from apm_revise.belief import TypeModel
    from apm_transfer.expression_map import TimeModel
    from apm_transfer.infoseek import Emissions, Task, cost
    model = TypeModel.from_dict(json.loads(types.read_text())["model"])
    tm = TimeModel(json.loads(time_model.read_text())["time_coef"])
    spec = json.loads((K / "configs" / "K_tasks.json").read_text())["tasks"]["handoff"]
    task = Task("handoff", **{k: v for k, v in spec.items() if k != "why"})
    em = Emissions.from_model(model.E, camera_block=model.E.shape[1] > 24)
    out = {}
    for s in data:
        if s not in CAPPED and not s.endswith("_K"):
            continue
        e = em if s in CAM else Emissions(em.imu, None)
        n = over = 0
        for hist in data[s].values():
            for rows in hist.values():
                for x in rows:
                    n += 1
                    if x["grid_i"] is not None and x["grid_i"] >= 0:
                        risk = cost(x["grid_i"], x["windows_planned"], np.asarray(x["b_before"]), tm, e, task)[1]
                        over += int(risk > task.risk_cap + TOL)
        out[s] = {"decisions": n, "over_cap": over, "share": round(over / max(n, 1), 4)}
    return out


def cap_audit_with(data: dict, types: Path, time_model: Path, vision: tuple[str, ...] = ()) -> dict:
    """`cap_audit` for programme L §C: the vision systems are capped and observe a camera block too."""
    from apm_revise.belief import TypeModel
    from apm_transfer.expression_map import TimeModel
    from apm_transfer.infoseek import Emissions, Task, cost
    model = TypeModel.from_dict(json.loads(types.read_text())["model"])
    tm = TimeModel(json.loads(time_model.read_text())["time_coef"])
    spec = json.loads((K / "configs" / "K_tasks.json").read_text())["tasks"]["handoff"]
    task = Task("handoff", **{k: v for k, v in spec.items() if k != "why"})
    em = Emissions.from_model(model.E, camera_block=model.E.shape[1] > 24)
    out = {}
    for s in data:
        if s not in CAPPED + vision:
            continue
        e = em if s in CAM + vision else Emissions(em.imu, None)
        n = over = 0
        for hist in data[s].values():
            for rows in hist.values():
                for x in rows:
                    n += 1
                    if x["grid_i"] is not None and x["grid_i"] >= 0:
                        risk = cost(x["grid_i"], x["windows_planned"], np.asarray(x["b_before"]), tm, e, task)[1]
                        over += int(risk > task.risk_cap + TOL)
        out[s] = {"decisions": n, "over_cap": over, "share": round(over / max(n, 1), 4)}
    return out


def outcomes(data: dict) -> dict:
    out = {}
    for s, hist in data.items():
        rows = [x for h in hist.values() for lt in h.values() for x in lt]
        b1 = [x for x in rows if x["entity"] == "B1"]
        calm = [x for x in b1 if x["type"] == "calm"]
        out[s] = {"hold_rate_B1": round(float(np.mean([x["no_safe_action"] for x in b1])), 4),
                  "observe_rate_B1": round(float(np.mean([x["standoff"] > 0 and not x["no_safe_action"]
                                                          for x in b1])), 4),
                  "reached_interaction_calm_B1": round(float(np.mean([x["reached_interaction"] for x in calm])), 4),
                  "hold_completed_calm_B1": round(float(np.mean([x["hold_completed"] for x in calm])), 4)}
    return out


def contact_ratio(data: dict, a: str = "DT_CAM_VOI", b: str = "DT", n: int = 4000) -> dict | None:
    """Contacts per pushy B1 meeting, a over b, pooled over paired lifetimes; bootstrap over lifetimes."""
    if a not in data or b not in data:
        return None
    keys = sorted({(h, i) for h in data[a] for i in data[a][h]} & {(h, i) for h in data[b] for i in data[b][h]})

    def per(s: str) -> list[list[int]]:
        return [[x["contacts"] for x in data[s][h][i] if x["entity"] == "B1" and x["type"] == "pushy"]
                for h, i in keys]
    pa, pb = per(a), per(b)
    sa, sb = np.array([sum(v) for v in pa], float), np.array([sum(v) for v in pb], float)
    na, nb = np.array([len(v) for v in pa], float), np.array([len(v) for v in pb], float)
    est = (sa.sum() / na.sum()) / (sb.sum() / nb.sum())
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        j = rng.integers(0, len(keys), len(keys))
        bs.append((sa[j].sum() / max(na[j].sum(), 1)) / max(sb[j].sum() / max(nb[j].sum(), 1), 1e-12))
    return {"ratio": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "lifetimes": len(keys)}


def task_cost(data: dict, a: str = "DT_CAM_VOI", b: str = "DT") -> dict | None:
    """Reported next to C5: realised handoff cost per B1 meeting with measured success (a hold is a failure)."""
    if a not in data or b not in data:
        return None
    K3B.MEASURED = True
    spec = json.loads((K / "configs" / "K_tasks.json").read_text())["tasks"]["handoff"]

    def per(s: str, h: str, i: int) -> float:
        return float(np.mean([K3B.enc_cost(x, spec) for x in data[s][h][i] if x["entity"] == "B1"] or [np.nan]))
    keys = sorted({(h, i) for h in data[a] for i in data[a][h]} & {(h, i) for h in data[b] for i in data[b][h]})
    return {"mean": {a: float(np.nanmean([per(a, h, i) for h, i in keys])),
                     b: float(np.nanmean([per(b, h, i) for h, i in keys]))},
            "diff": KA.boot([per(a, h, i) - per(b, h, i) for h, i in keys])}


def body(exp: str, path: Path, targets: dict) -> dict:
    robot, types_name, time_name = BODIES[exp]
    types = K / "configs" / types_name
    time_model = (REPO / "experiments" / "J_validation_transfer" / "configs" / "J6_model.json" if time_name == "J"
                  else K / "configs" / time_name)
    mcfg = json.loads(types.read_text())
    r = np.asarray(mcfg["model"]["r"], float)
    anchors = {t: float(np.asarray(v) @ r) for t, v in mcfg["oracle"].items()}
    data = KA.load([path])
    KA.LEGACY_LAST = False
    KA.RECOVERY_TARGET = "task"
    res_task = KA.analyse(data, r, anchors)
    KA.RECOVERY_TARGET = targets[exp]
    res_v = KA.analyse(data, r, anchors)
    cap = cap_audit(data, types, time_model)
    crit = {}
    for s in JUDGED:
        if s not in res_task["systems"]:
            continue
        ct = KA.criteria(res_task, (s,))[s]
        cv = KA.criteria(res_v, (s,))[s]
        c = {"P1": ct["P1_revision_pushy_to_calm"], "P2_task": ct["P2_behaviour_recovery"],
             "P2_V": cv["P2_behaviour_recovery"], "P3": ct["P3_revision_calm_to_pushy"],
             "P4": ct["P4_safety"] and cap.get(s, {}).get("over_cap", 0) == 0, "P5": ct["P5_task"]}
        c["milestone"] = all(c.values())
        crit[s] = c

    def med(res: dict, s: str, k: str):
        return ((res["systems"].get(s) or {}).get(k) or {}).get("median")

    def rev(res: dict, s: str):
        return (res["systems"][s].get("behaviour_recovery") or {}).get("revised")
    summary = {s: {"pushy_to_calm": med(res_task, s, "revision_pushy_to_calm"),
                   "recovery_task": med(res_task, s, "behaviour_recovery"), "recovered_task": rev(res_task, s),
                   "recovery_V": med(res_v, s, "behaviour_recovery"), "recovered_V": rev(res_v, s),
                   "calm_to_pushy": med(res_task, s, "revision_calm_to_pushy"),
                   "contacts_per_pushy_B1": res_task["systems"][s]["contacts_per_pushy_B1"],
                   "falls": res_task["systems"][s]["falls"], "trip_rate": res_task["systems"][s]["trip_rate"],
                   "arrival_rate": res_task["systems"][s]["arrival_rate"]} for s in res_task["systems"]}
    return {"robot": robot, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": summary, "criteria": crit, "cap_audit": cap, "outcomes": outcomes(data),
            "contact_ratio_DT_CAM_VOI_over_DT": contact_ratio(data),
            "contact_ratio_DT_CAM_VOI_K_over_DT_K": contact_ratio(data, "DT_CAM_VOI_K", "DT_K"),
            "task_cost_B1_DT_CAM_VOI_minus_DT": task_cost(data),
            "analysis_task": res_task, "analysis_V": res_v}


def verdict_milestone(c: dict | None) -> str:
    if c is None:
        return "not run"
    if c["milestone"]:
        return "survives"
    if c["P1"] and c["P3"] and c["P4"] and c["P5"] and (c["P2_task"] != c["P2_V"]):
        return "weakens"
    return "fails"


def verdicts(res: dict, lb4: dict | None) -> dict:
    nerva = res.get("LB1", {"criteria": {}, "contact_ratio_DT_CAM_VOI_over_DT": None})
    v = {"C1_nerva_camera_voi": verdict_milestone(nerva["criteria"].get("DT_CAM_VOI")),
         "C2_nerva_imu_only": verdict_milestone(nerva["criteria"].get("DT_VOI"))}
    for name, exp in (("C3_wheeled", "LB2"), ("C4_humanoid", "LB3")):
        if exp not in res:
            v[name] = "not run"
            continue
        cam = verdict_milestone(res[exp]["criteria"].get("DT_CAM_VOI"))
        imu_met = res[exp]["criteria"].get("DT_VOI", {}).get("milestone", False)
        v[name] = "fails" if cam == "fails" or imu_met else cam
    cr = nerva["contact_ratio_DT_CAM_VOI_over_DT"]
    v["C5_information_cuts_contacts"] = ("not run" if cr is None else
                                         "survives" if cr["ratio"] <= 0.5 and cr["ci95"][1] < 1 else
                                         "weakens" if cr["ci95"][1] < 1 else "fails")
    if lb4 is not None:
        p = lb4["tasks_passed"]
        v["C6_dt_beats_lookup"] = "survives" if p == 4 else "weakens" if p == 3 else "fails"
    p3 = [res[e]["summary"][s]["calm_to_pushy"] for e in res for s in CANDIDATES if s in res[e]["summary"]]
    v["C7_caution_returns"] = ("fails" if any(x is None or x > 4 for x in p3) else
                               "survives" if all(x <= 2 for x in p3) else "weakens")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb1", type=Path)
    ap.add_argument("--lb2", type=Path)
    ap.add_argument("--lb3", type=Path)
    ap.add_argument("--lb4", type=Path, help="k3b_analyze.py --measured-success output")
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    targets = json.loads(a.targets.read_text())["targets"]
    res = {exp: body(exp, p, targets) for exp, p in (("LB1", a.lb1), ("LB2", a.lb2), ("LB3", a.lb3)) if p}
    lb4 = json.loads(a.lb4.read_text()) if a.lb4 else None
    out = {"bodies": res, "lb4": lb4, "verdicts": verdicts(res, lb4)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for exp, o in res.items():
        print(f"== {exp} ({o['robot']})")
        for s, m in o["summary"].items():
            print(f"  {s:13} p->c {m['pushy_to_calm']} rec task {m['recovery_task']} ({m['recovered_task']}) "
                  f"V {m['recovery_V']} ({m['recovered_V']}) c->p {m['calm_to_pushy']} "
                  f"contacts {m['contacts_per_pushy_B1']} falls {m['falls']} arr {m['arrival_rate']:.3f} "
                  f"cap {o['cap_audit'].get(s, {}).get('share')} hold {o['outcomes'][s]['hold_rate_B1']} "
                  f"{'MET' if o['criteria'].get(s, {}).get('milestone') else ''}")
        print("  contact ratio", o["contact_ratio_DT_CAM_VOI_over_DT"])
        print("  task cost", o["task_cost_B1_DT_CAM_VOI_minus_DT"])
    print("verdicts", json.dumps(out["verdicts"], indent=1))


if __name__ == "__main__":
    main()
