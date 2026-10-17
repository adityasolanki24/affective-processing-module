"""J5 analysis: the six closed-loop questions (MASTER_PREREGISTRATION §4.3, J5 addendum).

    python j5_analyze.py --runs <lifetimes.jsonl ...> [--calibration <json>] --out <json>

Lifetimes are paired by index across systems and histories. "B1 encounters" are the encounters with the
manipulated entity. LATE is the encounter index from which behaviour counts as experienced (10).

1. **Outcomes → beliefs.** CL: B1 belief at its last encounter, risky_B − safe (paired), and history AUROC.
2. **Beliefs → expression.** CL: Spearman correlation of belief with stopping distance and with L0 caution over B1
   encounters; B1 stopping distance risky_B − safe after LATE.
3. **Expression → outcomes.** risky_B, B1 encounters after LATE: contacts per encounter and mean maximum push, CL vs
   FROZEN (same seeds). Expression must *change what happens*.
4. **Revision.** change_B (pushy → calm) and late_B (calm → pushy): B1 encounters after the switch until the belief
   decisively crosses the calibrated midpoint. Calibration: risky_B and safe B1 beliefs at the last encounter, from
   `--calibration` (V) or from the same file for pilots. CL vs NOFEEDBACK.
5. **Recovery from excess caution.** change_B: B1 encounters after the switch until the stopping distance is within
   0.05 m of the safe-history mean (CL), against NOFEEDBACK.
6. **Safety and task.** Falls, safety trips, arrival rate, and mean arrival time per system; CL vs FROZEN arrival-time
   difference.

Ablations (NOMEM, RANDHIST, H_MEM) get the same measures where defined.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm.learning.evaluation import auroc  # noqa: E402
from apm_select.survival import km_bootstrap  # noqa: E402

LATE, SWITCH, HORIZON_B = 10, 15, 8


def load(paths: list[Path]) -> dict:
    d: dict = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            d[r["system"]][r["history"]][r["index"]] = r["rows"]
    return d


def b1(rows: list[dict]) -> list[dict]:
    return [x for x in rows if x["entity"] == "B1"]


def boot(vals: list[float], n: int = 4000) -> dict:
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return {"estimate": None}
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return {"estimate": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(v))}


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def last_p(rows: list[dict]) -> float:
    bs = b1(rows)
    return bs[-1]["p_before"] if bs else float("nan")


def calibrate(data: dict, system: str = "CL") -> dict:
    mu_p = float(np.nanmean([last_p(r) for r in data[system]["risky_B"].values()]))
    mu_c = float(np.nanmean([last_p(r) for r in data[system]["safe"].values()]))
    return {"mu_pushy": mu_p, "mu_calm": mu_c, "threshold": (mu_p + mu_c) / 2}


def revision(rows_by_i: dict, cal: dict, to: str) -> dict:
    mu_post = cal["mu_calm"] if to == "calm" else cal["mu_pushy"]
    deep = (cal["threshold"] + mu_post) / 2
    sign = 1.0 if mu_post >= cal["threshold"] else -1.0
    groups = []
    for rows in rows_by_i.values():
        post = [x["p_before"] for x in b1(rows) if x["k"] >= SWITCH][1:]  # beliefs after ≥ 1 post-switch encounter
        t = next((j + 1 for j in range(len(post)) if sign * (post[j] - deep) > 0
                  and (j == len(post) - 1 or sign * (post[j + 1] - deep) > 0)), None)
        groups.append([(float(t), True)] if t is not None else [(float(HORIZON_B), False)])
    return km_bootstrap(groups, HORIZON_B, 2000)


def recovery(rows_by_i: dict, safe_d: float) -> dict:
    groups = []
    for rows in rows_by_i.values():
        post = [x["d_stop"] for x in b1(rows) if x["k"] >= SWITCH]
        t = next((j + 1 for j, d in enumerate(post) if d <= safe_d + 0.05), None)
        groups.append([(float(t), True)] if t is not None else [(float(HORIZON_B), False)])
    return km_bootstrap(groups, HORIZON_B, 2000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = load(a.runs)
    cal = json.loads(a.calibration.read_text()) if a.calibration else calibrate(data)
    res: dict = {"calibration": cal, "systems": {}}
    safe_d = float(np.mean([x["d_stop"] for r in data["CL"]["safe"].values() for x in b1(r) if x["k"] >= LATE]))
    res["safe_mean_d_stop_CL"] = safe_d
    for s, hist in data.items():
        out: dict = {}
        if "risky_B" in hist and "safe" in hist:
            keys = sorted(set(hist["risky_B"]) & set(hist["safe"]))
            rp = [last_p(hist["risky_B"][i]) for i in keys]
            sp = [last_p(hist["safe"][i]) for i in keys]
            out["q1_belief_effect"] = boot([x - y for x, y in zip(rp, sp, strict=True)])
            out["q1_history_auroc"] = auroc(rp + sp, [True] * len(rp) + [False] * len(sp))
            late = lambda r: [x for x in b1(r) if x["k"] >= LATE]  # noqa: E731
            out["q2_dstop_effect"] = boot([np.mean([x["d_stop"] for x in late(hist["risky_B"][i])])
                                           - np.mean([x["d_stop"] for x in late(hist["safe"][i])]) for i in keys])
            allb = [x for h in hist.values() for r in h.values() for x in b1(r)]
            out["q2_spearman_p_dstop"] = spearman([x["p_before"] for x in allb], [x["d_stop"] for x in allb])
            out["q2_spearman_p_caution"] = spearman([x["p_before"] for x in allb], [x["caution"] for x in allb])
            out["risky_B_contacts_per_B1"] = boot([np.mean([x["contacts"] for x in late(hist["risky_B"][i])])
                                                   for i in keys])
            out["risky_B_maxpush_per_B1"] = boot([np.mean([x["max_push"] for x in late(hist["risky_B"][i])])
                                                  for i in keys])
        if "change_B" in hist:
            out["q4_revision_change_B"] = revision(hist["change_B"], cal, "calm")
            out["q5_recovery_change_B"] = recovery(hist["change_B"], safe_d)
        if "late_B" in hist:
            out["q4_revision_late_B"] = revision(hist["late_B"], cal, "pushy")
        rows = [x for h in hist.values() for r in h.values() for x in r]
        out["q6"] = {"encounters": len(rows), "falls": int(sum(x["fell"] for x in rows)),
                     "trips": int(sum(x["trips"] for x in rows)),
                     "arrival_rate": float(np.mean([x["arrived"] for x in rows])),
                     "mean_t_arrive": float(np.mean([x["t_arrive"] for x in rows if x["t_arrive"] is not None])),
                     "contacts_per_encounter": float(np.mean([x["contacts"] for x in rows]))}
        res["systems"][s] = out
    if "CL" in data and "FROZEN" in data:  # Q3 and the Q6 time cost: paired CL vs FROZEN
        cl, fr = data["CL"]["risky_B"], data["FROZEN"]["risky_B"]
        keys = sorted(set(cl) & set(fr))
        late = lambda r: [x for x in b1(r) if x["k"] >= LATE]  # noqa: E731
        res["q3_contacts_CL_minus_FROZEN"] = boot([np.mean([x["contacts"] for x in late(cl[i])])
                                                  - np.mean([x["contacts"] for x in late(fr[i])]) for i in keys])
        res["q3_maxpush_CL_minus_FROZEN"] = boot([np.mean([x["max_push"] for x in late(cl[i])])
                                                 - np.mean([x["max_push"] for x in late(fr[i])]) for i in keys])
        res["q3_detected_CL_minus_FROZEN"] = boot([np.mean([x["detected"] for x in late(cl[i])])
                                                  - np.mean([x["detected"] for x in late(fr[i])]) for i in keys])
        t_cl = [np.mean([x["t_arrive"] for x in r if x["t_arrive"] is not None]) for h in data["CL"].values()
                for r in h.values()]
        t_fr = [np.mean([x["t_arrive"] for x in r if x["t_arrive"] is not None]) for h in data["FROZEN"].values()
                for r in h.values()]
        res["q6_arrival_time_CL_minus_FROZEN_s"] = boot([x - y for x, y in zip(t_cl, t_fr, strict=False)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for s, o in res["systems"].items():
        print(s, {k: (round(v["estimate"], 3) if isinstance(v, dict) and v.get("estimate") is not None else
                      v.get("median") if isinstance(v, dict) and "median" in v else
                      round(v, 3) if isinstance(v, float) else v)
                  for k, v in o.items() if k != "q6"}, o["q6"])
    for k in ("q3_contacts_CL_minus_FROZEN", "q3_maxpush_CL_minus_FROZEN", "q3_detected_CL_minus_FROZEN",
              "q6_arrival_time_CL_minus_FROZEN_s"):
        if k in res:
            print(k, res[k])


if __name__ == "__main__":
    main()
