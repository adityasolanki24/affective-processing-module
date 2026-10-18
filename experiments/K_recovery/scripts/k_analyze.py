"""Programme K analysis: recovery from negative experience, per system (PROTOCOL.md §4).

    python k_analyze.py --runs <lifetimes.jsonl> [...] --types-model configs/K_types.json --out <json>

**Belief scalar.** p = b·r, the belief's expected detection rate (as in J). The anchors p_calm and p_pushy are r
weighted by the oracle beliefs of K_types.json. They are fixed by the type model, identical for every system, and
never fitted on test data.

**Revision** after the switch at encounter 16:
- change_B means pushy → calm; late_B means calm → pushy.
- A system has revised at the first B1 meeting after which its post-meeting p is on the new side of the midpoint
  (p_calm + p_pushy)/2 for that meeting and the next.
- Time is counted in B1 meetings after the switch. Kaplan–Meier median; lifetimes that never revise are censored.

**Behavioural recovery** (change_B): the first B1 meeting after the switch from which the stopping distance is within
0.05 m of the same system's safe-history mean B1 stopping distance (k ≥ 10), for 2 meetings.

**Per system:**
- adaptation (both directions) and behavioural recovery;
- belief correctness: share of B1 meetings after the switch whose p is on the true side;
- excessive avoidance: calm B1 encounters stopped in the far band;
- informative observations per B1 meeting;
- contacts and max push per B1 meeting; repeated-push encounters (≥ 3 contacts) on pushy B1;
- falls, trips, arrival, task success, approach time, total encounter time;
- expressive differentiation: risky_B − safe B1 stopping distance (k ≥ 10).

Paired differences against DT use the same seeds; bootstrap 95% CIs over lifetimes.
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
from apm_select.survival import km_bootstrap  # noqa: E402

SWITCH, LATE = 16, 10


def load(paths: list[Path]) -> dict:
    d: dict = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            d[r["system"]][r["history"]][r["index"]] = r["rows"]
    return d


def boot(x: list[float], n: int = 4000) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {"estimate": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"estimate": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(x))}


def p_of(b: list[float], r: np.ndarray) -> float:
    return float(np.asarray(b) @ r)


def b1(rows: list[dict]) -> list[dict]:
    return [x for x in rows if x["entity"] == "B1"]


def revision_time(rows: list[dict], r: np.ndarray, mid: float, to: str) -> tuple[float, bool]:
    after = [x for x in b1(rows) if x["k"] >= SWITCH]
    ok = [(p_of(x["b_after"], r) < mid) if to == "calm" else (p_of(x["b_after"], r) > mid) for x in after]
    for j in range(len(ok)):
        if ok[j] and (j + 1 == len(ok) or ok[j + 1]):
            return float(j + 1), True
    return float(max(1, len(ok))), False


def recovery_time(rows: list[dict], target: float) -> tuple[float, bool]:
    after = [x for x in b1(rows) if x["k"] >= SWITCH]
    ok = [abs(x["d_stop"] - target) <= 0.05 for x in after]
    for j in range(len(ok)):
        if ok[j] and (j + 1 == len(ok) or ok[j + 1]):
            return float(j + 1), True
    return float(max(1, len(ok))), False


def km(items: list[tuple[float, bool]], horizon: float) -> dict:
    res = km_bootstrap([[it] for it in items], horizon, 2000)
    res["revised"] = int(sum(e for _, e in items))
    res["lifetimes"] = len(items)
    return res


def analyse(data: dict, r: np.ndarray, anchors: dict) -> dict:
    mid = (anchors["calm"] + anchors["pushy"]) / 2
    out: dict = {"anchors": anchors, "midpoint": mid, "systems": {}}
    for s, hist in data.items():
        o: dict = {}
        safe_d = [x["d_stop"] for rows in hist.get("safe", {}).values() for x in b1(rows) if x["k"] >= LATE]
        safe_mean = float(np.mean(safe_d)) if safe_d else None
        o["safe_mean_d_stop"] = safe_mean
        horizon = 14.0
        if "change_B" in hist:
            o["revision_pushy_to_calm"] = km([revision_time(rows, r, mid, "calm")
                                              for rows in hist["change_B"].values()], horizon)
            if safe_mean is not None:
                o["behaviour_recovery"] = km([recovery_time(rows, safe_mean) for rows in hist["change_B"].values()],
                                             horizon)
        if "late_B" in hist:
            o["revision_calm_to_pushy"] = km([revision_time(rows, r, mid, "pushy") for rows in hist["late_B"].values()],
                                             horizon)
        corr = []
        for h in ("change_B", "late_B"):
            for rows in hist.get(h, {}).values():
                xs = [x for x in b1(rows) if x["k"] >= SWITCH]
                if xs:
                    corr.append(np.mean([(p_of(x["b_after"], r) > mid) == (x["type"] == "pushy") for x in xs]))
        o["belief_correct_after_switch"] = boot(corr)
        allrows = [x for h in hist.values() for rows in h.values() for x in rows]
        calm_b1 = [x for x in allrows if x["entity"] == "B1" and x["type"] == "calm"]
        push_b1 = [x for x in allrows if x["entity"] == "B1" and x["type"] == "pushy"]
        o["excess_avoidance_calm_far"] = float(np.mean([x["band"] == 2 for x in calm_b1])) if calm_b1 else None
        o["informative_per_B1"] = float(np.mean([x["informative"] for x in allrows if x["entity"] == "B1"]))
        o["windows_per_B1"] = float(np.mean([x["windows_done"] for x in allrows if x["entity"] == "B1"]))
        o["contacts_per_pushy_B1"] = float(np.mean([x["contacts"] for x in push_b1])) if push_b1 else None
        o["maxpush_per_pushy_B1"] = float(np.mean([x["max_push"] for x in push_b1])) if push_b1 else None
        o["repeated_push_rate_pushy_B1"] = float(np.mean([x["contacts"] >= 3 for x in push_b1])) if push_b1 else None
        o["falls"] = int(sum(x["fell"] for x in allrows))
        o["trip_rate"] = float(np.mean([x["trips"] > 0 for x in allrows]))
        o["arrival_rate"] = float(np.mean([x["arrived"] for x in allrows]))
        o["task_success_calm_B1"] = float(np.mean([x["task_success"] for x in calm_b1])) if calm_b1 else None
        o["task_success_all"] = float(np.mean([x["task_success"] for x in allrows]))
        o["approach_time_s"] = float(np.mean([x["t_arrive"] for x in allrows if x["t_arrive"] is not None]))
        o["encounter_time_s"] = float(np.mean([(x["t_arrive"] or 0) + x["dwell_s"] for x in allrows]))
        if "risky_B" in hist and "safe" in hist:
            keys = sorted(set(hist["risky_B"]) & set(hist["safe"]))

            def late(rows):
                return [x["d_stop"] for x in b1(rows) if x["k"] >= LATE] or [np.nan]
            o["expressive_differentiation_m"] = boot([np.mean(late(hist["risky_B"][i])) - np.mean(late(hist["safe"][i]))
                                                      for i in keys])
        out["systems"][s] = o
    if "DT" in data:
        out["vs_DT"] = {}
        for s in data:
            if s == "DT":
                continue
            d = {}
            for h in data[s]:
                keys = sorted(set(data[s][h]) & set(data["DT"].get(h, {})))
                if not keys:
                    continue

                def m(rows, f):
                    v = [f(x) for x in b1(rows) if x["k"] >= SWITCH]
                    return float(np.mean(v)) if v else np.nan
                d[h] = {
                    "d_stop_after_switch": boot([m(data[s][h][i], lambda x: x["d_stop"])
                                                 - m(data["DT"][h][i], lambda x: x["d_stop"]) for i in keys]),
                    "contacts_after_switch": boot([m(data[s][h][i], lambda x: x["contacts"])
                                                   - m(data["DT"][h][i], lambda x: x["contacts"]) for i in keys]),
                    "repeated_push_after_switch": boot([m(data[s][h][i], lambda x: float(x["contacts"] >= 3))
                                                        - m(data["DT"][h][i], lambda x: float(x["contacts"] >= 3))
                                                        for i in keys])}
            out["vs_DT"][s] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--types-model", type=Path, default=REPO / "experiments" / "K_recovery" / "configs" /
                    "K_types.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mcfg = json.loads(a.types_model.read_text())
    r = np.asarray(mcfg["model"]["r"], float)
    anchors = {t: float(np.asarray(v) @ r) for t, v in mcfg["oracle"].items()}
    res = analyse(load(a.runs), r, anchors)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for s, o in res["systems"].items():
        def f(k: str, o: dict = o):
            return (o.get(k) or {}).get("median")
        print(f"{s:11} rev p→c {f('revision_pushy_to_calm')} rec {f('behaviour_recovery')} rev c→p "
              f"{f('revision_calm_to_pushy')} correct {(o['belief_correct_after_switch'] or {}).get('estimate')} "
              f"avoid {o['excess_avoidance_calm_far']} info {o['informative_per_B1']:.2f} "
              f"rep {o['repeated_push_rate_pushy_B1']} falls {o['falls']} trips {o['trip_rate']:.3f} "
              f"arr {o['arrival_rate']:.3f} task {o['task_success_all']:.2f} t {o['encounter_time_s']:.1f} "
              f"diff {(o.get('expressive_differentiation_m') or {}).get('estimate')}")


if __name__ == "__main__":
    main()
