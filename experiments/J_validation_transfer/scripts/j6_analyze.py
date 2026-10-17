"""J6 analysis: does a learned expression mapping improve on H's fixed L0? (MASTER_PREREGISTRATION §4.4, J6 addendum)

    python j6_analyze.py --runs <J6-X lifetimes.jsonl> --price 10 --v-price 40 --v-value 5 --out <json>

Per system:
- J5's loop measures (`j5_analyze`: Q1–Q6);
- **utility**, robot-observed: per lifetime, mean arrival time + price × the share of observable encounters with
  strong evidence (acc8 ≥ 6). Lower is better;
- **value utility** (DEVIATIONS D3): as above with `--v-price`, plus `--v-value` × the mean stopping shortfall
  max(0, d_stop − 0.3 m) / 0.1 m, i.e. the cost of not coming close enough for the task;
- **belief dependence:** Spearman(belief p, stopping distance) over B1 encounters, the number of distinct grid intents
  used, and the risky_B − safe stopping-distance effect;
- truth for evaluation: contacts and max push per encounter.

Paired comparisons (same seeds) of LEARNED, RESIDUAL and GRU (and their _V variants) against CL (L0): utility,
contacts, max push, arrival time. **Loop question (J5's Q3) per system:** risky_B, B1 encounters after LATE, mean max
push and contacts minus FROZEN's (paired).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from j5_analyze import LATE, b1, boot, last_p, load, spearman  # noqa: E402

STRONG = 6
D_TASK = 0.3


def utility(rows: list[dict], price: float, value: float = 0.0) -> float:
    t = np.mean([x["t_arrive"] for x in rows if x["t_arrive"] is not None])
    obs = [x for x in rows if x.get("observable", True)]
    strong = np.mean([x["category"] >= STRONG for x in obs]) if obs else 0.0
    short = np.mean([max(0.0, x["d_stop"] - D_TASK) / 0.1 for x in rows])
    return float(t + price * strong + value * short)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--price", type=float, default=10.0)
    ap.add_argument("--v-price", type=float, default=40.0)
    ap.add_argument("--v-value", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = load(a.runs)
    res: dict = {"price": a.price, "v_price": a.v_price, "v_value": a.v_value, "systems": {}, "vs_L0": {}}
    uv = lambda r: utility(r, a.v_price, a.v_value)  # noqa: E731
    for s, hist in data.items():
        lts = [r for h in hist.values() for r in h.values()]
        res["systems"][s] = {
            "utility": boot([utility(r, a.price) for r in lts]),
            "utility_v": boot([uv(r) for r in lts]),
            "spearman_p_dstop_B1": spearman([x["p_before"] for r in lts for x in r if x["entity"] == "B1"],
                                            [x["d_stop"] for r in lts for x in r if x["entity"] == "B1"]),
            "distinct_grid_intents": len({x["grid_i"] for r in lts for x in r if x.get("grid_i") is not None}),
            "mean_t_arrive": float(np.mean([x["t_arrive"] for r in lts for x in r if x["t_arrive"] is not None])),
            "strong_rate": float(np.mean([x["category"] >= STRONG for r in lts for x in r
                                          if x.get("observable", True)])),
            "contacts_per_encounter": float(np.mean([x["contacts"] for r in lts for x in r])),
            "maxpush_per_encounter": float(np.mean([x["max_push"] for r in lts for x in r])),
            "falls": int(sum(x["fell"] for r in lts for x in r)), "trips": int(sum(x["trips"] for r in lts for x in r)),
            "arrival_rate": float(np.mean([x["arrived"] for r in lts for x in r])),
            "d_stop_range": [float(np.min([x["d_stop"] for r in lts for x in r])),
                             float(np.max([x["d_stop"] for r in lts for x in r]))],
        }
        if "risky_B" in hist and "safe" in hist:
            keys = sorted(set(hist["risky_B"]) & set(hist["safe"]))
            res["systems"][s]["belief_effect"] = boot([last_p(hist["risky_B"][i]) - last_p(hist["safe"][i])
                                                      for i in keys])
            late = lambda r: [x["d_stop"] for x in r if x["entity"] == "B1" and x["k"] >= 10]  # noqa: E731
            res["systems"][s]["dstop_effect"] = boot([np.mean(late(hist["risky_B"][i])) - np.mean(late(hist["safe"][i]))
                                                     for i in keys])
    if "CL" in data:
        for s in ("LEARNED", "RESIDUAL", "GRU", "LEARNED_V", "RESIDUAL_V", "GRU_V"):
            if s not in data:
                continue
            pairs = [(data[s][h][i], data["CL"][h][i]) for h in data[s] for i in data[s][h]
                     if i in data["CL"].get(h, {})]
            res["vs_L0"][s] = {
                "utility_diff": boot([utility(x, a.price) - utility(y, a.price) for x, y in pairs]),
                "utility_v_diff": boot([uv(x) - uv(y) for x, y in pairs]),
                "contacts_diff": boot([np.mean([e["contacts"] for e in x]) - np.mean([e["contacts"] for e in y])
                                       for x, y in pairs]),
                "maxpush_diff": boot([np.mean([e["max_push"] for e in x]) - np.mean([e["max_push"] for e in y])
                                      for x, y in pairs]),
                "time_diff_s": boot([np.mean([e["t_arrive"] for e in x if e["t_arrive"] is not None])
                                     - np.mean([e["t_arrive"] for e in y if e["t_arrive"] is not None])
                                     for x, y in pairs])}
    res["q3_vs_FROZEN"] = {}
    if "FROZEN" in data:
        fr = data["FROZEN"].get("risky_B", {})
        late = lambda r: [x for x in b1(r) if x["k"] >= LATE]  # noqa: E731
        for s, hist in data.items():
            if s == "FROZEN" or "risky_B" not in hist:
                continue
            keys = sorted(set(hist["risky_B"]) & set(fr))
            res["q3_vs_FROZEN"][s] = {
                f: boot([np.mean([x[f] for x in late(hist["risky_B"][i])]) - np.mean([x[f] for x in late(fr[i])])
                         for i in keys]) for f in ("max_push", "contacts")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for s, o in res["systems"].items():
        print(f"{s:10} U {o['utility']['estimate']:.2f} UV {o['utility_v']['estimate']:.2f} "
              f"rho {o['spearman_p_dstop_B1']:.2f} n_int {o['distinct_grid_intents']} "
              f"t {o['mean_t_arrive']:.1f}s strong {o['strong_rate']:.3f} "
              f"contacts {o['contacts_per_encounter']:.3f} push {o['maxpush_per_encounter']:.4f} d {o['d_stop_range']} "
              f"falls {o['falls']} belief {(o.get('belief_effect') or {}).get('estimate')}")
    for s, o in res["vs_L0"].items():
        print(s, "vs L0:", {k: (round(v["estimate"], 4), [round(c, 4) for c in v["ci95"]]) for k, v in o.items()})

    for s, o in res["q3_vs_FROZEN"].items():
        print(s, "Q3 vs FROZEN:",
              {k: (round(v["estimate"], 4), [round(c, 4) for c in v["ci95"]]) for k, v in o.items()})


if __name__ == "__main__":
    main()
