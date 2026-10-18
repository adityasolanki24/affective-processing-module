"""K3b: frozen-task confirmation of the decision-theoretic mapping against L0 (PROTOCOL.md §5).

    python k3b_analyze.py --runs out/K/K3b_*.jsonl --out results/K3b_X.json

**Realised task cost** per encounter, with the frozen task values:

    approach time + 4 s × (windows − 1) + 20 s × strong perturbations felt + 12.6 s × (1 − task success)

A strong perturbation felt is an observable window with acc8 ≥ 6. Task success is `Task.p_success` at the measured
stopping distance. Each lifetime contributes its mean, and DT − L0 is taken over paired seeds, with bootstrap 95% CIs.

**Promotion rule per task:**
1. cost difference: 95% upper bound < 0;
2. contacts per encounter difference: upper bound ≤ +0.05;
3. 0 falls.

DT replaces L0 if this holds in ≥ 3 of 4 tasks. Expressive differentiation (risky − safe B1 stopping distance,
encounters ≥ 10) is reported per task, and is never a criterion.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STRONG = 6


def boot(x: list[float], n: int = 4000) -> dict:
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    rng = np.random.default_rng(0)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return {"estimate": float(a.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(a))}


def enc_cost(x: dict, task: dict) -> float:
    strong = sum(c >= STRONG for c in x["imu_cats"])
    return ((x["t_arrive"] or 45.0) + 4.0 * (x["windows_done"] - 1) + task["price_s"] * strong
            + task["retry_s"] * (1.0 - x["task_success"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--tasks", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "K_tasks.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tasks = json.loads(a.tasks.read_text())["tasks"]
    d: dict = defaultdict(lambda: defaultdict(dict))
    for p in a.runs:
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            d[r["task"]][(r["system"], r["history"])][r["index"]] = r["rows"]
    res: dict = {"tasks": {}}
    passed = 0
    for t, runs in sorted(d.items()):
        spec = tasks[t]
        o: dict = {}
        for s in ("L0", "DT"):
            rows = [x for h in ("risky_B", "safe") for lt in runs[(s, h)].values() for x in lt]
            o[s] = {"cost": float(np.mean([enc_cost(x, spec) for x in rows])),
                    "contacts": float(np.mean([x["contacts"] for x in rows])),
                    "task_success": float(np.mean([x["task_success"] for x in rows])),
                    "approach_s": float(np.mean([x["t_arrive"] for x in rows if x["t_arrive"] is not None])),
                    "falls": int(sum(x["fell"] for x in rows)), "trips": int(sum(x["trips"] > 0 for x in rows))}
            late = {h: {i: [x["d_stop"] for x in lt if x["entity"] == "B1" and x["k"] >= 10]
                        for i, lt in runs[(s, h)].items()} for h in ("risky_B", "safe")}
            keys = sorted(set(late["risky_B"]) & set(late["safe"]))
            o[s]["expressive_differentiation_m"] = boot([np.mean(late["risky_B"][i] or [np.nan])
                                                         - np.mean(late["safe"][i] or [np.nan]) for i in keys])
        diffs: dict = {}
        for name, f in (("cost", lambda x, sp=spec: enc_cost(x, sp)), ("contacts", lambda x: x["contacts"])):
            vals = []
            for h in ("risky_B", "safe"):
                for i in sorted(set(runs[("DT", h)]) & set(runs[("L0", h)])):
                    vals.append(np.mean([f(x) for x in runs[("DT", h)][i]]) - np.mean([f(x) for x in
                                                                                   runs[("L0", h)][i]]))
            diffs[name] = boot(vals)
        ok = (diffs["cost"]["ci95"][1] < 0 and diffs["contacts"]["ci95"][1] <= 0.05 and o["DT"]["falls"] == 0)
        passed += int(ok)
        res["tasks"][t] = {"systems": o, "DT_minus_L0": diffs, "DT_promoted_for_task": ok}
        print(f"{t:10} cost L0 {o['L0']['cost']:.1f} DT {o['DT']['cost']:.1f} diff {diffs['cost']['estimate']:+.2f} "
              f"{np.round(diffs['cost']['ci95'], 2)} contacts diff {diffs['contacts']['estimate']:+.3f} "
              f"{np.round(diffs['contacts']['ci95'], 3)} success L0 {o['L0']['task_success']:.2f} DT "
              f"{o['DT']['task_success']:.2f} diffm L0 {o['L0']['expressive_differentiation_m']['estimate']:.3f} DT "
              f"{o['DT']['expressive_differentiation_m']['estimate']:.3f} falls {o['DT']['falls']} -> {ok}")
    res["tasks_passed"] = passed
    res["DT_replaces_L0"] = passed >= 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("tasks passed", passed, "of", len(d), "-> DT replaces L0:", res["DT_replaces_L0"])


if __name__ == "__main__":
    main()
