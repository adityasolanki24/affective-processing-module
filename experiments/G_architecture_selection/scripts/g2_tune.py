"""G2 tuning and validation selection (partition V only; prereg/G2_ADDENDUM.md fixes K, N and the scenarios).

Stage A (tuning): every parametric memory (M1 γ, M2 W, M4 hazard, M5 q_low/q_high/switch) gets K Latin-hypercube
configurations; parameter-free memories (M0, M3) one. Each configuration runs in closed loop with the G1-selected
regulator on G2-V lifetimes 0…N−1 of the tuning scenarios; objective = per-encounter Brier on observable encounters
(lower is better); one selected configuration per memory.
Stage B (validation selection): every selected memory, plus M0 with both evidence models, runs on G2-V lifetimes
100…100+NV−1 of all scenarios; M* = the memory with the fewest state variables per identity that meets criteria
(i)–(iii) on validation (point estimates; adaptation medians ≤ 6, collision rate within +10% of M0_all, false blame
within ±0.03); ties → lower Brier. Written to configs/G2_selected.json.

    python scripts/g2_tune.py --regulator configs/G1_selected.json --k 16 --n 6 --nv 12 --workers 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g2_analyze import encounter_brier, load, summarise  # noqa: E402
from g2_run import run_jobs  # noqa: E402

from apm_research.provenance import environment, write_json  # noqa: E402
from apm_select.tuning import lhs  # noqa: E402

SPACES = {"M1": {"gamma": ("lin", 0.5, 0.99)}, "M2": {"window": ("lin", 3.0, 40.0)},
          "M4": {"hazard": ("log", 0.005, 0.3)},
          "M5": {"q_low": ("log", 0.005, 0.1), "q_high": ("lin", 0.1, 0.6), "switch": ("log", 0.005, 0.3)}}
TUNE_SCENARIOS = ("S1", "S2", "S3", "S5")
SEED = 20261012


def regulator_spec(path: Path) -> dict:
    sel = json.loads(path.read_text())
    if "regulator_for_G2" in sel:
        return sel["regulator_for_G2"]
    best = sel["systems"][sel["best_simple"]]
    return {k: v for k, v in best.items() if k not in ("config_id", "label")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regulator", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--nv", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G2"))
    args = ap.parse_args()
    start = time.time()
    reg = regulator_spec(args.regulator)
    cfgs = [("M0", {}, "M0:0"), ("M3", {}, "M3:0")]
    for m, space in SPACES.items():
        offset = sorted(SPACES).index(m)
        for i, v in enumerate(lhs(space, args.k, SEED + 100 * offset)):
            cfgs.append((m, v, f"{m}:{i}"))
    jobs = [{"spec": {**reg, "label": m, "config_id": cid, "memory": {"model": m, "params": p, "evidence": "exposed"}},
             "scenario": sc, "partition": "V", "index": i}
            for m, p, cid in cfgs for sc in TUNE_SCENARIOS for i in range(args.n)]
    (args.out / "tune").mkdir(parents=True, exist_ok=True)
    res = run_jobs(jobs, args.workers, args.out / "tune" / "lifetimes.jsonl")
    by = defaultdict(list)
    for r in res:
        by[r["config_id"]].append(r)
    table = {cid: {"model": m, "params": p, **encounter_brier(by[cid])} for m, p, cid in cfgs}
    chosen = {}
    for m in ("M0", "M3", *SPACES):
        rows = [(cid, t) for cid, t in table.items() if t["model"] == m]
        cid, t = min(rows, key=lambda x: x[1]["brier"])
        chosen[m] = {"model": m, "params": t["params"], "config_id": cid, "brier": t["brier"]}
    # Stage B: validation selection over all scenarios
    systems = {"M0_all": {"model": "M0", "params": {}, "evidence": "all"},
               "M0_exp": {"model": "M0", "params": {}, "evidence": "exposed"}}
    systems |= {m: {"model": m, "params": c["params"], "evidence": "exposed"} for m, c in chosen.items() if m != "M0"}
    vjobs = [{"spec": {**reg, "label": name, "config_id": name, "memory": mem}, "scenario": sc, "partition": "V",
              "index": 100 + i}
             for name, mem in systems.items() for sc in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
             for i in range(args.nv)]
    (args.out / "Vsel").mkdir(parents=True, exist_ok=True)
    run_jobs(vjobs, args.workers, args.out / "Vsel" / "lifetimes.jsonl")
    summ = summarise(load(args.out / "Vsel" / "lifetimes.jsonl"))
    ref = summ["M0_all"]
    ref_col = np.mean([b["collision_rate"] for b in ref["behaviour"].values()])
    verdict = {}
    for name, s in summ.items():
        col = np.mean([b["collision_rate"] for b in s["behaviour"].values()])
        a1, a2 = s.get("adaptation_S1", {}).get("median"), s.get("adaptation_S2", {}).get("median")
        fb = (s.get("false_blame") or {}).get("estimate")
        ok = {"i": a1 is not None and a2 is not None and a1 <= 6 and a2 <= 6,
              "ii": col <= 1.10 * ref_col, "iii": fb is not None and abs(fb) < 0.03}
        verdict[name] = {"criteria": ok, "all": all(ok.values()), "state_size": s["state_size"],
                         "brier": s["brier"]["brier"], "collision_rate_mean": col, "adaptation": [a1, a2],
                         "false_blame": fb}
    passing = [n for n, v in verdict.items() if v["all"] and n != "M0_all"]
    star = min(passing, key=lambda n: (verdict[n]["state_size"], verdict[n]["brier"])) if passing else None
    write_json(HERE / "results" / "G2" / "tuning_table.json", {"table": table, "chosen": chosen,
                                                                "validation": verdict, "environment": environment(),
                                                                "wall_s": round(time.time() - start, 1)})
    out = {"rule": "fewest state variables per identity among memories meeting (i)-(iii) on validation; ties: Brier",
           "regulator": reg, "candidate": star, "validation": verdict,
           "systems": {n: {**reg, "memory": m} for n, m in systems.items()}}
    (HERE / "configs" / "G2_selected.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"candidate": star, "validation": verdict}, indent=1, default=str))


if __name__ == "__main__":
    main()
