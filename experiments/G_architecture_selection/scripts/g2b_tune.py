"""G2b (prereg/G2b_ADDENDUM.md): revision under selective feedback — evidence-free time decay and safe probing.

Stage A: M1t (time decay τ_s) gets K Latin-hypercube values on G2-V lifetimes 0…N−1 of S1, S2, S3, S5 (Brier on
observable encounters, as G2). Stage B: the candidates below on G2-V lifetimes 200…200+NV−1 of S1–S9; M\\* = among
{M1t, M1t_probeU, M1_probeU} the one meeting G2's criteria (i)–(iii) on validation, ties → lower Brier, then fewer
probes; written to configs/G2b_selected.json together with the X systems.

    python scripts/g2b_tune.py --k 16 --n 6 --nv 12 --workers 90
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
from g2_systems import PROBE_R, PROBE_U  # noqa: E402

from apm_research.provenance import environment, write_json  # noqa: E402
from apm_select.tuning import lhs  # noqa: E402

SPACE = {"tau_s": ("log", 60.0, 3000.0)}
TUNE_SCENARIOS = ("S1", "S2", "S3", "S5")
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
SEED = 20261013
CANDIDATES = ("M1t", "M1t_probeU", "M1_probeU")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--nv", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G2b"))
    args = ap.parse_args()
    start = time.time()
    g2 = json.loads((HERE / "configs" / "G2_selected.json").read_text())
    reg = g2["regulator"]
    m1 = g2["systems"]["M1"]["memory"]
    cfgs = [(v, f"M1t:{i}") for i, v in enumerate(lhs(SPACE, args.k, SEED))]
    jobs = [{"spec": {**reg, "label": "M1t", "config_id": cid,
                      "memory": {"model": "M1t", "params": p, "evidence": "exposed"}},
             "scenario": sc, "partition": "V", "index": i}
            for p, cid in cfgs for sc in TUNE_SCENARIOS for i in range(args.n)]
    (args.out / "tune").mkdir(parents=True, exist_ok=True)
    res = run_jobs(jobs, args.workers, args.out / "tune" / "lifetimes.jsonl")
    by = defaultdict(list)
    for r in res:
        by[r["config_id"]].append(r)
    table = {cid: {"params": p, **encounter_brier(by[cid])} for p, cid in cfgs}
    best_cid = min(table, key=lambda c: table[c]["brier"])
    m1t = {"model": "M1t", "params": table[best_cid]["params"], "evidence": "exposed"}
    systems = {"M0_all": {**reg, "memory": {"model": "M0", "evidence": "all"}},
               "M1": {**reg, "memory": m1},
               "M1t": {**reg, "memory": m1t},
               "M1t_probeU": {**reg, "memory": {**m1t, "probe": PROBE_U}},
               "M1_probeU": {**reg, "memory": {**m1, "probe": PROBE_U}},
               "M1t_probeR": {**reg, "memory": {**m1t, "probe": PROBE_R}}}
    vjobs = [{"spec": {**s, "label": name, "config_id": name}, "scenario": sc, "partition": "V", "index": 200 + i}
             for name, s in systems.items() for sc in SCENARIOS for i in range(args.nv)]
    (args.out / "Vsel").mkdir(parents=True, exist_ok=True)
    vres = run_jobs(vjobs, args.workers, args.out / "Vsel" / "lifetimes.jsonl")
    probes = defaultdict(float)
    for r in vres:
        probes[r["label"]] += r.get("probes", 0)
    summ = summarise(load(args.out / "Vsel" / "lifetimes.jsonl"))
    ref_col = np.mean([b["collision_rate"] for b in summ["M0_all"]["behaviour"].values()])
    verdict = {}
    for name, s in summ.items():
        col = float(np.mean([b["collision_rate"] for b in s["behaviour"].values()]))
        a1, a2 = s["adaptation_S1"]["median"], s["adaptation_S2"]["median"]
        fb = s["false_blame"]["estimate"]
        ok = {"i": bool(a1 <= 6 and a2 <= 6), "ii": bool(col <= 1.10 * ref_col), "iii": bool(abs(fb) < 0.03)}
        verdict[name] = {"criteria": ok, "all": all(ok.values()), "brier": float(s["brier"]["brier"]),
                         "collision_rate_mean": col, "adaptation": [a1, a2], "false_blame": fb,
                         "probes": probes[name], "no_evidence_share": s["selective"]["no_evidence_share"]}
    passing = [n for n in CANDIDATES if verdict[n]["all"]]
    star = min(passing, key=lambda n: (verdict[n]["brier"], verdict[n]["probes"])) if passing else None
    write_json(HERE / "results" / "G2" / "g2b_tuning.json", {"table": table, "chosen_M1t": m1t, "validation": verdict,
                                                             "environment": environment(),
                                                             "wall_s": round(time.time() - start, 1)})
    out = {"rule": "among M1t, M1t_probeU, M1_probeU: meets G2 (i)-(iii) on validation; ties: Brier, then probes",
           "candidate": star, "validation": verdict, "systems": systems, "regulator": reg}
    (HERE / "configs" / "G2b_selected.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"candidate": star, "validation": verdict}, indent=1, default=str))


if __name__ == "__main__":
    main()
