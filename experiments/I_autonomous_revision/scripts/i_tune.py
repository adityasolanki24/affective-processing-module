"""Programme I tuning on partition V (MASTER_PREREGISTRATION.md §4). Equal budget K = 16 per tuned parameter.

    python i_tune.py lam --systems <cfg> --system R_dense --out <json>
    python i_tune.py explore --systems <cfg> --lam-from <json> --out <json>

lam: λ ∈ 16 log-uniform values in [1/5000, 1/50] s⁻¹ (sample seed 20261017), on V lifetimes 0–39 of S1, S2, S3 and
     S10 under the weak-coupling rule. Objective (self-supervised, no ground truth): mean log predictive probability
     of each informative encounter's outcome category before the update. Higher is better.
explore: strict keep-away world (systems KA_R*), V lifetimes 100–139 of S1, S3 and S10.
     ν (VOI) ∈ 16 log-uniform values in [0.01, 10]; conf_max (uncertainty) ∈ 16 uniform values in [0.1, 0.95].
     Objective: mean tuning utility J (G's J), subject to avoidable collisions per minute ≤ 1.10 × KA_R's on the
     same lifetimes. If no value qualifies, the one with the fewest avoidable collisions. Then p_random (random
     control) ∈ 16 values in [0.02, 0.6]: the value whose mean probes per lifetime is closest to the selected VOI's
     (budget matching).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i_common import init, run_job  # noqa: E402

SEED = 20261017


def grid_log(lo: float, hi: float, k: int = 16, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    return sorted(float(x) for x in np.exp(rng.uniform(np.log(lo), np.log(hi), k)))


def grid_lin(lo: float, hi: float, k: int = 16, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    return sorted(float(x) for x in rng.uniform(lo, hi, k))


def run(jobs: list[dict], workers: int) -> list[dict]:
    with Pool(workers, initializer=init) as pool:
        return list(pool.imap_unordered(run_job, jobs, chunksize=1))


def lam_mode(a, systems: dict) -> dict:
    base = systems[a.system]
    lams = grid_log(1 / 5000, 1 / 50)
    jobs = [{"spec": {**base, "lam": lam, "label": f"lam={lam:.6g}", "lam_value": lam}, "scenario": sc,
             "experiment": "I1", "partition": "V", "index": i}
            for lam in lams for sc in ("S1", "S2", "S3", "S10") for i in range(40)]
    res = run(jobs, a.workers)
    score = {}
    for lam in lams:
        lp = [x for r in res if r["label"] == f"lam={lam:.6g}" for x in r["pred_logp"]]
        score[lam] = float(np.mean(lp))
    best = max(score, key=score.get)
    return {"system": a.system, "grid": lams, "objective": {str(k): v for k, v in score.items()}, "selected": best}


def summarise(res: list[dict], label: str) -> dict:
    rows = [r for r in res if r["label"] == label]
    t = [r["totals"] for r in rows]
    return {"J": float(np.mean([x["J"] for x in t])),
            "avoidable_per_min": sum(x["avoidable_collisions"] for x in t) / sum(x["minutes"] for x in t),
            "completion": sum(x["goals_completed"] for x in t) / max(1.0, sum(x["goals_issued"] for x in t)),
            "probes": float(np.mean([r["probes"] for r in rows]))}


def explore_mode(a, systems: dict) -> dict:
    lam = json.loads(Path(a.lam_from).read_text())["selected"]
    scen, idx = ("S1", "S3", "S10"), range(100, 140)

    def jobs_for(specs: dict) -> list[dict]:
        return [{"spec": {**s, "lam": lam, "label": name}, "scenario": sc, "experiment": "I2", "partition": "V",
                 "index": i} for name, s in specs.items() for sc in scen for i in idx]

    def with_explore(name: str, ex: dict) -> dict:
        s = json.loads(json.dumps(systems[name]))
        s["keep_away"]["explore"] = {**s["keep_away"]["explore"], **ex}
        return s
    nus, confs = grid_log(0.01, 10.0), grid_lin(0.1, 0.95)
    specs = {"KA_R": systems["KA_R"]}
    specs |= {f"voi:{v:.6g}": with_explore("KA_R_voi", {"nu": v}) for v in nus}
    specs |= {f"unc:{c:.6g}": with_explore("KA_R_unc", {"conf_max": c}) for c in confs}
    res = run(jobs_for(specs), a.workers)
    ref = summarise(res, "KA_R")
    cap = 1.10 * ref["avoidable_per_min"]

    def select(prefix: str, values: list[float]) -> tuple[float, dict]:
        s = {v: summarise(res, f"{prefix}:{v:.6g}") for v in values}
        ok = [v for v in values if s[v]["avoidable_per_min"] <= cap]
        best = max(ok, key=lambda v: s[v]["J"]) if ok else min(values, key=lambda v: s[v]["avoidable_per_min"])
        return best, {str(v): s[v] for v in values}
    nu, nu_s = select("voi", nus)
    conf, conf_s = select("unc", confs)
    target = nu_s[str(nu)]["probes"]
    ps = grid_lin(0.02, 0.6)
    res2 = run(jobs_for({f"rand:{p:.6g}": with_explore("KA_R_rand", {"p_random": p}) for p in ps}), a.workers)
    rand_s = {str(p): summarise(res2, f"rand:{p:.6g}") for p in ps}
    p_sel = min(ps, key=lambda p: abs(rand_s[str(p)]["probes"] - target))
    return {"lam": lam, "reference_KA_R": ref, "collision_cap": cap,
            "nu": {"grid": nus, "results": nu_s, "selected": nu},
            "conf_max": {"grid": confs, "results": conf_s, "selected": conf},
            "p_random": {"grid": ps, "results": rand_s, "selected": p_sel, "target_probes": target}}


def random_mode(a, systems: dict) -> dict:
    """Re-selection of p_random over [p_lo, p_hi] (DEVIATIONS.md D1), against the selected VOI's probe budget."""
    prev = json.loads(Path(a.explore_from).read_text())
    lam, target = prev["lam"], prev["p_random"]["target_probes"]
    ps = grid_lin(a.p_lo, a.p_hi)
    jobs = []
    for p in ps:
        s = json.loads(json.dumps(systems["KA_R_rand"]))
        s["keep_away"]["explore"] = {**s["keep_away"]["explore"], "p_random": p}
        jobs += [{"spec": {**s, "lam": lam, "label": f"rand:{p:.6g}"}, "scenario": sc, "experiment": "I2",
                  "partition": "V", "index": i} for sc in ("S1", "S3", "S10") for i in range(100, 140)]
    res = run(jobs, a.workers)
    rand_s = {str(p): summarise(res, f"rand:{p:.6g}") for p in ps}
    p_sel = min(ps, key=lambda p: abs(rand_s[str(p)]["probes"] - target))
    return {"lam": lam, "p_random": {"grid": ps, "range": [a.p_lo, a.p_hi], "results": rand_s, "selected": p_sel,
                                     "target_probes": target}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("lam", "explore", "random"))
    ap.add_argument("--systems", type=Path, required=True)
    ap.add_argument("--system", default="R_dense")
    ap.add_argument("--lam-from", default="")
    ap.add_argument("--explore-from", default="")
    ap.add_argument("--p-lo", type=float, default=0.02)
    ap.add_argument("--p-hi", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    start = time.time()
    systems = json.loads(a.systems.read_text())["systems"]
    out = {"lam": lam_mode, "explore": explore_mode, "random": random_mode}[a.mode](a, systems)
    out["wall_s"] = round(time.time() - start, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("selected", "lam", "wall_s")}),
          {k: out[k]["selected"] for k in ("nu", "conf_max", "p_random") if k in out})


if __name__ == "__main__":
    main()
