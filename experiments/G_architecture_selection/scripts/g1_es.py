"""G1 R5: compact learned scalar risk regulator, trained by evolution strategies with the tuning budget of one
family (K × N lifetime evaluations, prereg/G1_ADDENDUM.md).

θ = (w ∈ R^8, log λc, λs); m(t) = exp(clip(w·φ(t), −1.5, 2)) (apm_select.regulators.LearnedScalar). Antithetic
Gaussian ES (Salimans et al. 2017) with centred-rank fitness shaping and Adam on θ. Training lifetimes: G1-T, a fresh
set every generation, shared by the whole population (common random numbers). Start: w = 0, λc = 1, λs = 2
(m ≡ 1, i.e. programme F's tuned B1). Budget accounting: generations × population × lifetimes per candidate plus
the final validation evaluation (G1-V 0…N−1, the same lifetimes the other families are tuned on) ≤ K × N.
Fitness = mean tuning utility J.

    python scripts/g1_es.py --k 48 --n 30 --workers 32
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g1_tune import summarise  # noqa: E402
from g_common import run_jobs  # noqa: E402

from apm_research.provenance import environment, write_json  # noqa: E402

POP_PAIRS, PER_CANDIDATE, SIGMA, LR, ES_SEED = 8, 6, 0.25, 0.08, 20261012


def to_spec(theta: np.ndarray, cid: str) -> dict:
    return {"kind": "regulated", "regulator": "learned", "family": "R5", "label": "R5", "config_id": cid,
            "lam_c": float(math.exp(theta[8])), "lam_s": float(max(0.0, theta[9])),
            "params": {"w": [float(x) for x in theta[:8]]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G1/es"))
    args = ap.parse_args()
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    budget = args.k * args.n
    per_gen = 2 * POP_PAIRS * PER_CANDIDATE
    gens = (budget - args.n) // per_gen
    rng = np.random.default_rng(ES_SEED)
    theta = np.zeros(10)
    theta[9] = 2.0
    m, v = np.zeros(10), np.zeros(10)
    curve, next_index, used = [], 0, 0
    for g in range(gens):
        eps = rng.standard_normal((POP_PAIRS, 10))
        cands = [theta + SIGMA * e for e in eps] + [theta - SIGMA * e for e in eps]
        idx = list(range(next_index, next_index + PER_CANDIDATE))
        next_index += PER_CANDIDATE
        jobs = [{"spec": to_spec(c, f"g{g}c{j}"), "experiment": "G1", "partition": "T", "index": i,
                 "n_episodes": args.episodes} for j, c in enumerate(cands) for i in idx]
        res = run_jobs(jobs, args.workers)
        used += len(jobs)
        fit = np.array([np.mean([r["totals"]["J"] for r in res if r["config_id"] == f"g{g}c{j}"])
                        for j in range(len(cands))])
        ranks = np.empty(len(fit))
        ranks[np.argsort(fit)] = np.arange(len(fit))
        shaped = ranks / (len(fit) - 1) - 0.5
        grad = (shaped[:POP_PAIRS] - shaped[POP_PAIRS:]) @ eps / (2 * POP_PAIRS * SIGMA)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta = theta + LR * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
        curve.append({"generation": g, "fitness_mean": float(fit.mean()), "fitness_max": float(fit.max()),
                      "theta": theta.round(4).tolist()})
        print(g, round(float(fit.mean()), 2), round(float(fit.max()), 2), flush=True)
    final = to_spec(theta, "R5:final")
    val = run_jobs([{"spec": final, "experiment": "G1", "partition": "V", "index": i, "n_episodes": args.episodes}
                    for i in range(args.n)], args.workers, args.out / "validation.jsonl")
    used += len(val)
    assert used <= budget, (used, budget)
    out = {"selected": {**final, **summarise(val)}, "curve": curve, "lifetime_evaluations": used, "budget": budget,
           "generations": gens, "environment": environment(), "wall_s": round(time.time() - start, 1)}
    write_json(HERE / "results" / "G1" / "es_R5.json", out)
    print(json.dumps({k: out["selected"][k] for k in ("J_mean", "collision_rate", "completion", "lam_c", "lam_s")}))


if __name__ == "__main__":
    main()
