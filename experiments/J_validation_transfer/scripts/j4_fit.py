"""Fit NERVA type models from J3-T standing encounters (no labels read by EM). Used by J4 and J5.

    python j4_fit.py --rows <T_a.jsonl> <T_b.jsonl> --out-dir <dir>

Steps:
1. **acc8 edges:** pooled quantiles of peak acceleration deviation over all J3-T rows (all types and distances).
2. **J4 model (unconditional, neutral 0.4 m).** Rows at 0.4 m are grouped into entities of 12 encounters. Each group
   holds encounters of one entity, as a robot observes repeated meetings with a tagged identity. The grouping uses only
   which entity was met, never the entity's type label as an EM input.
   EM with K = 1..4 by BIC on acc8 counts. The rate per type is the self-calibrated detector's frequency.
3. **J5 model (conditioned on the robot's own stopping distance).** Entities of 12 encounters (4 at each of 0.2,
   0.4 and 0.6 m); `fit_conditional` on counts[entity, distance band, acc8]. K by BIC.
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
from apm_revise.belief import fit_types  # noqa: E402
from apm_transfer.conditioned import bic_conditional, fit_conditional  # noqa: E402

DISTANCES = (0.2, 0.4, 0.6)
UNIT = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for p in a.rows for x in p.read_text(encoding="utf-8").splitlines()]
    acc = np.array([r["evidence"][1] for r in rows])
    edges = np.unique(np.quantile(acc, np.linspace(0, 1, 9)[1:-1])).tolist()
    code = lambda r: int(np.searchsorted(edges, r["evidence"][1], side="right"))  # noqa: E731
    by: dict = defaultdict(list)
    for r in rows:
        by[(r["type"], r["distance"])].append(r)
    rng = np.random.default_rng(20261018)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "J4_edges.json").write_text(json.dumps({"acc8": edges, "source": "J3-T pooled quantiles"}),
                                             encoding="utf-8")
    # J4: unconditional at 0.4 m
    counts, enc, det = [], [], []
    for (_t, d), rs in by.items():
        if d != 0.4:
            continue
        idx = rng.permutation(len(rs))
        for k in range(0, len(idx) - UNIT + 1, UNIT):
            c = np.zeros(8)
            for i in idx[k:k + UNIT]:
                c[code(rs[i])] += 1
            counts.append(c)
            enc.append(float(UNIT))
            det.append(float(sum(bool(rs[i]["evidence"][9]) for i in idx[k:k + UNIT])))
    fits = {K: fit_types(np.stack(counts), np.array(enc), np.array(det), K, "acc8", restarts=10, seed=K)
            for K in (1, 2, 3, 4)}
    best = min(fits, key=lambda k: fits[k].bic())
    (a.out_dir / "J4_types.json").write_text(json.dumps(fits[best].to_dict(), indent=1), encoding="utf-8")
    print("J4 unconditional: units", len(counts), "BIC", {k: round(f.bic(), 1) for k, f in fits.items()}, "K*", best,
          "pi", np.round(fits[best].pi, 3), "detect rate", np.round(fits[best].r, 3))
    # J5: conditioned on stopping distance
    types = sorted({t for t, _ in by})
    counts3, enc3, det3 = [], [], []
    for t in types:
        pools = {d: list(rng.permutation(len(by[(t, d)]))) for d in DISTANCES}
        while all(len(pools[d]) >= UNIT // 3 for d in DISTANCES):
            c = np.zeros((3, 8))
            n_det = 0
            for b, d in enumerate(DISTANCES):
                for _ in range(UNIT // 3):
                    r = by[(t, d)][pools[d].pop()]
                    c[b, code(r)] += 1
                    n_det += bool(r["evidence"][9])
            counts3.append(c)
            enc3.append(float(UNIT))
            det3.append(float(n_det))
    fits3 = {K: fit_conditional(np.stack(counts3), np.array(enc3), np.array(det3), K, "acc8_dist", restarts=10,
                                seed=K) for K in (1, 2, 3, 4)}
    bics = {K: bic_conditional(m, 3) for K, m in fits3.items()}
    best3 = min(bics, key=bics.get)
    (a.out_dir / "J5_types_distance.json").write_text(json.dumps(fits3[best3].to_dict(), indent=1), encoding="utf-8")
    print("J5 distance-conditioned: units", len(counts3), "BIC", {k: round(v, 1) for k, v in bics.items()}, "K*",
          best3, "pi", np.round(fits3[best3].pi, 3), "detect rate", np.round(fits3[best3].r, 3))


if __name__ == "__main__":
    main()
