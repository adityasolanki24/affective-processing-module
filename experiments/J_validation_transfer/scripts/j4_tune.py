"""J4: select the type memory's forgetting rate λ (per encounter) on V lifetimes, self-supervised (no labels).

    python j4_tune.py --bank <X_bank.jsonl> --rows 0:300 --types-model <J4_types.json> --edges <J4_edges.json>
        --n 100 --out <json>

Objective: the mean log predictive probability of each encounter's acc8 category under the identity's belief just
before the update, summed over every history and lifetime (`type_kind`, as in J4). The grid is fixed:
λ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from j4_lifetimes import code, load_bank  # noqa: E402

from apm_revise.belief import TypeModel  # noqa: E402
from apm_transfer.nerva_memory import HISTORIES, N_ENCOUNTERS, ObsX, TypeStateModel, lifetime_types  # noqa: E402

GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def score(lam: float, bank: dict, edges: list, model: TypeModel, n: int, seed: int) -> float:
    lp = []
    for h in HISTORIES:
        for i in range(n):
            rng = np.random.default_rng([seed, i])
            seq = lifetime_types(h, np.random.default_rng([seed, i, 1]))
            draws = rng.random(N_ENCOUNTERS)
            m = TypeStateModel(model, lam, kind_prior=True)
            for k, (ident, zone, etype) in enumerate(seq):
                rows = bank[etype]
                ev = rows[int(draws[k] * len(rows))]["evidence"]
                cat = code(ev, edges)
                b = m._at(ident) if ident in m.b else m._prior(ident[0])
                lp.append(math.log(max(1e-12, float(b @ model.E[:, cat]))))
                m.observe(ObsX(ident, ident[0], zone, float(bool(ev[9])), float(ev[1]), cat))
    return float(np.mean(lp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--rows", default="0:300")
    ap.add_argument("--types-model", type=Path, required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.rows.split(":"))
    bank = load_bank(a.bank, (lo, hi))
    model = TypeModel.from_dict(json.loads(a.types_model.read_text()))
    edges = json.loads(a.edges.read_text())["acc8"]
    scores = {lam: score(lam, bank, edges, model, a.n, a.seed) for lam in GRID}
    best = max(scores, key=scores.get)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"grid": list(GRID), "objective": {str(k): v for k, v in scores.items()},
                                 "selected": best}, indent=1), encoding="utf-8")
    print({k: round(v, 4) for k, v in scores.items()}, "selected", best)


if __name__ == "__main__":
    main()
