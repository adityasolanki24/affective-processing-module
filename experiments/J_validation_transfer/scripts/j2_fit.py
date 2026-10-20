"""J2: fit EU-T (unconditional), EA (action-conditioned) and EC (context-conditioned) person-type models from J2-T.

J2-T is nominal S3 driven by A0, B0, a cautious rule and R1. No labels are read. K ∈ {1, 2, 3} is chosen by BIC for
each model.

    python j2_fit.py --runs <J2-T lifetimes.jsonl> --out-dir <dir>

Units: one identity within one lifetime and driver. Counts are confidence-weighted reliable informative encounters,
by (band, dense category). Collision rate per type is over all reliable encounters.
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
from apm_transfer.conditioned import BANDS, bic_conditional, fit_conditional  # noqa: E402

DENSE, COLL, ACT, CTX = 8, 4, 13, 14  # encounter-row columns (j_common)
C = 8


def units(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    acc: dict = defaultdict(lambda: {"dense": np.zeros((1, C)), "dense_action": np.zeros((2, C)),
                                     "dense_context": np.zeros((4, C)), "n": 0.0, "k": 0.0})
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for e in r["encounters"]:
            if e[2] is None or e[3] < 0.5:
                continue
            u = acc[(r["label"], r["scenario"], r["index"], e[2])]
            u["n"] += 1.0
            u["k"] += float(e[COLL])
            cat = e[DENSE]
            if cat is None:
                continue
            u["dense"][0, cat] += e[3]
            u["dense_action"][e[ACT] or 0, cat] += e[3]
            u["dense_context"][e[CTX] or 0, cat] += e[3]
    vals = list(acc.values())
    return {k: np.stack([v[k] for v in vals]) for k in ("dense", "dense_action", "dense_context")}, \
        np.array([v["n"] for v in vals]), np.array([v["k"] for v in vals])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    counts, enc, coll = units(a.runs)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for coding, name in (("dense", "EU-T"), ("dense_action", "EA"), ("dense_context", "EC")):
        fits, bics = {}, {}
        for K in (1, 2, 3):
            if coding == "dense":
                m = fit_types(counts[coding][:, 0, :], enc, coll, K, "dense", restarts=10, seed=K)
                bics[K] = m.bic()
            else:
                m = fit_conditional(counts[coding], enc, coll, K, coding, restarts=10, seed=K)
                bics[K] = bic_conditional(m, BANDS[coding])
            fits[K] = m
        best = min(bics, key=bics.get)
        m = fits[best]
        (a.out_dir / f"types_J2T_{name}.json").write_text(json.dumps(m.to_dict(), indent=1), encoding="utf-8")
        summary[name] = {"coding": coding, "bic": bics, "K": best, "pi": m.pi.tolist(), "r": m.r.tolist(),
                         "units": int(len(enc))}
        print(name, "K*", best, "BIC", {k: round(v, 1) for k, v in bics.items()}, "pi", np.round(m.pi, 3),
              "r", np.round(m.r, 4))
    (a.out_dir / "types_J2T_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
