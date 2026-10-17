"""Fit person-type models (EM, K by BIC) from the robot's own experience logs. No ground truth is read.

Units: one identity within one lifetime. Counts: reliable informative encounters per outcome category, weighted by
identity confidence. Collision rate per type uses all reliable encounters, censored ones included.

    python experiments/I_autonomous_revision/scripts/i_fit.py --runs runs/I/I0/T/lifetimes.jsonl \
        --out experiments/I_autonomous_revision/configs/types_T.json
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
from apm_revise.evidence import CODINGS, RELIABLE_CONF  # noqa: E402

COL = {"dense": 8, "sparse": 9}  # encounter-row index of each coding's category


def units(paths: list[Path], coding: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = CODINGS[coding]
    acc: dict[tuple, list] = defaultdict(lambda: [np.zeros(C), 0.0, 0.0])
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key0 = (r["experiment"], r["partition"], r["scenario"], r["index"], r["label"])
            for row in r["encounters"]:
                ident, conf, coll, cat = row[2], row[3], row[4], row[COL[coding]]
                if ident is None or conf < RELIABLE_CONF:
                    continue
                u = acc[(*key0, ident)]
                u[1] += 1.0
                u[2] += float(coll)
                if cat is not None:
                    u[0][cat] += conf
    vals = list(acc.values())
    return np.stack([v[0] for v in vals]), np.array([v[1] for v in vals]), np.array([v[2] for v in vals])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kmax", type=int, default=3)
    a = ap.parse_args()
    out = {}
    for coding in CODINGS:
        counts, enc, coll = units(a.runs, coding)
        fits = {K: fit_types(counts, enc, coll, K, coding, restarts=10, seed=K) for K in range(1, a.kmax + 1)}
        best = min(fits, key=lambda K: fits[K].bic())
        out[coding] = {"selected_K": best, "bic": {K: f.bic() for K, f in fits.items()},
                       "model": fits[best].to_dict(), "all": {K: f.to_dict() for K, f in fits.items()},
                       "units": int(len(enc))}
        m = fits[best]
        print(coding, "units", len(enc), "BIC", {K: round(f.bic(), 1) for K, f in fits.items()}, "K*", best,
              "pi", np.round(m.pi, 3), "r", np.round(m.r, 4))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for coding in CODINGS:  # one file per coding (BIC-selected K, and K = 2), as `TypeModel.from_dict` reads
        a.out.with_name(a.out.stem + f"_{coding}.json").write_text(json.dumps(out[coding]["model"], indent=1),
                                                                    encoding="utf-8")
        a.out.with_name(a.out.stem + f"_{coding}_K2.json").write_text(json.dumps(out[coding]["all"][2], indent=1),
                                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
