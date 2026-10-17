"""J3: how much does one NERVA encounter tell the robot about the entity's type, by evidence coding?

    python j3_info.py --rows <encounters.jsonl> [--units 6] --out <json>

Codings, all computed from robot-observable evidence:
- `fall`: fell or not;
- `binary`: perturbation detected (gyro exceeded 0.35 rad/s at least once; fixed a priori);
- `gyro8`, `acc8`, `stab8`: 8 quantile bins of peak gyro, peak horizontal acceleration deviation and stabilisation
  time. Bin edges are quantiles of the pooled rows, without labels;
- `gyro_stab`: 4 gyro bins × 3 stabilisation bins.

Per stopping distance and per pair of types: KL divergence in both directions (nats per encounter, add-½ smoothing)
and AUROC of each continuous feature. Types are evaluation-only.

EM discovery: rows are grouped into synthetic entities of `--units` encounters of one type at one distance, as in a
lifetime. EM with BIC (K = 1..4) on the `gyro8` coding, without labels. The agreement of the discovered types with the
true types is reported (evaluation only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm.learning.evaluation import auroc  # noqa: E402
from apm_revise.belief import fit_types  # noqa: E402

FEATURES = {"gyro": 0, "acc": 1, "stab": 2}


def coder(rows: list[dict]) -> dict:
    ev = np.array([r["evidence"][:3] for r in rows], float)
    edges = {f: np.quantile(ev[:, i], np.linspace(0, 1, 9)[1:-1]) for f, i in FEATURES.items()}
    e4 = np.quantile(ev[:, 0], [0.25, 0.5, 0.75])
    s3 = np.quantile(ev[:, 2], [1 / 3, 2 / 3])

    def code(r: dict, scheme: str) -> int:
        e = r["evidence"]
        if scheme == "fall":
            return int(bool(e[6]))
        if scheme == "binary":
            return int(e[2] > 0)
        if scheme == "gyro_stab":
            return int(np.searchsorted(e4, e[0], side="right")) * 3 + int(np.searchsorted(s3, e[2], side="right"))
        f = scheme[:-1]
        return int(np.searchsorted(edges[f], e[FEATURES[f]], side="right"))
    return {"code": code, "sizes": {"fall": 2, "binary": 2, "gyro8": 8, "acc8": 8, "stab8": 8, "gyro_stab": 12},
            "edges": {k: v.tolist() for k, v in edges.items()}}


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--units", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.rows.read_text(encoding="utf-8").splitlines()]
    C = coder(rows)
    res: dict = {"edges": C["edges"], "by_distance": {}, "summary": {}}
    by = defaultdict(list)
    for r in rows:
        by[(r["distance"], r["type"])].append(r)
    dists = sorted({r["distance"] for r in rows})
    types = sorted({r["type"] for r in rows})
    for d in dists:
        out: dict = {"contact_rate": {}, "fall_rate": {}, "pairs": {}}
        for t in types:
            rs = by[(d, t)]
            out["contact_rate"][t] = float(np.mean([r["truth"]["contacts"] > 0 for r in rs]))
            out["fall_rate"][t] = float(np.mean([bool(r["evidence"][6]) for r in rs]))
        for ta, tb in itertools.combinations(types, 2):
            ra, rb = by[(d, ta)], by[(d, tb)]
            pair: dict = {"kl": {}, "auroc": {}}
            for scheme, n in C["sizes"].items():
                pa = np.bincount([C["code"](r, scheme) for r in ra], minlength=n) + 0.5
                pb = np.bincount([C["code"](r, scheme) for r in rb], minlength=n) + 0.5
                pa, pb = pa / pa.sum(), pb / pb.sum()
                pair["kl"][scheme] = [kl(pa, pb), kl(pb, pa)]
            for f, i in FEATURES.items():
                pair["auroc"][f] = auroc([r["evidence"][i] for r in ra + rb], [True] * len(ra) + [False] * len(rb))
            out["pairs"][f"{ta}|{tb}"] = pair
        res["by_distance"][str(d)] = out
    for scheme in C["sizes"]:
        vals = [np.mean(p["kl"][scheme]) for o in res["by_distance"].values() for p in o["pairs"].values()]
        res["summary"][scheme] = float(np.mean(vals))
    # EM discovery on synthetic entities (units of `--units` encounters of one type at one distance)
    rng = np.random.default_rng(0)
    counts, enc, coll, truth = [], [], [], []
    for (d, t), rs in by.items():
        idx = rng.permutation(len(rs))
        for k in range(0, len(idx) - a.units + 1, a.units):
            c = np.zeros(8)
            for i in idx[k:k + a.units]:
                c[C["code"](rs[i], "gyro8")] += 1
            counts.append(c)
            enc.append(float(a.units))
            coll.append(float(sum(bool(rs[i]["evidence"][6]) for i in idx[k:k + a.units])))
            truth.append((t, d))
    counts_a = np.stack(counts)
    fits = {K: fit_types(counts_a, np.array(enc), np.array(coll), K, "gyro8", restarts=8, seed=K) for K in (1, 2, 3, 4)}
    best = min(fits, key=lambda k: fits[k].bic())
    m = fits[best]
    L = counts_a @ np.log(m.E).T + np.log(m.pi)
    assign = L.argmax(1)
    table: dict = defaultdict(lambda: np.zeros(best, int))
    for (t, d), z in zip(truth, assign, strict=True):
        table[f"{t}@{d}"][z] += 1
    res["em"] = {"bic": {k: f.bic() for k, f in fits.items()}, "K": best, "pi": m.pi.tolist(),
                 "assignment_by_true_type": {k: v.tolist() for k, v in table.items()}, "units": len(counts)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("mean pairwise KL (nats/encounter):", {k: round(v, 3) for k, v in res["summary"].items()})
    for d, o in res["by_distance"].items():
        print("d", d, "contact", {k: round(v, 2) for k, v in o["contact_rate"].items()}, "falls",
              {k: round(v, 3) for k, v in o["fall_rate"].items()})
        for pname, p in o["pairs"].items():
            print("   ", pname, f"KL gyro8 {np.mean(p['kl']['gyro8']):.3f} binary {np.mean(p['kl']['binary']):.3f} "
                  f"fall {np.mean(p['kl']['fall']):.3f} gyro_stab {np.mean(p['kl']['gyro_stab']):.3f} | "
                  f"AUROC gyro {p['auroc']['gyro']:.2f} stab {p['auroc']['stab']:.2f}")
    print("EM K*", best, "BIC", {k: round(v, 1) for k, v in res["em"]["bic"].items()})
    for k, v in sorted(res["em"]["assignment_by_true_type"].items()):
        print("   ", k, v)


if __name__ == "__main__":
    main()
