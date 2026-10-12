"""H5 analysis (prereg/H5_ADDENDUM.md): learned vs fixed expression mappings on X.

python scripts/h5_analyze.py --root results/H34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h34_analyze import _end, _mean_ci, crossing_times, e1_from, e1_points, kappa_e, paired_diff  # noqa: E402

from apm_select.survival import km  # noqa: E402

ORDER = ("L0_fixed", "L1", "L2", "L3")
TRUTH = ("B1_plain", "B1_marked", "A1_plain", "C1_plain", "B3_novel")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    ev = json.loads((a.root / "X" / "h5_eval.json").read_text(encoding="utf-8"))
    rows = ev["rows"]
    keys = sorted({(r["history"], r["index"]) for r in rows})
    by = {s: [r for r in rows if r["system"] == s] for s in ORDER if any(r["system"] == s for r in rows)}
    pts = {s: e1_points(b) for s, b in by.items()}
    ce = {
        s: {
            (r["history"], r["index"]): [abs(kappa_e(_end(r, p)["intent"]) - _end(r, p)["truth"]) for p in TRUTH]
            for r in b
        }
        for s, b in by.items()
    }
    res: dict = {"systems": {}, "training": ev["training"]}
    for s, b in by.items():
        safe_l = float(np.mean([kappa_e(_end(r, "B1_plain")["intent"]) for r in b if r["history"] == "safe"]))
        risky_l = float(np.mean([kappa_e(_end(r, "B1_plain")["intent"]) for r in b if r["history"] == "risky_B"]))
        rev = crossing_times(b, "change_B", False, safe_l, risky_l)
        res["systems"][s] = {
            "CE": _mean_ci([x for k in keys for x in ce[s][k]]),
            "E1": e1_from(pts[s], keys),
            "E2": paired_diff(b, "B1_plain", "risky_B", "safe"),
            "E4_revision_median": km(np.array([t for t, _ in rev]), np.array([e for _, e in rev]), 24.0)[0],
            "E4_revised_fraction": float(np.mean([e for _, e in rev])),
            "E6": paired_diff(b, "B3_novel", "risky_B", "safe"),
        }
    rng = np.random.default_rng(0)
    boots = {s: [] for s in by}
    for _ in range(2000):
        samp = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals = {s: float(np.mean([x for k in samp for x in ce[s][k]])) for s in by}
        for s in by:
            boots[s].append(vals[s])
    best = min(by, key=lambda s: res["systems"][s]["CE"]["mean"])
    ni = {s: float(np.percentile(np.array(boots[s]) - np.array(boots[best]), 95)) for s in by}
    preferred = next(s for s in ORDER if s in by and ni[s] < 0.02)
    beats = {}
    for s in by:
        if s == "L0_fixed":
            continue
        d = np.array(boots[s]) - np.array(boots["L0_fixed"])
        r, r0 = res["systems"][s], res["systems"]["L0_fixed"]
        beats[s] = {
            "CE_diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "E1_not_lower": bool(r["E1"] >= r0["E1"]),
            "E4_not_slower": bool(not (r["E4_revision_median"] > r0["E4_revision_median"])),
        }
        beats[s]["beats_L0"] = bool(
            beats[s]["CE_diff_ci"][1] < 0 and beats[s]["E1_not_lower"] and beats[s]["E4_not_slower"]
        )
    res.update({"best_CE": best, "ni_upper_vs_best": ni, "preferred_simplest": preferred, "vs_L0": beats})
    (a.root / "X" / "h5_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for s, r in res["systems"].items():
        print(s, {k: (round(v["mean"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in r.items()})
    print("best", best, "preferred", preferred, json.dumps(beats))


if __name__ == "__main__":
    main()
