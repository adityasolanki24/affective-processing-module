"""H7: cross-embodiment correspondence of expressive qualities (prereg/H7_ADDENDUM.md). Numpy only.

    python scripts/h7_embodiment.py --out results/H7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h1_analyze import LDA, _scores, classify_logo  # noqa: E402
from h1_repertoire import GEOMETRIES, S  # noqa: E402
from h34_analyze import spearman  # noqa: E402

from apm_express.features import episode_features  # noqa: E402
from apm_express.styles import DESIGNED, NEUTRAL  # noqa: E402
from apm_express.wheeled import run_wheeled  # noqa: E402

SHARED = ("progress_speed", "pauses_per_m", "mean_pause_s", "switch_rate", "backward_s", "straightness",
          "max_lateral_dev", "final_distance", "final_bearing_abs", "body_yaw_rest_std", "retreat_m")
SHARED_NON_SPEED = tuple(f for f in SHARED if f != "progress_speed")


def wheeled_rows(seeds: int) -> list[dict]:
    rows, i = [], 0
    for name, it in {"neutral": NEUTRAL, **DESIGNED}.items():
        for g, (d, b) in enumerate(GEOMETRIES):
            for _ in range(seeds):
                target = (d * math.cos(b), d * math.sin(b))
                a = run_wheeled(it, target, S.sim_seed("H7", "X", i))
                rows.append({"trial": {"style": name, "geometry": g}, "features": episode_features(a, target)})
                i += 1
    return rows


def _z(rows, feats):
    X = np.array([[r["features"][f] for f in feats] for r in rows], float)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 1e-9, sd, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock
    lock = confirmatory_lock(HERE / "prereg" / "H7_ADDENDUM.md")
    biped = json.loads((HERE / "results/H1/X/core_primary.json").read_text(encoding="utf-8"))["rows"]
    wheel = wheeled_rows(a.seeds)
    c1 = classify_logo(wheel, SHARED_NON_SPEED)
    styles = sorted({r["trial"]["style"] for r in biped})
    c2 = {}
    for f in SHARED:
        mb = [np.mean([r["features"][f] for r in biped if r["trial"]["style"] == s]) for s in styles]
        mw = [np.mean([r["features"][f] for r in wheel if r["trial"]["style"] == s]) for s in styles]
        c2[f] = spearman(mb, mw)
    model = LDA().fit(_z(biped, SHARED_NON_SPEED), [r["trial"]["style"] for r in biped])
    c3 = _scores([r["trial"]["style"] for r in wheel], model.predict(_z(wheel, SHARED_NON_SPEED)))
    med = float(np.median(list(c2.values())))
    res = {"lock": lock, "n_wheeled": len(wheel), "n_biped": len(biped), "C1_wheeled_distinctness": c1,
           "C2_correspondence": c2, "C2_median": med, "C3_transfer": c3,
           "supports_correspondence": bool(c1["balanced_accuracy"] >= 0.80 and med >= 0.6
                                           and c3["balanced_accuracy"] >= 0.50),
           "wheeled_style_means": {s: {f: float(np.mean([r["features"][f] for r in wheel if r["trial"]["style"] == s]))
                                       for f in SHARED} for s in styles}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "h7_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("C2_median", "supports_correspondence")}),
          round(c1["balanced_accuracy"], 2), round(c3["balanced_accuracy"], 2), {k: round(v, 2) for k, v in c2.items()})


if __name__ == "__main__":
    main()
