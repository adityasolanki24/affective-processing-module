"""M6 (Q6): is expression physically measurable and graded, not just binary avoidance?

    python experiments/M_release/scripts/m6_motion.py --runs <X dirs> --out results/m6/motion.json

From the X lifetimes (all systems), per robot and per encounter (approaches only, standoff 0):
- intent channels chosen by the runtime: proximity, tempo, hesitation;
- physically measured motion from the robot's own pose trace (Isaac `motion` features): final distance to the
  person, mean speed while moving, pauses, reversing time, path straightness; MuJoCo: stopping distance and
  arrival time.

Reported:
- Spearman correlations: proximity vs final distance (expected negative), tempo vs mean speed (positive), hesitation
  vs pauses (positive);
- how many distinct intents were executed, and the spread (IQR) of the measured features;
- the share of approaches that were not simply "approach" or "stay away" (graded proximity strictly between the grid
  extremes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_analyze import load  # noqa: E402


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Zero-based midranks: tied observations receive their group's mean rank."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), float)
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    rx, ry = average_ranks(x[ok]), average_ranks(y[ok])
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recs = [r for r in load(a.runs) if r.get("partition") == "X"]
    res: dict = {}
    for robot in sorted({r["robot"] for r in recs}):
        rows = [x for r in recs if r["robot"] == robot for x in r["rows"] if x.get("standoff", 0) == 0
                and x.get("arrived")]
        if not rows:
            continue
        prox = [x["proximity"] for x in rows]
        tempo = [x["tempo"] for x in rows]
        hes = [x["hesitation"] for x in rows]
        mot = [x.get("motion") or {} for x in rows]
        physical = [(x.get("measured_distance") or {}).get("radial_surface_gap_m") for x in rows]
        # Old M records do not contain the moving person's pose. Never relabel
        # their root-to-station distance as a body surface gap.
        has_physical = all(v is not None for v in physical)
        dist = (physical if has_physical else
                [m.get("final_dist_to_person", x["d_stop"]) for m, x in zip(mot, rows, strict=True)])
        speed = [m.get("mean_speed", np.nan) for m in mot]
        pauses = [m.get("pauses", np.nan) for m in mot]
        t_arr = [x.get("t_arrive") if x.get("t_arrive") is not None else np.nan for x in rows]
        intents = {(round(p, 2), round(t, 2), round(h, 2)) for p, t, h in zip(prox, tempo, hes, strict=True)}
        res[robot] = {
            "approaches": len(rows), "distinct_intents": len(intents),
            "distance_metric": "radial_surface_gap_m" if has_physical else "legacy_station_or_abstract_distance",
            "graded_proximity_share": float(np.mean([0.05 < p < 0.95 for p in prox])),
            "spearman_proximity_vs_final_distance": spearman(prox, dist),
            "spearman_tempo_vs_mean_speed": spearman(tempo, speed),
            "spearman_tempo_vs_arrival_time": spearman(tempo, t_arr),
            "spearman_hesitation_vs_pauses": spearman(hes, pauses),
            "final_distance_iqr": [float(np.nanpercentile(dist, 25)), float(np.nanpercentile(dist, 75))],
            "mean_speed_iqr": ([float(np.nanpercentile(speed, 25)), float(np.nanpercentile(speed, 75))]
                               if np.isfinite(speed).any() else None),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
