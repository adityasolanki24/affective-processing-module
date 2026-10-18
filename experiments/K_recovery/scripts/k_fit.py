"""Fit programme K's NERVA type model from K-T exploration (RANDINTENT), with an IMU part and a camera part.

    python k_fit.py --runs <T_explore.jsonl> --out configs/K_types.json

**Units.** One unit per (lifetime, identity) in the *training world*: histories risky_B and safe, which have no
change, so each identity keeps one behaviour.
- Counts per unit: [4 blocks, 8 categories].
  - Blocks 0–2: IMU acc8 category per observable sensing window, in the band of the robot's own stopping distance.
  - Block 3: camera code per window (the declared sensor extension).
- EM (`apm_transfer.conditioned.fit_conditional`) reads only these counts. K = 1–4 is chosen by BIC.
- The rate r_k is the robot's own detector frequency per window.

**Programme L** (`--edges FILE`): the body's fixed acc8 edges re-categorise the IMU peaks, and nothing else is
refitted (vision type models share K's body configuration).

**Wheeled robot** (`--robot wheeled`; APM v1.0: also `nova_carter`, `t1_isaac`, `nerva_isaac`). Its own acc8 edges
come from pooled quantiles of its own window peaks (all exploration lifetimes), and its own time model from least
squares of its own arrival times on intent (as J6). Both are written next to `--out`.

IMU-only systems use the same model and ignore the camera block. Because blocks are conditionally independent given
the type, that is exact marginalisation.

**Oracle reference** (evaluation only, never a deployable input): the mean posterior of units whose true type is calm,
and of units whose true type is pushy. This is the belief an observer with complete evidence would hold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_transfer.conditioned import bic_conditional, fit_conditional  # noqa: E402

CAM_OFFSET, BLOCKS, C = 24, 4, 8


def units(path: Path, edges: list[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    counts, enc, det, types = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        lt = json.loads(line)
        if edges is not None:
            for r in lt["rows"]:
                r["imu_cats"] = [int(np.searchsorted(edges, v, side="right")) for v in r["imu_peaks"]]
        if lt["history"] not in ("risky_B", "safe"):
            continue
        per: dict[str, dict] = {}
        for r in lt["rows"]:
            u = per.setdefault(r["entity"], {"c": np.zeros((BLOCKS, C)), "n": 0, "d": 0, "type": r["type"]})
            for c in r["imu_cats"]:
                u["c"][r["band"], c] += 1
                u["n"] += 1
            for c in r["cam_codes"]:
                u["c"][3, c] += 1
            u["d"] += int(r["detected"])
        for u in per.values():
            counts.append(u["c"])
            enc.append(max(1, u["n"]))
            det.append(u["d"])
            types.append(u["type"])
    return np.array(counts), np.array(enc, float), np.array(det, float), types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--robot", choices=("nerva", "wheeled", "humanoid", "nova_carter", "t1_isaac", "nerva_isaac"),
                    default="nerva", help="APM v1.0: the Isaac robots fit their own edges and time model")
    ap.add_argument("--edges", type=Path, default=None,
                    help="programme L: categorise with these fixed acc8 edges (no edge or time-model refit)")
    a = ap.parse_args()
    edges = None
    if a.edges is not None:
        edges = json.loads(a.edges.read_text())["acc8"]
    elif a.robot != "nerva":
        from apm_express.intent import ExpressionIntent
        from apm_express.styles import NEUTRAL
        from apm_transfer.expression_map import TimeModel
        lts = [json.loads(x) for x in a.runs.read_text(encoding="utf-8").splitlines()]
        peaks = [v for lt in lts for r in lt["rows"] for v in r["imu_peaks"]]
        edges = np.unique(np.quantile(peaks, np.linspace(0, 1, 9)[1:-1])).tolist()
        (a.out.parent / f"K_{a.robot}_edges.json").write_text(
            json.dumps({"acc8": edges, "source": "K5-T pooled quantiles"}), encoding="utf-8")
        base = NEUTRAL.as_dict()
        rows = [r for lt in lts for r in lt["rows"] if r["t_arrive"] is not None]
        its = [ExpressionIntent(**{**base, "proximity": r["proximity"], "tempo": r["tempo"],
                                   "hesitation": r["hesitation"]}) for r in rows]
        tm = TimeModel.fit(its, [r["t_arrive"] for r in rows])
        pred = np.array([tm.predict(i) for i in its])
        t = np.array([r["t_arrive"] for r in rows])
        r2 = float(1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2))
        (a.out.parent / f"K_{a.robot}_time.json").write_text(json.dumps({"time_coef": tm.coef, "r2": r2,
                                                                      "encounters": len(rows)}), encoding="utf-8")
        print(a.robot, "edges", np.round(edges, 4), "time model", np.round(tm.coef, 2), "R2", round(r2, 3))
    counts, enc, det, types = units(a.runs, edges)
    fits = {K: fit_conditional(counts, enc, det, K, "k_imu_cam", restarts=20) for K in (1, 2, 3, 4)}
    bics = {K: bic_conditional(m, BLOCKS) for K, m in fits.items()}
    K = min(bics, key=lambda k: bics[k])
    m = fits[K]
    flat = counts.reshape(len(counts), -1)
    L = flat @ np.log(m.E).T + np.log(m.pi)
    post = np.exp(L - L.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    oracle = {t: post[[i for i, x in enumerate(types) if x == t]].mean(0).tolist() for t in ("calm", "pushy")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": m.to_dict(), "bic": {str(k): v for k, v in bics.items()}, "oracle": oracle,
                                 "units": len(counts)}, indent=1), encoding="utf-8")
    strong = m.E.reshape(K, BLOCKS, C)[:, :3, 6:].sum(2)
    print("units", len(counts), "BIC", {k: round(v, 1) for k, v in bics.items()}, "K", K)
    print("pi", np.round(m.pi, 3), "r", np.round(m.r, 3))
    print("P(strong | type, band)", np.round(strong, 3))
    print("camera emissions", np.round(m.E.reshape(K, BLOCKS, C)[:, 3], 3))
    print("oracle", {t: np.round(v, 3).tolist() for t, v in oracle.items()})


if __name__ == "__main__":
    main()
