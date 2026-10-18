"""K1 diagnosis: how much a NERVA encounter can reveal about calm vs pushy, by distance and sensor.

    python k1_info.py --rows <J3-T T_a.jsonl> <T_b.jsonl> --edges <J4_edges.json> --out results/K1_information.json

- IMU detector and graded IMU (acc8): from J3-T physically simulated standing encounters, 500 per type and distance.
  These are robot-observable.
- Ideal touch sensor: any contact at all, from the same encounters' contact log. A hypothetical reference.
- Camera on the entity's lunges (declared extension): Monte Carlo of the entity model in `nerva_world`, with the
  `Camera` parameters, over a 4 s window. It does not depend on distance while in view.

KL divergences use add-½ (IMU) or add-0.1 (camera) smoothing. KL(pushy‖calm) governs learning caution;
KL(calm‖pushy) governs recovering from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_transfer.nerva_world import TYPES, Camera, camera_code  # noqa: E402


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def dist(vals: list[int], K: int, a: float = 0.5) -> np.ndarray:
    c = np.bincount(vals, minlength=K) + a
    return c / c.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for p in a.rows for x in p.read_text(encoding="utf-8").splitlines()]
    edges = json.loads(a.edges.read_text())["acc8"]
    res: dict = {"imu": {}, "camera": {}}
    for d in (0.2, 0.4, 0.6):
        R = {t: [r for r in rows if r["type"] == t and r["distance"] == d] for t in ("calm", "pushy")}
        det = {t: dist([int(r["evidence"][9]) for r in R[t]], 2) for t in R}
        a8 = {t: dist([int(np.searchsorted(edges, r["evidence"][1], side="right")) for r in R[t]], 8) for t in R}
        tac = {t: dist([int(r["truth"]["contacts"] > 0) for r in R[t]], 2) for t in R}
        res["imu"][str(d)] = {
            "n": {t: len(R[t]) for t in R},
            "detector_rate": {t: float(det[t][1]) for t in R},
            "detector_kl_pushy_calm": kl(det["pushy"], det["calm"]), "detector_kl_calm_pushy": kl(det["calm"],
                                                                                                    det["pushy"]),
            "acc8_kl_pushy_calm": kl(a8["pushy"], a8["calm"]), "acc8_kl_calm_pushy": kl(a8["calm"], a8["pushy"]),
            "touch_rate": {t: float(tac[t][1]) for t in R},
            "touch_kl_calm_pushy": kl(tac["calm"], tac["pushy"]),
            "calm_meetings_to_halve_log_odds_3": 3.0 / kl(a8["calm"], a8["pushy"])}
    rng, cam = np.random.default_rng(0), Camera()
    codes: dict = {}
    for t in ("calm", "pushy"):
        p = TYPES[t]
        out = []
        for _ in range(20000):
            k = rng.poisson(p["rate"] * 4.0)
            seen = [float(r + rng.normal(0, cam.sigma)) for r in rng.uniform(*p["reach"], k)
                    if rng.random() < cam.p_detect]
            out.append(camera_code(seen))
        codes[t] = dist(out, 8, 0.1)
    res["camera"] = {"kl_pushy_calm": kl(codes["pushy"], codes["calm"]), "kl_calm_pushy": kl(codes["calm"],
                                                                                          codes["pushy"]),
                     "codes": {t: v.tolist() for t, v in codes.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for d, v in res["imu"].items():
        print(d, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if "kl" in k or "halve" in k})
    print("camera", round(res["camera"]["kl_pushy_calm"], 3), round(res["camera"]["kl_calm_pushy"], 3))


if __name__ == "__main__":
    main()
