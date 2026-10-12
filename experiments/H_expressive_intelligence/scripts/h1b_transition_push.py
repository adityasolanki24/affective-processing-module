"""H1b: push tolerance across stop transitions (walking -> rest), the state the H1 fall happened in.

The robot walks forward for 5 s, then is commanded to rest. A push of magnitude m arrives at delay d after the stop
command, from heading-relative direction a; 10 s total. Monitor in shadow. Fall = tilt > 45 deg or non-finite.

Grid:
- magnitude m in {0.05, 0.075, 0.10, 0.125, 0.15} m/s;
- delay d in {0.0, 0.2, 0.4, 0.7, 1.0, 2.0} s;
- direction a in 8 directions.

    python scripts/h1b_transition_push.py --partition P --seeds 2 --out results/H1b/pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h0_interface import S, _init  # noqa: E402

MAGNITUDES = (0.05, 0.075, 0.10, 0.125, 0.15)
DELAYS = (0.0, 0.2, 0.4, 0.7, 1.0, 2.0)
DIRECTIONS = tuple(range(0, 360, 45))
STOP_S = 5.0


def run(job) -> dict:
    mag, delay, deg, index, partition = job
    from apm_express.adapter import MotionRequest
    from apm_express.biped import NervaBiped

    seed = S.sim_seed("H1", partition, index)
    ctrl = NervaBiped(policy="neutral", seed=seed)
    k_stop, k_push = int(STOP_S / ctrl.dt), int(round((STOP_S + delay) / ctrl.dt))
    tilts = []
    for k in range(500):
        ctrl.apply(MotionRequest("forward" if k < k_stop else "rest", 0.0, (0.0, 0.0, 0.0, 0.0)))
        if k == k_push:
            a = ctrl.pose()[2] + math.radians(deg)
            ctrl.push(mag * math.cos(a), mag * math.sin(a))
        row = ctrl.advance()
        tilts.append(row["tilt"])
        if not row["finite"]:
            break
    tilts = np.array(tilts)
    return {"magnitude": mag, "delay": delay, "direction_deg": deg, "index": index,
            "fell": bool((tilts > 45).any() or len(tilts) < 500), "max_tilt": float(np.nanmax(tilts)),
            "tilt_at_push": float(tilts[min(k_push, len(tilts) - 1)])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["P", "X"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--index-offset", type=int, default=700_000)
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock, environment, nerva_state
    lock = (confirmatory_lock(HERE / "prereg" / "H1b_ADDENDUM.md") if a.partition == "X"
            else {**environment(), **nerva_state(), "partition": "P", "exploratory": True})
    jobs = [(m, d, g, a.index_offset + s, a.partition) for m in MAGNITUDES for d in DELAYS for g in DIRECTIONS
            for s in range(a.seeds)]
    start = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        rows = pool.map(run, jobs, chunksize=2)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "transition_push.json"
    path.write_text(json.dumps({"lock": lock, "rows": rows, "wall_s": round(time.time() - start, 1)}, indent=1),
                    encoding="utf-8")
    print(json.dumps({"trials": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
