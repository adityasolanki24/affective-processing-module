"""H0b: push robustness of the validated controllers while standing (rest command), not covered by NERVA's gate.

Robot at rest (head offsets zero unless --head), push at t = 3 s with magnitude m in heading-relative direction d,
8 s total. Fall = tilt > 45 deg or non-finite state. Monitor in shadow.

    python scripts/h0b_rest_push.py --partition P --seeds 3 --out results/H0b/pilot
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

MAGNITUDES = (0.1, 0.15, 0.2, 0.25, 0.3)
DIRECTIONS = tuple(range(0, 360, 45))


def run(job) -> dict:
    policy, mag, deg, index, partition, head = job
    from apm_express.adapter import MotionRequest
    from apm_express.biped import NervaBiped

    seed = S.sim_seed("H0", partition, index)
    ctrl = NervaBiped(policy=policy, seed=seed, head_channel=head is not None)
    req = MotionRequest("rest", 0.0, (0.0, *(head or (0.0, 0.0)), 0.0))
    tilts = []
    for k in range(400):
        if k == 150:
            yaw = ctrl.pose()[2]
            a = yaw + math.radians(deg)
            ctrl.push(mag * math.cos(a), mag * math.sin(a))
        ramp = min(1.0, k / 50)
        ctrl.apply(MotionRequest("rest", 0.0, tuple(v * ramp for v in req.head)))
        row = ctrl.advance()
        tilts.append(row["tilt"])
        if not row["finite"]:
            break
    tilts = np.array(tilts)
    return {"policy": policy, "magnitude": mag, "direction_deg": deg, "index": index, "head": head,
            "fell": bool((tilts > 45).any() or len(tilts) < 400), "max_tilt": float(np.nanmax(tilts)),
            "pre_tilt": float(tilts[100:150].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["P", "X"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policies", default="neutral,e1_prime")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--index-offset", type=int, default=2_000_000)
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock, environment, nerva_state
    lock = (confirmatory_lock(HERE / "prereg" / "H0b_ADDENDUM.md") if a.partition == "X"
            else {**environment(), **nerva_state(), "partition": "P", "exploratory": True})
    jobs = [(p, m, d, a.index_offset + s, a.partition, None)
            for p in a.policies.split(",") for m in MAGNITUDES for d in DIRECTIONS for s in range(a.seeds)]
    start = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        rows = pool.map(run, jobs, chunksize=2)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "rest_push.json"
    path.write_text(json.dumps({"lock": lock, "rows": rows, "wall_s": round(time.time() - start, 1)}, indent=1),
                    encoding="utf-8")
    print(json.dumps({"trials": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
