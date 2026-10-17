"""H4 physical "most important test" (prereg/H34_ADDENDUM.md §6).

For each X lifetime index i and each system:
- two robots with identical simulator seed, start state and situation (approach B1 in the plain zone, 1.0 m ahead);
- one lived `risky_B`, the other `safe`;
- each executes its own end-of-lifetime expression intent (from `states_eval.json`) through the H0b interface.

    NERVA_ROOT=... PYTHONPATH="src;<NERVA>" <nerva-venv>/python scripts/h34_probes.py --root results/H34 \
        --systems M_id,S5_gru,S4_pad,S0_none,S1_fixed
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

from h0_interface import _init, interface  # noqa: E402
from h1_repertoire import S  # noqa: E402

from apm_express.adapter import COMMAND_NAMES, D_FAR, D_NEAR, BipedExpressionAdapter  # noqa: E402
from apm_express.episode import EpisodeSpec, run_episode  # noqa: E402
from apm_express.features import episode_features, stability  # noqa: E402
from apm_express.intent import ExpressionIntent  # noqa: E402
from apm_express.safety import SafetyMonitor  # noqa: E402

TARGET = (1.0, 0.0)
HISTORIES = ("risky_B", "safe")


def run(job: dict) -> dict:
    from apm_express.biped import NervaBiped

    ch, limits = interface(job["iface"])
    seed = S.sim_seed("H4", "X", 900_000 + job["index"])
    ctrl = NervaBiped(
        policy=job["iface"]["policy"], seed=seed, head_channel=ch.head, max_head_rate=1.5, max_pitch_rate=2.0
    )
    intent = ExpressionIntent(**job["intent"])
    r = run_episode(
        ctrl, BipedExpressionAdapter(ch, seed=seed), SafetyMonitor(limits), EpisodeSpec(60.0, TARGET, [(0.0, intent)])
    )
    a = r["arrays"]
    d_stop = D_FAR - (D_FAR - D_NEAR) * intent.proximity
    final = float(math.hypot(TARGET[0] - a["x"][-1], TARGET[1] - a["y"][-1]))
    moving = a["command"] != COMMAND_NAMES.index("rest")
    return {
        "system": job["system"],
        "history": job["history"],
        "index": job["index"],
        "sim_seed": seed,
        "intent": job["intent"],
        "d_stop": d_stop,
        "final_distance": final,
        "arrived": bool(abs(final - d_stop) <= 0.15),
        **stability(a, r["completed"]),
        "trips": r["trips"],
        "features": episode_features(a, TARGET),
        "head_while_moving_max": float(np.abs(a["head_request"][moving]).max()) if moving.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--interface", type=Path, default=HERE / "results/H0/X/admitted_interface.json")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock

    lock = confirmatory_lock(HERE / "prereg" / "H34_ADDENDUM.md")
    iface = json.loads(a.interface.read_text(encoding="utf-8"))["interface"]
    from h34_analyze import load_json

    ev = load_json(a.root / "X" / "states_eval.json")
    jobs = []
    for row in ev["rows"]:
        if row["control"] != "none" or row["history"] not in HISTORIES or row["system"] not in a.systems.split(","):
            continue
        intent = row["course"][-1]["probes"]["B1_plain"]["intent"]
        jobs.append(
            {
                "system": row["system"],
                "history": row["history"],
                "index": row["index"],
                "intent": intent,
                "iface": iface,
            }
        )
    start = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    path = a.root / "X" / "probes_physical.json"
    path.write_text(json.dumps({"lock": lock, "rows": rows, "wall_s": round(time.time() - start, 1)}), encoding="utf-8")
    print(json.dumps({"trials": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
