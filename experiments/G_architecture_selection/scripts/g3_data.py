"""G3 data: exploration lifetimes (F's randomised safe action variation, 5 episodes) on G3 partitions, recording per
decision the 41 features, the APM v0.1 readout (V, A, D, u), the action and 2 s outcome labels.

    python scripts/g3_data.py --partition T --n 150 --out runs/G3/data --workers 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_select.stateful import ExploreRecorder, LabelRecorder, lifetime_arrays  # noqa: E402
from apm_testbeds.wheeled_ctrl import child  # noqa: E402

EPISODES = 5


def collect(job: tuple[str, int]) -> dict:
    partition, index = job
    ss = S.world_seed("G3", partition, index)
    pol, rec = ExploreRecorder(child(ss, 500)), LabelRecorder()
    run_lifetime(Lifetime(ss, n_episodes=EPISODES), pol, recorder=rec)
    return {"index": index, **lifetime_arrays(pol, rec)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/G3/data"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    start = time.time()
    with Pool(args.workers) as pool:
        lts = pool.map(collect, [(args.partition, i) for i in range(args.n)], chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.partition}.pkl"
    path.write_bytes(pickle.dumps(lts))
    info = {"partition": args.partition, "lifetimes": len(lts), "decisions": int(sum(len(x["A"]) for x in lts)),
            "labelled": int(sum(x["mask"].sum() for x in lts)),
            "collision_rate": float(sum(x["y_col"][x["mask"]].sum() for x in lts) / sum(x["mask"].sum() for x in lts)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "wall_s": round(time.time() - start, 1)}
    (args.out / f"{args.partition}.manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
