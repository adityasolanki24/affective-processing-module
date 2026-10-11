"""G6: candidates under held-out procedural variations (apm_select.variations), no retuning.

`--systems` JSON: {"rules": {name: g_common spec}, "models": [G4/G5 cell names, all seeds evaluated]}.
Each (condition, lifetime) is shared by every system (common random numbers within a condition).

    python scripts/g6_run.py --partition X --n 150 --systems configs/G6_systems.json --models runs/G45/models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import seeds as S  # noqa: E402
from g45_eval import _NETS  # noqa: E402
from g45_eval import _init as _init_nets  # noqa: E402
from g45_train import _W, make_learner  # noqa: E402
from g_common import RESULT_KEYS, make_policy  # noqa: E402

from apm_research.harness import run_lifetime  # noqa: E402
from apm_select.instrument import Instrumented  # noqa: E402
from apm_select.provenance import confirmatory_lock, environment, write_json  # noqa: E402
from apm_select.variations import CONDITIONS, lifetime, patched, wrap  # noqa: E402

EPISODES = 5


def job(j: dict) -> dict:
    cond = j["condition"]
    lt = lifetime(cond, S.world_seed("G6", j["partition"], j["index"]), EPISODES)
    if "model" in j:
        arch, net = _NETS[j["model"]]
        _W["arch"] = arch
        inner = make_learner(net, "greedy", 0)
        label, seed = j["model"].rsplit("_s", 1)
        seed = int(seed)
    else:
        inner, label, seed = make_policy(j["spec"]), j["label"], 0
    pol = Instrumented(wrap(inner, cond, seed=j["index"]))
    with patched(cond):
        r = run_lifetime(lt, pol)
    return {"label": label, "seed": seed, "index": j["index"], "condition": cond,
            "result": {k: float(r[k]) for k in RESULT_KEYS} | {k: float(v) for k, v in pol.stats.items()},
            "actions": r["actions"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--systems", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("runs/G45/models"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G6"))
    args = ap.parse_args()
    provenance = confirmatory_lock(HERE / "prereg" / "G6_ADDENDUM.md") if args.partition in ("X", "S") \
        else environment()
    start = time.time()
    cfg = json.loads(args.systems.read_text())
    models = sorted(f.stem for f in args.models.glob("*_s?.pt") if f.stem.rsplit("_s", 1)[0] in cfg["models"])
    jobs = []
    for cond in args.conditions.split(","):
        for i in range(args.n):
            jobs += [{"condition": cond, "partition": args.partition, "index": i, "spec": s, "label": k}
                     for k, s in cfg["rules"].items()]
            jobs += [{"condition": cond, "partition": args.partition, "index": i, "model": m} for m in models]
    jobs.sort(key=lambda j: j.get("spec", {}).get("kind") != "b2v")
    out = args.out / args.partition
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init_nets, initargs=(str(args.models), args.predictor)) as pool, \
            open(out / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(job, jobs, chunksize=4):
            fh.write(json.dumps(r) + "\n")
    write_json(out / "run_meta.json", {"provenance": provenance, "systems": cfg, "models": models, "n": args.n,
                                       "conditions": args.conditions.split(","), "episodes": EPISODES,
                                       "wall_s": round(time.time() - start, 1)})
    print("done", len(jobs), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
