"""G3 closed loop: the transparent rule with each predictor kind (5-seed ensemble) — equal λ tuning on G3-V, then the
test on G3-X (prereg/G3_ADDENDUM.md) with negative controls for the stateful kinds.

    python scripts/g3_closed.py tune --models runs/G3/models --k 16 --n 30 --workers 90
    python scripts/g3_closed.py test --models runs/G3/models --n <addendum> --workers 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g_common import RESULT_KEYS, init, lifetime_for, make_policy  # noqa: E402

from apm_research.harness import run_lifetime  # noqa: E402
from apm_select.instrument import Instrumented  # noqa: E402
from apm_select.provenance import confirmatory_lock, environment, write_json  # noqa: E402
from apm_select.stateful import KINDS, StatefulPredictor, StatefulRule  # noqa: E402
from apm_select.tuning import lhs  # noqa: E402

SPACE = {"lam_c": ("log", 0.1, 10.0), "lam_s": ("lin", 0.0, 4.0)}
EPISODES, SEED = 10, 20261012
_MODELS: dict = {}


def _init(models_dir: str) -> None:
    import torch
    torch.set_num_threads(1)
    init()
    for f in sorted(Path(models_dir).glob("*_s?.pt")):
        saved = torch.load(f, weights_only=False)
        _MODELS.setdefault(saved["kind"], []).append(StatefulPredictor.load(saved))


def run_job(job: dict) -> dict:
    spec = job["spec"]
    lt = lifetime_for("G3", job["partition"], job["index"], EPISODES)
    if spec["kind"] == "stateful":
        shuffle = np.asarray(job["shuffle"], np.float32) if job.get("shuffle") is not None else None
        inner = StatefulRule(_MODELS[spec["predictor"]], spec["lam_c"], spec["lam_s"], control=spec.get("control"),
                             shuffle_history=shuffle)
    else:
        inner = make_policy(spec)
    pol = Instrumented(inner)
    r = run_lifetime(lt, pol)
    out = {"label": spec["label"], "config_id": spec.get("config_id"), "index": job["index"],
           "partition": job["partition"], "condition": "nominal",
           "totals": {k: float(r[k]) for k in RESULT_KEYS} | {k: float(v) for k, v in pol.stats.items()},
           "actions": r["actions"]}
    if job.get("keep_z"):
        out["z"] = np.stack(inner.z_log).round(5).tolist()
    return out


def run_jobs(jobs, workers, models, out_path=None):
    res, fh = [], (open(out_path, "w", encoding="utf-8") if out_path else None)
    try:
        with Pool(workers, initializer=_init, initargs=(str(models),)) as pool:
            for r in pool.imap_unordered(run_job, jobs, chunksize=1):
                res.append(r)
                if fh:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != "z"}) + "\n")
    finally:
        if fh:
            fh.close()
    return res


def tune(args) -> None:
    start = time.time()
    specs = []
    for kind in KINDS:
        for i, v in enumerate(lhs(SPACE, args.k, SEED + 10 * KINDS.index(kind))):
            specs.append({"kind": "stateful", "predictor": kind, "label": kind, "config_id": f"{kind}:{i}", **v})
    jobs = [{"spec": s, "partition": "V", "index": 100 + i} for s in specs for i in range(args.n)]
    args.out.mkdir(parents=True, exist_ok=True)
    res = run_jobs(jobs, args.workers, args.models, args.out / "tune_lifetimes.jsonl")
    by = defaultdict(list)
    for r in res:
        by[r["config_id"]].append(r["totals"])
    table = []
    for s in specs:
        rows = by[s["config_id"]]
        t = {k: sum(r[k] for r in rows) for k in ("collisions", "minutes", "goals_completed", "goals_issued")}
        table.append({**s, "J_mean": float(np.mean([r["J"] for r in rows])),
                      "collision_rate": t["collisions"] / t["minutes"],
                      "completion": t["goals_completed"] / t["goals_issued"]})
    best = {k: max((t for t in table if t["predictor"] == k), key=lambda t: t["J_mean"]) for k in KINDS}
    stateful = [k for k in KINDS if k != "none"]
    single = ("ema", "lds", "gru")
    sel = {"rule": "per kind max mean J on G3-V 100..; best_state = argmax over stateful kinds; "
                   "best_single = argmax over ema/lds/gru",
           "best_state": max(stateful, key=lambda k: best[k]["J_mean"]),
           "best_single": max(single, key=lambda k: best[k]["J_mean"]), "selected": best}
    write_json(HERE / "results" / "G3" / "tuning_table.json", {"table": table, "environment": environment(),
                                                                "wall_s": round(time.time() - start, 1)})
    (HERE / "configs" / "G3_selected.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["config_id"], round(v["J_mean"], 2), round(v["collision_rate"], 4))
                      for k, v in best.items()} | {"best_state": sel["best_state"], "best_single": sel["best_single"]},
                     indent=1))


def test(args) -> None:
    provenance = confirmatory_lock(HERE / "prereg" / "G3_ADDENDUM.md") if args.partition in ("X", "S") \
        else environment()
    start = time.time()
    sel = json.loads((HERE / "configs" / "G3_selected.json").read_text())
    g1 = json.loads((HERE / "configs" / "G1_selected.json").read_text())
    systems = {k: {"kind": "stateful", "predictor": k, "lam_c": v["lam_c"], "lam_s": v["lam_s"], "label": k}
               for k, v in sel["selected"].items()}
    systems["A0_R2"] = {**g1["systems"][g1["best_simple"]], "label": "A0_R2"}
    controls = sorted({sel["best_state"], "gru", "fastslow"})
    jobs = [{"spec": s, "partition": args.partition, "index": i, "keep_z": s.get("predictor") in controls}
            for s in systems.values() for i in range(args.n)]
    out_dir = args.out / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_jobs(jobs, args.workers, args.models, out_dir / "lifetimes_base.jsonl")
    z = {(r["label"], r["index"]): r["z"] for r in res if "z" in r}
    cjobs = []
    for k in controls:
        for i in range(args.n):
            cjobs.append({"spec": {**systems[k], "label": f"{k}_reset", "control": "reset"},
                          "partition": args.partition, "index": i})
            cjobs.append({"spec": {**systems[k], "label": f"{k}_shuffled"}, "partition": args.partition,
                          "index": i, "shuffle": z[(k, (i + 1) % args.n)]})
    cres = run_jobs(cjobs, args.workers, args.models, out_dir / "lifetimes_controls.jsonl")
    with open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in res + cres:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "z"}) + "\n")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "systems": systems, "controls": controls,
                                           "selection": {k: sel[k] for k in ("best_state", "best_single")},
                                           "systems_file": "configs/G3_selected.json", "n": args.n,
                                           "wall_s": round(time.time() - start, 1)})
    print("done", len(res) + len(cres), round(time.time() - start, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tune", "test"))
    ap.add_argument("--models", type=Path, default=Path("runs/G3/models"))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--partition", default="X")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G3"))
    args = ap.parse_args()
    (tune if args.mode == "tune" else test)(args)


if __name__ == "__main__":
    main()
