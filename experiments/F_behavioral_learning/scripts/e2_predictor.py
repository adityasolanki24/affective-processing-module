"""E2 predictor (B1/B2 shared): exploration lifetimes on E2 partitions T and V (randomised safe action variation),
5 seeds of the action-conditioned MLP, saved as an ensemble. Reports validation metrics; no system is evaluated here.

    python scripts/e2_predictor.py --train 150 --val 40 --episodes 5 --out runs/E2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm.learning.evaluation import auroc, brier, ece  # noqa: E402
from apm_research.dataset import DecisionRecorder  # noqa: E402
from apm_research.features import WIDTH  # noqa: E402
from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.policies import Explore  # noqa: E402
from apm_research.predictor import ActionPredictor, EnsemblePredictor  # noqa: E402
from apm_research.provenance import environment, write_json  # noqa: E402
from apm_testbeds.wheeled_ctrl import child  # noqa: E402


def collect(job):
    partition, index, episodes, experiment = job
    ss = S.world_seed(experiment, partition, index)
    rec = DecisionRecorder()
    run_lifetime(Lifetime(ss, n_episodes=episodes), Explore(child(ss, 500)), recorder=rec)
    return rec.rows


def fit(args):
    seed, data = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    (X, A, c, s, p), (Xv, Av, cv, sv, pv) = data
    return ActionPredictor(WIDTH, seed=seed).fit(X, A, c, s, p, Xv, Av, cv, sv, pv)


def arrays(rows):
    rec = DecisionRecorder()
    rec.rows = rows
    return rec.arrays()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("runs/E2"))
    args = ap.parse_args()
    start = time.time()
    jobs = [("T", i, args.episodes, args.experiment) for i in range(args.train)]
    jobs += [("V", i, args.episodes, args.experiment) for i in range(args.val)]
    with Pool(4) as pool:
        rows = pool.map(collect, jobs, chunksize=2)
    train = [r for rs in rows[:args.train] for r in rs]
    val = [r for rs in rows[args.train:] for r in rs]
    data = (arrays(train), arrays(val))
    with Pool(4) as pool:
        members = pool.map(fit, [(seed, data) for seed in range(5)])
    ens = EnsemblePredictor(members)
    Xv, Av, cv, sv, pv = data[1]
    pc, ps, pp = ens.predict(Xv, Av)
    report = {"environment": environment(), "train_decisions": len(train), "val_decisions": len(val),
              "train_rates": {"collision": float(data[0][2].mean()), "slip": float(data[0][3].mean()),
                              "progress_mean": float(data[0][4].mean())},
              "val": {"collision": {"brier": brier(pc, cv), "auroc": auroc(pc, cv), "ece": ece(pc, cv)},
                      "slip": {"brier": brier(ps, sv), "auroc": auroc(ps, sv), "ece": ece(ps, sv)},
                      "progress_rmse": float(np.sqrt(np.mean((pp - pv) ** 2))),
                      "progress_r2": float(1 - np.mean((pp - pv) ** 2) / np.var(pv))},
              "member_epochs": [len(m.history) for m in members], "parameters_each": members[0].parameters,
              "wall_s": round(time.time() - start, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "predictor.json"
    path.write_text(json.dumps(ens.to_dict()))
    report["predictor_sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    write_json(HERE / "results" / args.experiment / "predictor_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=1))


if __name__ == "__main__":
    main()
