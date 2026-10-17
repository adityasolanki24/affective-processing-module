"""E1 model training: one job = (embodiment, target, arm, seed). Models train on T, early-stop on V, and their
predictions on every evaluation partition present (P, X, S) are saved. Nothing here computes a criterion.

    python scripts/e1_train.py --embodiment wheeled --target collision --data runs/E1 --out runs/E1/preds
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from e1_data import BLOCKS  # noqa: E402

from apm.learning.dataset import armed_mask  # noqa: E402
from apm.learning.features import WIDTH  # noqa: E402

HISTORY = 20
TARGETS = {"collision": 1.0, "stability_loss": 0.5}
SEEDS = (0, 1, 2, 3, 4)


def block(name: str) -> list[int]:
    lo, hi = BLOCKS[name]
    return [WIDTH + i for i in range(lo, hi)]


CANON = list(range(WIDTH))
R = CANON + block("RISK")
ARMS: dict[str, tuple[list[int], int]] = {  # columns, GRU hidden size (0 = logistic regression)
    "LR": (R, 0),
    "G": (CANON, 64),
    "R": (R, 64),
    "RL": (R + block("LONG"), 64),
    "M": (R + block("MEM"), 64),
    "RLM": (R + block("LONG") + block("MEM"), 64),
    "U": (R + block("U"), 64),
    "P": (R + block("PAD"), 64),
    "UP": (R + block("U") + block("PAD"), 64),
    "N": (R + block("NOISE"), 64),
    "C": (R, 67),  # parameters within ±5% of UP (checked in the runner)
}
NERVA_SKIP = {"M", "RLM"}  # NERVA has no entities, so no memory expectations


def windows(episodes, kind: str, horizon: float, cols: list[int]):
    """[N, HISTORY, len(cols)] windows of armed samples, labels in (t, t + horizon], cluster ids, sample flags."""
    xs, ys, gs, benign = [], [], [], []
    for e in episodes:
        full = np.concatenate([e.features, e.extra], axis=1)[:, cols].astype(np.float32)
        padded = np.concatenate([np.zeros((HISTORY - 1, len(cols)), np.float32), full])
        armed = armed_mask(e.times, e.events)
        occ = np.asarray(e.events.get(kind, []))
        truth = e.meta["truth"]
        idx = np.flatnonzero(armed)
        xs.append(np.stack([padded[i:i + HISTORY] for i in idx]) if len(idx) else np.zeros((0, HISTORY, len(cols))))
        ys.append(np.array([float(np.any((occ > e.times[i]) & (occ <= e.times[i] + horizon))) for i in idx]))
        gs.append(np.full(len(idx), e.meta["lifetime"]))
        benign.append(np.array([truth[i, 0] == 0.0 and truth[i, 1] < 2.0 for i in idx]))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.concatenate(gs),
            np.concatenate(benign))


def job(spec: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from apm.learning.baselines import Logistic
    from apm.learning.predictor import GRUPredictor
    data, emb, target, arm, seed = Path(spec["data"]), spec["emb"], spec["target"], spec["arm"], spec["seed"]
    cols, hidden = ARMS[arm]
    horizon = TARGETS[target]
    load = {p: pickle.loads((data / f"{emb}_{p}.pkl").read_bytes()) for p in spec["parts"]}
    X, y, _, _ = windows(load["T"], target, horizon, cols)
    Xv, yv, _, _ = windows(load["V"], target, horizon, cols)
    start = time.time()
    if hidden == 0:
        model = Logistic().fit(X, y)
        params = int(model.w.size)
    else:
        model = GRUPredictor(len(cols), hidden=hidden, seed=seed).fit(X, y, Xv, yv)
        params = model.parameters
    out = {"emb": emb, "target": target, "arm": arm, "seed": seed, "params": params,
           "train_s": time.time() - start, "train_n": int(len(y)), "train_pos": int(y.sum()),
           "val_brier": float(np.mean((model.predict(Xv) - yv) ** 2)),
           "epochs": len(getattr(model, "history", []))}
    preds = {"V_p": model.predict(Xv).astype(np.float32), "V_y": yv}
    for p in spec["parts"]:
        if p in ("T", "V"):
            continue
        Xe, ye, ge, be = windows(load[p], target, horizon, cols)
        t0 = time.perf_counter()
        pe = model.predict(Xe)
        out[f"{p}_infer_us_per_sample"] = 1e6 * (time.perf_counter() - t0) / max(1, len(ye))
        preds.update({f"{p}_p": pe.astype(np.float32), f"{p}_y": ye, f"{p}_g": ge, f"{p}_benign": be})
    dest = Path(spec["out"]) / f"{emb}_{target}_{arm}_s{seed}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **preds)
    (dest.with_suffix(".json")).write_text(json.dumps(out))
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", choices=("wheeled", "nerva"), required=True)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--data", type=Path, default=Path("runs/E1"))
    ap.add_argument("--out", type=Path, default=Path("runs/E1/preds"))
    ap.add_argument("--parts", default="T,V,P")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if not (args.embodiment == "nerva" and a in NERVA_SKIP)]
    specs = []
    for arm in arms:
        for seed in ([0] if ARMS[arm][1] == 0 else [int(s) for s in args.seeds.split(",")]):
            dest = args.out / f"{args.embodiment}_{args.target}_{arm}_s{seed}.json"
            if args.skip_existing and dest.exists():
                continue
            specs.append({"data": str(args.data), "emb": args.embodiment, "target": args.target, "arm": arm,
                          "seed": seed, "parts": args.parts.split(","), "out": str(args.out)})
    specs.sort(key=lambda s: -ARMS[s["arm"]][1])
    with Pool(args.workers, maxtasksperchild=1) as pool:
        pool.map(job, specs, chunksize=1)


if __name__ == "__main__":
    main()
