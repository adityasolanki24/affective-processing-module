"""G3 training: every predictor kind × seed on G3-T exploration lifetimes, early stopping on G3-V; saves each model
(torch state + normalisation) and its predictions on the evaluation partitions given (incl. controls).

    python scripts/g3_train.py --data runs/G3/data --out runs/G3/models --parts P --workers 30
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
from apm_select.stateful import KINDS, StatefulPredictor  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def with_apm(lts: list[dict], kind: str) -> list[dict]:
    """The model input Z: features (41), plus the APM readout (8) for the `pad` kind only."""
    return [{**lt, "Z": np.concatenate([lt["X"], lt["APM"]], 1) if kind == "pad" else lt["X"]} for lt in lts]


def job(spec: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    data, kind, seed = Path(spec["data"]), spec["kind"], spec["seed"]
    load = {p: with_apm(pickle.loads((data / f"{p}.pkl").read_bytes()), kind) for p in ["T", "V", *spec["parts"]]}
    start = time.time()
    width = load["T"][0]["Z"].shape[1]
    m = StatefulPredictor(kind, width, seed).fit(load["T"], load["V"])
    out = {"kind": kind, "seed": seed, "params": m.parameters, "train_s": round(time.time() - start, 1),
           "epochs": len(m.history), "val_loss": min(h["val_loss"] for h in m.history)}
    dest = Path(spec["out"])
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({"state": m.net.state_dict(), "mean": m.mean, "scale": m.scale, "kind": kind, "width": width,
                "seed": seed}, dest / f"{kind}_s{seed}.pt")
    preds = {}
    controls = [None, "reset", "shuffle"] if kind != "none" else [None]
    for p in spec["parts"]:
        lts = load[p]
        for ctrl in controls:
            rows = []
            for i, lt in enumerate(lts):
                other = lts[(i + 1) % len(lts)]["Z"] if ctrl == "shuffle" else None
                o = m.predict_lifetime(lt["Z"], lt["A"], control="reset" if ctrl == "reset" else None,
                                       shuffle_from=other)
                rows.append(o[lt["mask"]])
            tag = ctrl or "live"
            preds[f"{p}_{tag}"] = np.concatenate(rows).astype(np.float32)
        preds[f"{p}_y"] = np.concatenate([lt["y_col"][lt["mask"]] for lt in lts])
        preds[f"{p}_g"] = np.concatenate([np.full(lt["mask"].sum(), lt["index"]) for lt in lts])
    np.savez_compressed(dest / f"{kind}_s{seed}_preds.npz", **preds)
    (dest / f"{kind}_s{seed}.json").write_text(json.dumps(out))
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/G3/data"))
    ap.add_argument("--out", type=Path, default=Path("runs/G3/models"))
    ap.add_argument("--parts", default="P")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    specs = [{"data": str(args.data), "kind": k, "seed": int(s), "parts": args.parts.split(","), "out": str(args.out)}
             for k in args.kinds.split(",") for s in args.seeds.split(",")]
    specs.sort(key=lambda s: s["kind"] not in ("gru", "fastslow", "lds"))  # slow ones first
    with Pool(args.workers, maxtasksperchild=1) as pool:
        pool.map(job, specs, chunksize=1)


if __name__ == "__main__":
    main()
