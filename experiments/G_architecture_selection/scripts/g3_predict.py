"""G3 predictive evaluation of saved models on an exploration partition (live, reset and shuffled histories), and the
lifetime-clustered relative Brier of each kind vs `none` (programme F's E1 statistics).

    python scripts/g3_predict.py --data runs/G3/data --models runs/G3/models --part X --workers 30
    python scripts/g3_predict.py --analyze --models runs/G3/models --part X
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g3_train import with_apm  # noqa: E402

from apm_research.stats import holm, relative_loss_difference  # noqa: E402
from apm_select.provenance import write_json  # noqa: E402
from apm_select.stateful import KINDS, StatefulPredictor  # noqa: E402

MARGIN = 0.05


def job(spec: dict) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    saved = torch.load(spec["model"], weights_only=False)
    m = StatefulPredictor.load(saved)
    lts = with_apm(pickle.loads(Path(spec["data"]).read_bytes()), m.kind)
    preds = {}
    for ctrl in ([None, "reset", "shuffle"] if m.kind != "none" else [None]):
        rows = []
        for i, lt in enumerate(lts):
            other = lts[(i + 1) % len(lts)]["Z"] if ctrl == "shuffle" else None
            o = m.predict_lifetime(lt["Z"], lt["A"], control="reset" if ctrl == "reset" else None, shuffle_from=other)
            rows.append(o[lt["mask"]])
        preds[f"{spec['part']}_{ctrl or 'live'}"] = np.concatenate(rows).astype(np.float32)
    preds[f"{spec['part']}_y"] = np.concatenate([lt["y_col"][lt["mask"]] for lt in lts])
    preds[f"{spec['part']}_g"] = np.concatenate([np.full(lt["mask"].sum(), lt["index"]) for lt in lts])
    dest = Path(spec["model"]).with_name(Path(spec["model"]).stem + f"_{spec['part']}_preds.npz")
    np.savez_compressed(dest, **preds)
    return str(dest)


def analyze(models: Path, part: str) -> dict:
    """Relative Brier (collision) of each kind and control vs `none` (5 seeds each, two-level bootstrap)."""
    data = {}
    for f in sorted(models.glob(f"*_s?_{part}_preds.npz")):
        kind = f.name.split("_s")[0]
        data.setdefault(kind, []).append(np.load(f))
    y, g = data["none"][0][f"{part}_y"], data["none"][0][f"{part}_g"]
    sq = lambda runs, key: np.stack([(r[key][:, 0] - y) ** 2 for r in runs])  # noqa: E731
    base = sq(data["none"], f"{part}_live")
    out, pv = {}, {}
    for kind in KINDS:
        if kind == "none" or kind not in data:
            continue
        for ctrl in ("live", "reset", "shuffle"):
            c = relative_loss_difference(sq(data[kind], f"{part}_{ctrl}"), base, g)
            c.pop("draws")
            out[f"{kind}_{ctrl}"] = c
            if ctrl == "live":
                pv[kind] = c["p_not_lower"]
    ph = holm(pv)
    for k, p in ph.items():
        c = out[f"{k}_live"]
        c["p_holm"] = p
        lo, hi = c["ci95"]
        c["decision"] = ("adds information" if c["relative_difference"] <= -MARGIN and p < 0.05 else
                         "no meaningful information" if -MARGIN < lo and hi < MARGIN else "inconclusive")
    out["brier_none"] = float(base.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/G3/data"))
    ap.add_argument("--models", type=Path, default=Path("runs/G3/models"))
    ap.add_argument("--part", default="X")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        res = analyze(args.models, args.part)
        print(write_json(HERE / "results" / "G3" / f"g3_predictive_{args.part}.json", res))
        print(json.dumps({k: (round(v["relative_difference"], 4), [round(x, 4) for x in v["ci95"]], v.get("decision"))
                          for k, v in res.items() if isinstance(v, dict)}, indent=1))
        return
    specs = [{"model": str(f), "data": str(args.data / f"{args.part}.pkl"), "part": args.part}
             for f in sorted(args.models.glob("*_s?.pt"))]
    with Pool(args.workers, maxtasksperchild=1) as pool:
        print(pool.map(job, specs, chunksize=1))


if __name__ == "__main__":
    main()
