"""G4/G5 test evaluation: every trained (architecture, method, seed) greedy at its best validation checkpoint, plus the
rule references (A0 = the shadow teacher's rule; A1 = G1's fairly tuned APM v0.1 / B2), on the same G4 lifetimes.

    python scripts/g45_eval.py --partition X --n 400 --models runs/G45/models --workers 96
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
import torch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def _load_g_seeds():
    """Programme G's seeds.py under a unique module name (programme F also has a `seeds` module)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("g_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("g_seeds", mod)
    spec.loader.exec_module(mod)
    return sys.modules["g_seeds"] if sys.modules["g_seeds"] is not mod else mod


S = _load_g_seeds()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from g45_train import _W, VAL_EPISODES, _winit, arch_net, make_learner  # noqa: E402
from g_common import RESULT_KEYS, init, make_policy  # noqa: E402

from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_select.instrument import Instrumented, collision_split  # noqa: E402
from apm_select.provenance import confirmatory_lock, environment, sha256, write_json  # noqa: E402

EPISODES = VAL_EPISODES
_NETS: dict = {}


def _init(models: str, predictor: str) -> None:
    init(predictor)
    torch.set_num_threads(1)
    _winit(predictor, "A3")
    for f in Path(models).glob("*_s?.pt"):
        arch = f.stem.split("_")[0]
        net = arch_net(arch)
        net.load_state_dict(torch.load(f, map_location="cpu"))
        net.eval()
        _NETS[f.stem] = (arch, net)


def job(j: dict) -> dict:
    lt = Lifetime(S.world_seed("G4", j["partition"], j["index"]), n_episodes=EPISODES)
    if "model" in j:
        arch, net = _NETS[j["model"]]
        _W["arch"] = arch
        inner = make_learner(net, "greedy", 0)
        label = j["model"].rsplit("_s", 1)[0]
        seed = int(j["model"].rsplit("_s", 1)[1])
    else:
        inner, label, seed = make_policy(j["spec"]), j["label"], 0
    pol = Instrumented(inner)
    r = run_lifetime(lt, pol)
    return {"label": label, "seed": seed, "index": j["index"], "partition": j["partition"],
            "result": {k: float(r[k]) for k in RESULT_KEYS} | {k: float(v) for k, v in pol.stats.items()}
            | collision_split(r),
            "actions": r["actions"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--models", type=Path, default=Path("runs/G45/models"))
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--addendum", default="G45_ADDENDUM.md")
    ap.add_argument("--only", default="", help="comma list of model cells to evaluate (default: all)")
    ap.add_argument("--out", type=Path, default=Path("runs/G45"))
    args = ap.parse_args()
    provenance = confirmatory_lock(HERE / "prereg" / args.addendum) if args.partition in ("X", "S") \
        else environment()
    start = time.time()
    g1 = json.loads((HERE / "configs" / "G1_selected.json").read_text())
    refs = {"A0": g1["systems"][g1["best_simple"]], "A1": g1["systems"]["R6"]}
    models = sorted(f.stem for f in args.models.glob("*_s?.pt")
                    if not args.only or f.stem.rsplit("_s", 1)[0] in args.only.split(","))
    idx = range(args.first, args.first + args.n)
    jobs = [{"partition": args.partition, "index": i, "spec": s, "label": k} for i in idx for k, s in refs.items()]
    jobs += [{"partition": args.partition, "index": i, "model": m} for i in idx for m in models]
    jobs.sort(key=lambda j: j.get("label") != "A1")  # slowest first
    out_dir = args.out / f"eval_{args.partition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init, initargs=(str(args.models), args.predictor)) as pool, \
            open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(job, jobs, chunksize=4):
            fh.write(json.dumps(r) + "\n")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "n": args.n, "episodes": EPISODES,
                                           "references": refs, "models": {m: sha256(args.models / f"{m}.pt")
                                                                          for m in models},
                                           "wall_s": round(time.time() - start, 1)})
    print("done", len(jobs), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
