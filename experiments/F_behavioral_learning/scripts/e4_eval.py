"""E4 evaluation: trained B3/B4 variants (greedy, best-validation checkpoint) and the B1/B2 references on the same
E4 lifetimes (common random numbers). Partition X needs the committed E4 addendum.

    python scripts/e4_eval.py --partition P --n 40 --models runs/E4/pilot --seeds 0
    python scripts/e4_eval.py --partition X --n <addendum> --models runs/E4/X --seeds 0,1,2,3,4

Evaluation-only negative controls on B3 networks: B3_shuffledIdentity (memory inputs permuted across tracked people
each decision) and B3_historyReset (latent reset at every decision).
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import seeds as S  # noqa: E402
from e2_common import _STATE, _init, make_policy  # noqa: E402
from e4_train import VAL_EPISODES, policy_for, variant_spec  # noqa: E402

from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.provenance import confirmatory_lock, environment, sha256, write_json  # noqa: E402
from apm_research.rl import make_net  # noqa: E402

EPISODES = VAL_EPISODES  # test lifetimes have the training/validation length (5 × 60 s)
CONTROLS = {"B3_shuffledIdentity": ("B3", {"shuffle_identity": True}),
            "B3_historyReset": ("B3", {"reset_every_decision": True})}


def _job(job: dict) -> dict:
    torch.set_num_threads(1)
    lt = Lifetime(S.world_seed("E4", job["partition"], job["index"]), n_episodes=EPISODES)
    if "model" in job:
        spec = variant_spec(job["variant"])
        net = make_net(spec)
        net.load_state_dict(torch.load(job["model"], map_location="cpu"))
        net.eval()
        policy = policy_for(net, spec, _STATE["predictor"], job["lam"], **job.get("kw", {}))
    else:
        policy = make_policy(job["spec"])
    r = run_lifetime(lt, policy)
    r.pop("episodes")
    return {"label": job["label"], "seed": job.get("seed", 0), "index": job["index"], "result": r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--variants", default="B3,B4,B3_noMemory,B3_noPersist,B3_fastOnly,B3_slowOnly,B4_half,B4_double")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--tuned", type=Path, default=HERE / "configs" / "E2_tuned.json")
    ap.add_argument("--out", type=Path, default=Path("runs/E4"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    provenance = (confirmatory_lock(HERE / "prereg" / "E4_ADDENDUM.md") if args.partition in ("X", "S")
                  else environment())
    start = time.time()
    tuned = json.loads(args.tuned.read_text())["best"]
    lam = {"lam_c": tuned["B1"]["lam_c"], "lam_s": tuned["B1"]["lam_s"]}
    refs = {"B1": {"system": "B1", **lam, "label": "B1"},
            "B2": {"system": "B2", **{k: tuned["B2"][k] for k in ("lam_c", "lam_s", "k_pad", "k_t")}, "label": "B2"}}
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    models, jobs = {}, []
    for v in variants:
        for s in seeds:
            path = args.models / f"{v}_s{s}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            models[path.name] = sha256(path)
    for i in range(args.n):
        jobs += [{"label": k, "spec": spec, "partition": args.partition, "index": i} for k, spec in refs.items()]
        for v in variants:
            jobs += [{"label": v, "variant": v, "seed": s, "model": str(args.models / f"{v}_s{s}.pt"), "lam": lam,
                      "partition": args.partition, "index": i} for s in seeds]
        if "B3" in variants:
            jobs += [{"label": c, "variant": base, "seed": s, "model": str(args.models / f"{base}_s{s}.pt"),
                      "lam": lam, "kw": kw, "partition": args.partition, "index": i}
                     for c, (base, kw) in CONTROLS.items() for s in seeds]
    jobs.sort(key=lambda j: j["label"] != "B2")  # slowest first
    out_dir = args.out / f"eval_{args.partition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init, initargs=(args.predictor,)) as pool, \
            open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(_job, jobs, chunksize=1):
            fh.write(json.dumps(r, default=float) + "\n")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "n": args.n, "episodes": EPISODES,
                                           "variants": variants, "seeds": seeds, "references": refs,
                                           "model_sha256": models, "jobs": len(jobs),
                                           "wall_s": round(time.time() - start, 1)})
    print("done", len(jobs), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
