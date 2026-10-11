"""E4 training: one run = (system variant, training seed). Recurrent PPO on E4 partition T lifetimes (5 episodes,
persistent memory and latent), checkpoint selection on E4 partition V by greedy tuning utility J only.

    python scripts/e4_train.py --variant B3 --seed 0 --budget 600000
    python scripts/e4_train.py --variant B4 --seed 0 --budget 600000

Variants: B3, B3_noMemory, B3_noPersist, B3_fastOnly, B3_slowOnly, B4, B4_half, B4_double.
Equal environment-step budgets: `--budget` counts decisions (each = 10 physics steps) for every variant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm_research.harness import EPISODE_GAP_S, Lifetime, run_episode, run_lifetime  # noqa: E402
from apm_research.predictor import EnsemblePredictor  # noqa: E402
from apm_research.provenance import environment, write_json  # noqa: E402
from apm_research.rl import (  # noqa: E402
    B3Config,
    B3Net,
    LearnedPolicy,
    RewardProbe,
    b4_hidden,
    gae,
    make_net,
    parameters,
    ppo_update,
)
from apm_testbeds.wheeled_ctrl import child  # noqa: E402

TRAIN_EPISODES, VAL_EPISODES = 5, 5


def variant_spec(variant: str) -> dict:
    b3 = {"B3": {}, "B3_noMemory": {"memory_features": False}, "B3_noPersist": {"persistent": False},
          "B3_fastOnly": {"slow": 0}, "B3_slowOnly": {"fast": 0}}
    if variant in b3:
        return {"kind": "B3", "b3": b3[variant]}
    target = parameters(B3Net(B3Config()))
    scale = {"B4": 1.0, "B4_half": 0.5, "B4_double": 2.0}[variant]
    return {"kind": "B4", "hidden": b4_hidden(target, scale)}


def policy_for(net, spec: dict, pred, lam: dict, **kw) -> LearnedPolicy:
    b3 = spec.get("b3", {})
    return LearnedPolicy(net, pred, lam["lam_c"], lam["lam_s"], persistent=b3.get("persistent", True),
                         memory_features=b3.get("memory_features", True), **kw)


def validate(net, spec, pred, lam, n: int) -> float:
    net.eval()
    total = 0.0
    for i in range(n):
        lt = Lifetime(S.world_seed("E4", "V", i), n_episodes=VAL_EPISODES)
        total += run_lifetime(lt, policy_for(net, spec, pred, lam))["J"]
    net.train()
    return total / n


def collect(net, spec, pred, lam, index: int, seed: int) -> dict:
    ss = S.world_seed("E4", "T", index)
    lt = Lifetime(ss, n_episodes=TRAIN_EPISODES)
    policy = policy_for(net, spec, pred, lam, sample=True, seed=child(ss, 700 + seed))
    probe = RewardProbe(policy)
    probe.begin_lifetime()
    for e in range(TRAIN_EPISODES):
        run_episode(lt.world(e), probe, e * (lt.config.episode_s + EPISODE_GAP_S))
    traj = policy.trajectory
    assert len(traj) == len(probe.rewards), (len(traj), len(probe.rewards))
    z, u, a, logp, v, reset = (np.array(c) for c in zip(*traj, strict=True))
    r = np.array(probe.rewards, np.float32)
    adv, ret = gae(r, v.astype(np.float32))
    return {"z": z.astype(np.float32), "u": u.astype(np.float32), "a": a.astype(np.int64),
            "logp": logp.astype(np.float32), "adv": adv.astype(np.float32), "ret": ret.astype(np.float32),
            "reset": reset.astype(bool), "J": float(r.sum())}


_W: dict = {}


def _winit(predictor: str, spec: dict, lam: dict) -> None:
    torch.set_num_threads(1)
    _W.update(pred=EnsemblePredictor.from_dict(json.loads(Path(predictor).read_text())), spec=spec, lam=lam,
              net=make_net(spec))


def _wcollect(job: tuple) -> dict:
    state, index, seed = job
    _W["net"].load_state_dict(state)
    return collect(_W["net"], _W["spec"], _W["pred"], _W["lam"], index, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=600_000, help="decisions")
    ap.add_argument("--batch", type=int, default=8, help="lifetimes per PPO update")
    ap.add_argument("--eval-every", type=int, default=10, help="updates")
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--tuned", type=Path, default=HERE / "configs" / "E2_tuned.json")
    ap.add_argument("--out", type=Path, default=Path("runs/E4"))
    ap.add_argument("--collect-workers", type=int, default=1,
                    help="parallel rollout collection; every lifetime is seeded independently, so results do not "
                         "depend on this")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.variant}_s{args.seed}"
    pred = EnsemblePredictor.from_dict(json.loads(Path(args.predictor).read_text()))
    b1 = json.loads(args.tuned.read_text())["best"]["B1"]
    lam = {"lam_c": b1["lam_c"], "lam_s": b1["lam_s"]}
    spec = variant_spec(args.variant)
    net = make_net(spec)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    start, decisions, updates, index = time.time(), 0, 0, args.seed * 1_000_000
    best = {"J": validate(net, spec, pred, lam, args.val_n), "decisions": 0}
    torch.save(net.state_dict(), args.out / f"{name}.pt")
    curve = [{"decisions": 0, "val_J": best["J"], "train_J": None}]
    print(name, "params", parameters(net), "init val J", round(best["J"], 2), flush=True)
    pool = None
    if args.collect_workers > 1:
        from multiprocessing import Pool
        pool = Pool(args.collect_workers, initializer=_winit, initargs=(args.predictor, spec, lam))
    while decisions < args.budget:
        if pool is None:
            batch = [collect(net, spec, pred, lam, index + k, args.seed) for k in range(args.batch)]
        else:
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            batch = pool.map(_wcollect, [(state, index + k, args.seed) for k in range(args.batch)])
        index += args.batch
        decisions += sum(len(b["a"]) for b in batch)
        stats = ppo_update(net, opt, batch)
        updates += 1
        if updates % args.eval_every == 0 or decisions >= args.budget:
            j = validate(net, spec, pred, lam, args.val_n)
            curve.append({"decisions": decisions, "val_J": j, "train_J": float(np.mean([b["J"] for b in batch])),
                          **stats, "wall_s": round(time.time() - start, 1)})
            if j > best["J"]:
                best = {"J": j, "decisions": decisions}
                torch.save(net.state_dict(), args.out / f"{name}.pt")
            print(name, decisions, "val J", round(j, 2), "train J", round(curve[-1]["train_J"], 2), stats,
                  round(time.time() - start), flush=True)
    if pool is not None:
        pool.close()
    write_json(args.out / f"{name}.json", {"variant": args.variant, "seed": args.seed, "spec": spec,
                                           "params": parameters(net), "budget": args.budget, "lam": lam,
                                           "best": best, "curve": curve, "environment": environment(),
                                           "wall_s": round(time.time() - start, 1)})


if __name__ == "__main__":
    main()
