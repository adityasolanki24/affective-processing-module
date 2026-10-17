"""G4/G5 training: one run = (architecture, method, seed). Interaction budgets count every environment decision
used to obtain the model (teacher rollouts included). Validation: greedy mean J on G4-V lifetimes 0–7 every
`--eval-every` decisions; the best checkpoint is kept (J only, §4.3). Training lifetimes: G4-T, 5 episodes.

Architectures: A2 ordinary GRU actor-critic without explicit-memory inputs; A3 the same with them (F's B4);
A4 bounded learned residual on the A0 utility (F's B3 network, anchored at the fairly tuned A0, starts = A0).
Methods: ppo (from scratch), bc (teacher rollouts, cross-entropy), bcppo (bc data then PPO), dagger
(β = 0.5^iteration mixtures, every visited state labelled by the teacher, aggregated).

    python scripts/g45_train.py --arch A3 --method ppo --seed 0 --budget 2400000 --collect-workers 2
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

from apm_research.harness import EPISODE_GAP_S, Lifetime, run_episode, run_lifetime  # noqa: E402
from apm_research.predictor import EnsemblePredictor  # noqa: E402
from apm_research.provenance import environment, write_json  # noqa: E402
from apm_research.rl import B3Config, B3Net, B4Net, RewardProbe, b4_hidden, gae, parameters, ppo_update  # noqa: E402
from apm_select.learning import Learner, bc_loss, labelled_arrays, teacher_from_spec  # noqa: E402
from apm_testbeds.wheeled_ctrl import child  # noqa: E402

TRAIN_EPISODES, VAL_EPISODES, BATCH = 5, 5, 8
BC_DECISIONS = 300_000
_W: dict = {}


def arch_net(arch: str):
    if arch == "A4":
        return B3Net(B3Config())
    return B4Net(b4_hidden(parameters(B3Net(B3Config()))))


def teacher_spec() -> dict:
    g1 = json.loads((HERE / "configs" / "G1_selected.json").read_text())
    return g1["systems"][g1["best_simple"]]


def _winit(predictor: str, arch: str) -> None:
    torch.set_num_threads(1)
    _W.update(pred=EnsemblePredictor.from_dict(json.loads(Path(predictor).read_text())), arch=arch,
              net=arch_net(arch), teacher=teacher_spec())


def make_learner(net, mode: str, seed, beta: float = 0.0) -> Learner:
    return Learner(net, teacher_from_spec(_W["pred"], _W["teacher"]), mode=mode,
                   memory_features=_W["arch"] != "A2", beta=beta, seed=seed)


def rollout(job: tuple) -> dict:
    """job = (state_dict, lifetime index, run seed, mode, beta) → PPO trajectory or labelled data."""
    state, index, seed, mode, beta = job
    _W["net"].load_state_dict(state)
    ss = S.world_seed("G4", "T", index)
    lt = Lifetime(ss, n_episodes=TRAIN_EPISODES)
    pol = make_learner(_W["net"], mode, child(ss, 700 + seed), beta)
    if mode == "sample":
        probe = RewardProbe(pol)
        probe.begin_lifetime()
        for e in range(TRAIN_EPISODES):
            run_episode(lt.world(e), probe, e * (lt.config.episode_s + EPISODE_GAP_S))
        z, u, a, logp, v, reset = (np.array(c) for c in zip(*pol.trajectory, strict=True))
        r = np.array(probe.rewards, np.float32)
        adv, ret = gae(r, v.astype(np.float32))
        return {"z": z.astype(np.float32), "u": u.astype(np.float32), "a": a.astype(np.int64),
                "logp": logp.astype(np.float32), "adv": adv.astype(np.float32), "ret": ret.astype(np.float32),
                "reset": reset.astype(bool), "J": float(r.sum())}
    run_lifetime(lt, pol)
    return labelled_arrays(pol)


def validate(state, n: int) -> float:
    _W["net"].load_state_dict(state)
    _W["net"].eval()
    total = 0.0
    for i in range(n):
        lt = Lifetime(S.world_seed("G4", "V", i), n_episodes=VAL_EPISODES)
        total += run_lifetime(lt, make_learner(_W["net"], "greedy", 0))["J"]
    _W["net"].train()
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=("A2", "A3", "A4"))
    ap.add_argument("--method", required=True, choices=("ppo", "bc", "bcppo", "dagger"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=2_400_000)
    ap.add_argument("--eval-every", type=int, default=96_000)
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--collect-workers", type=int, default=1)
    ap.add_argument("--bc-decisions", type=int, default=BC_DECISIONS)
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--out", type=Path, default=Path("runs/G45/models"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    _winit(args.predictor, args.arch)
    net, start = _W["net"], time.time()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    name = f"{args.arch}_{args.method}_s{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    pool = None
    if args.collect_workers > 1:
        from multiprocessing import Pool
        pool = Pool(args.collect_workers, initializer=_winit, initargs=(args.predictor, args.arch))
    index = args.seed * 1_000_000
    used, curve, best = 0, [], {"J": -1e9, "decisions": 0}

    def collect(mode: str, beta: float = 0.0, k: int = BATCH) -> list[dict]:
        nonlocal index
        state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        jobs = [(state, index + i, args.seed, mode, beta) for i in range(k)]
        index += k
        return pool.map(rollout, jobs) if pool else [rollout(j) for j in jobs]

    def checkpoint() -> None:
        state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        j = validate(state, args.val_n)
        curve.append({"decisions": used, "val_J": j, "wall_s": round(time.time() - start, 1)})
        if j > best["J"]:
            best.update(J=j, decisions=used)
            torch.save(net.state_dict(), args.out / f"{name}.pt")
        print(name, used, "val J", round(j, 2), round(time.time() - start), flush=True)

    def fit_bc(data: list[dict], epochs: int) -> None:
        for _ in range(epochs):
            for i in np.random.default_rng(args.seed + len(curve)).permutation(len(data)):
                opt.zero_grad()
                bc_loss(net, [data[i]]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

    checkpoint()
    if args.method in ("bc", "bcppo", "dagger"):
        data: list[dict] = []
        rounds = 3 if args.method == "dagger" else 1
        per_round = min(args.bc_decisions, args.budget) // rounds
        for r in range(rounds):
            mode, beta = ("teacher", 1.0) if r == 0 or args.method != "dagger" else ("dagger", 0.5 ** r)
            got = 0
            while got < per_round:
                batch = collect(mode, beta)
                data += batch
                got += sum(len(b["a"]) for b in batch)
            used += got
            fit_bc(data, epochs=4)
            checkpoint()
    if args.method in ("ppo", "bcppo"):
        updates = 0
        while used < args.budget:
            batch = collect("sample")
            used += sum(len(b["a"]) for b in batch)
            ppo_update(net, opt, batch)
            updates += 1
            if used // args.eval_every != (used - sum(len(b["a"]) for b in batch)) // args.eval_every \
                    or used >= args.budget:
                checkpoint()
    if pool is not None:
        pool.close()
    area = float(np.trapezoid([c["val_J"] for c in curve], [c["decisions"] for c in curve]) / max(1, used)) \
        if len(curve) > 1 else curve[0]["val_J"]
    write_json(args.out / f"{name}.json", {"arch": args.arch, "method": args.method, "seed": args.seed,
                                           "params": parameters(net), "budget": args.budget, "decisions_used": used,
                                           "best": best, "curve": curve, "curve_area_per_decision": area,
                                           "teacher": teacher_spec(), "environment": environment(),
                                           "wall_s": round(time.time() - start, 1)})


if __name__ == "__main__":
    main()
