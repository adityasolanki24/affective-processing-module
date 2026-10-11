"""Milestone D: portability (same core, two bodies) and cross-embodiment transfer (protocol: PREREGISTRATION.md).

    python experiments/D_portability/run_portability.py --nerva-experience runs/nerva_experience --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.wheeled import WheeledAdapter
from apm.learning.baselines import Logistic
from apm.learning.dataset import episode_from_stream, samples
from apm.learning.evaluation import auroc, brier, summary
from apm.learning.predictor import GRUPredictor
from apm.protocol import validate
from apm.protocol.messages import OutcomeSignal
from apm_testbeds.wheeled import WheeledWorld

ROOT = Path(__file__).resolve().parents[2]
STRIDE, HISTORY, HORIZON, COLLISION_HORIZON = 0.1, 20, 0.5, 1.0
WHEELED_SECONDS, WHEELED_EPISODES, WHEELED_SEED0 = 60.0, 400, 10_000
NS, SEEDS = (5, 20, 80, 240), (0, 1, 2)
CORE_DIRS = ("src/apm/core", "src/apm/memory", "src/apm/protocol")


def core_hashes() -> list[str]:
    rows = []
    for d in CORE_DIRS:
        for f in sorted((ROOT / d).rglob("*.py")):
            rows.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT).as_posix()}")
    return sorted(rows, key=lambda r: r.split()[1])


def wheeled_stream(seed: int, apm: APM | None = None, stats: collections.Counter | None = None):
    world = WheeledWorld(seed)
    adapter = WheeledAdapter(Emitter(f"wheeled-{seed}"))
    yield 0.0, [adapter.profile()]
    for _ in range(int(WHEELED_SECONDS / 0.05)):
        tel = world.step()
        msgs = adapter.step(tel)
        if apm is not None:
            for m in msgs:
                validate(m)
                reason = apm.ingest(m)
                if stats is not None:
                    stats["rejected" if reason else "accepted"] += 1
                    if reason is None and isinstance(m, OutcomeSignal):
                        stats[m.kind] += 1
        yield tel.t, msgs


def wheeled_episodes():
    out = []
    for i in range(WHEELED_EPISODES):
        seed = WHEELED_SEED0 + i
        profile = WheeledAdapter(Emitter(f"wheeled-{seed}")).profile()
        out.append(episode_from_stream(str(seed), profile, wheeled_stream(seed), STRIDE, meta={"seed": seed}))
    return out


def nerva_episodes(experience: Path, cache: Path):
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    spec = importlib.util.spec_from_file_location("run_learning", ROOT / "experiments/C_learning/run_learning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episodes = module.build_episodes(experience)
    cache.write_bytes(pickle.dumps(episodes))
    return episodes


def permute(episodes, seed=0):
    ids = np.random.default_rng(seed).permutation(len(episodes))
    n = len(ids)
    return ([episodes[i] for i in ids[:int(.6 * n)]], [episodes[i] for i in ids[int(.6 * n):int(.8 * n)]],
            [episodes[i] for i in ids[int(.8 * n):]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerva-experience", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/D_portability/results"))
    ap.add_argument("--cache", type=Path, default=Path("runs/nerva_episodes.pkl"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    results: dict = {"core_hashes_start": core_hashes()}

    # ---------------- portability demonstration: the same APM class on the wheeled body
    stats: collections.Counter = collections.Counter()
    demo = APM(profile=WheeledAdapter(Emitter("wheeled-demo")).profile())
    trajectory = []
    for t, _ in wheeled_stream(WHEELED_SEED0 + 999, demo, stats):
        if t > 0 and int(round(t / 0.05)) % 10 == 0:
            out = demo.step(t)
            trajectory.append([round(t, 2), *out.affect_vector().round(3).tolist(),
                               max((x.avoid for x in out.tendencies), default=0.0)])
    results["wheeled_demo"] = {"message_stats": dict(stats), "affect_trajectory_1hz": trajectory,
                               "entity_memory_records": len(demo.entities.records),
                               "diagnostics_health": demo.step(WHEELED_SECONDS + 0.1).diagnostics.health}
    print("portability demo", dict(stats), flush=True)

    # ---------------- data
    wheeled = wheeled_episodes()
    counts = {k: sum(len(e.events.get(k, [])) for e in wheeled) for k in
              ("collision", "stability_loss", "recovery", "task_completed", "task_interrupted", "resource_critical")}
    print("wheeled episodes", len(wheeled), counts, f"{time.monotonic() - start:.0f}s", flush=True)
    w_pool, w_val, w_test = permute(wheeled)
    nerva = nerva_episodes(args.nerva_experience, args.cache)
    n_train, n_val, n_test = permute(nerva)
    results["data"] = {"wheeled_event_counts": counts, "wheeled_split": [len(w_pool), len(w_val), len(w_test)],
                       "nerva_episodes": len(nerva)}

    # ---------------- NERVA model (Milestone C M4 procedure)
    Xn, yn, _, _ = samples(n_train, "stability_loss", HORIZON, HISTORY)
    Xnv, ynv, _, _ = samples(n_val, "stability_loss", HORIZON, HISTORY)
    Xnt, ynt, _, _ = samples(n_test, "stability_loss", HORIZON, HISTORY)
    nerva_model = GRUPredictor(Xn.shape[2], hidden=64, seed=0).fit(Xn, yn, Xnv, ynv)
    results["nerva_model_on_nerva_test"] = summary(nerva_model.predict(Xnt), ynt)

    # ---------------- T1 transfer arms on the wheeled body
    Xwv, ywv, _, _ = samples(w_val, "stability_loss", HORIZON, HISTORY)
    Xwt, ywt, _, _ = samples(w_test, "stability_loss", HORIZON, HISTORY)
    zero = nerva_model.predict(Xwt)
    results["T1_zero_shot"] = summary(zero, ywt)
    arms = {}
    for n in NS:
        for seed in SEEDS:
            subset = [w_pool[i] for i in np.random.default_rng(100 + seed).permutation(len(w_pool))[:n]]
            X, y, _, _ = samples(subset, "stability_loss", HORIZON, HISTORY)
            scratch = GRUPredictor(X.shape[2], hidden=64, seed=seed).fit(X, y, Xwv, ywv)
            tuned = nerva_model.clone()
            tuned.lr = 1e-3
            tuned.rng = np.random.default_rng(seed)
            tuned.fit(X, y, Xwv, ywv)
            logistic = Logistic().fit(X, y)
            for arm, model in (("A_scratch", scratch), ("C_finetuned", tuned), ("D_logistic", logistic)):
                p = model.predict(Xwt)
                arms[f"{arm}|N={n}|seed={seed}"] = {"brier": brier(p, ywt), "auroc": auroc(p, ywt),
                                                    "train_samples": int(len(y)), "train_positives": int(y.sum())}
        print("N", n, "done", f"{time.monotonic() - start:.0f}s", flush=True)
    results["T1_arms"] = arms

    def mean(arm, n, key="brier"):
        return float(np.mean([arms[f"{arm}|N={n}|seed={s}"][key] for s in SEEDS]))

    def wins(n):
        return sum(arms[f"C_finetuned|N={n}|seed={s}"]["brier"] < arms[f"A_scratch|N={n}|seed={s}"]["brier"]
                   for s in SEEDS)

    criteria = {"P1_core_unchanged": None, "P2_no_protocol_rejections": stats["rejected"] == 0,
                "TR1_zero_shot_auroc_ge_0.75": bool(results["T1_zero_shot"]["auroc"] >= 0.75),
                "TR2_sample_efficiency": bool(all(wins(n) >= 2 and mean("C_finetuned", n) <= 0.9 * mean("A_scratch", n)
                                                  for n in (5, 20))),
                "TR3_no_harm_full_data": bool(mean("C_finetuned", 240) <= 1.05 * mean("A_scratch", 240))}
    results["T1_means"] = {f"{arm}|N={n}": {"brier": mean(arm, n), "auroc": mean(arm, n, "auroc")}
                           for arm in ("A_scratch", "C_finetuned", "D_logistic") for n in NS}

    # ---------------- T2 collision (wheeled only)
    Xc, yc, _, _ = samples(w_pool, "collision", COLLISION_HORIZON, HISTORY)
    Xcv, ycv, _, _ = samples(w_val, "collision", COLLISION_HORIZON, HISTORY)
    Xct, yct, _, _ = samples(w_test, "collision", COLLISION_HORIZON, HISTORY)
    gru_c = GRUPredictor(Xc.shape[2], hidden=64, seed=0).fit(Xc, yc, Xcv, ycv)
    results["T2_collision"] = {"gru": summary(gru_c.predict(Xct), yct),
                               "logistic": summary(Logistic().fit(Xc, yc).predict(Xct), yct)}

    results["core_hashes_end"] = core_hashes()
    criteria["P1_core_unchanged"] = results["core_hashes_start"] == results["core_hashes_end"]
    results["criteria"] = criteria
    results["learned_transfer_claimed"] = bool(criteria["TR1_zero_shot_auroc_ge_0.75"]
                                               and criteria["TR2_sample_efficiency"])
    results["wall_seconds"] = round(time.monotonic() - start, 1)
    (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({"criteria": criteria, "T1_zero_shot": results["T1_zero_shot"], "T1_means": results["T1_means"],
                      "T2": results["T2_collision"]}, indent=1))


if __name__ == "__main__":
    main()
