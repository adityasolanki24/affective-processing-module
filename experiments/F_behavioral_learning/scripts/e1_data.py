"""E1 datasets: canonical features (as C/D) plus extra input blocks computed from the *same* full-rate message stream.

Per 10 Hz sample, `extra` columns (layout fixed in `prereg/E1_ADDENDUM.md`):
  RISK  0–9    window summary of the full-rate stream since the previous sample: max / mean live hypothesis
               probability, nearest-entity proximity exp(−d), max closing/2, counts of collision, stability_loss,
               recovery, task_completed, task_interrupted, fall
  LONG  10–21  exponentially weighted averages (τ = 3, 30, 120 s) of [max p, collision count, stability_loss count,
               proximity]: the long-history control for persistent affect
  U     22–26  APM tonic appraisal vector u (threat, promise, strain, control, unknown)
  PAD   27–29  APM affect (V, A, D)
  MEM   30–32  APM memory expectation for the nearest entity (p_adverse, confidence, familiarity); 0 without entities
  NOISE 33–40  standard-normal noise (input-width control), from its own seeded stream
The APM runs observe-only on the stream and is stepped at the control rate (as in B2). Ground truth used only for
evaluation (nearest person's disposition) is stored in `meta["truth"]`, never in `features`/`extra`.

    python scripts/e1_data.py --embodiment wheeled --partition T --n 100 --out runs/E1
    python scripts/e1_data.py --embodiment nerva --partition T --experience runs/E1/nerva/T --out runs/E1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm import APM  # noqa: E402
from apm.adapters.base import Emitter  # noqa: E402
from apm.adapters.nerva import NervaAdapter  # noqa: E402
from apm.adapters.wheeled_ctrl import WheeledCtrlAdapter  # noqa: E402
from apm.learning.dataset import episode_from_stream  # noqa: E402
from apm.protocol.messages import EntityObservation, OutcomeHypothesis, OutcomeSignal  # noqa: E402
from apm_research.harness import EPISODE_GAP_S, Lifetime  # noqa: E402
from apm_testbeds.wheeled_ctrl import GO, WorldConfig, child  # noqa: E402

KINDS = ("collision", "stability_loss", "recovery", "task_completed", "task_interrupted", "fall")
TAUS = (3.0, 30.0, 120.0)
BLOCKS = {"RISK": (0, 10), "LONG": (10, 22), "U": (22, 27), "PAD": (27, 30), "MEM": (30, 33), "NOISE": (33, 41)}
EXTRA_WIDTH = 41
STRIDE = 0.1
EPISODES_PER_LIFETIME = 5
SHIFT = WorldConfig(people=(4, 6), ped_speed=(0.4, 1.3), patches=(2, 4))


class Extras:
    def __init__(self, apm: APM, noise_seed: np.random.SeedSequence):
        self.apm = apm
        self.noise = np.random.default_rng(noise_seed)
        self.long = np.zeros((len(TAUS), 4))
        self.last_t: float | None = None
        self._reset()

    def _reset(self) -> None:
        self.ps: list[float] = []
        self.min_d = math.inf
        self.max_closing = -math.inf
        self.counts = dict.fromkeys(KINDS, 0)
        self.nearest: tuple[float, str | None, str] | None = None

    def ingest(self, messages: list, t: float) -> None:
        for m in messages:
            self.apm.ingest(m)
            if isinstance(m, OutcomeHypothesis):
                self.ps.append(m.probability)
            elif isinstance(m, EntityObservation) and m.distance.usable:
                d = float(m.distance.value)
                if d < self.min_d:
                    self.min_d = d
                    self.nearest = (d, m.identity, m.entity_type)
                if m.closing_speed.usable:
                    self.max_closing = max(self.max_closing, float(m.closing_speed.value))
            elif isinstance(m, OutcomeSignal) and m.kind in self.counts:
                self.counts[m.kind] += 1
        self.out = self.apm.step(t)

    def sample(self, t: float) -> np.ndarray:
        live = [p for p in self.ps if p > 0]
        prox = math.exp(-self.min_d) if math.isfinite(self.min_d) else 0.0
        risk = [max(self.ps, default=0.0), float(np.mean(live)) if live else 0.0, prox,
                float(np.clip(self.max_closing / 2.0, -1, 1)) if math.isfinite(self.max_closing) else 0.0,
                *[float(self.counts[k]) for k in KINDS]]
        dt = STRIDE if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        current = np.array([risk[0], self.counts["collision"], self.counts["stability_loss"], prox])
        for i, tau in enumerate(TAUS):
            a = 1.0 - math.exp(-dt / tau)
            self.long[i] += a * (current - self.long[i])
        mem = [0.0, 0.0, 0.0]
        if self.nearest is not None:
            e = self.apm.entities.expectation(self.nearest[1], self.nearest[2])
            mem = [e["p_adverse"], e["confidence"], e["familiarity"]]
        vec = np.concatenate([risk, self.long.ravel(), self.apm.affect.u, self.out.affect_vector(), mem,
                              self.noise.standard_normal(8)])
        self._reset()
        return vec.astype(np.float32)


def wheeled_lifetime(job: tuple[str, int]) -> list:
    partition, index = job
    ss = S.world_seed("E1", partition, index)
    lt = Lifetime(ss, n_episodes=EPISODES_PER_LIFETIME, config=SHIFT if partition == "S" else WorldConfig(),
                  p_careless=0.6 if partition == "S" else 0.4)
    adapter = WheeledCtrlAdapter(Emitter(f"wheeled-{partition}{index}"))
    apm = APM(profile=adapter.profile())
    apm.ingest(adapter.profile(), now_s=0.0)
    extras = Extras(apm, child(ss, 999))
    episodes = []
    for e in range(lt.n_episodes):
        world = lt.world(e)
        t0 = e * (lt.config.episode_s + EPISODE_GAP_S)
        truth: list[tuple[float, float]] = []

        def stream(world=world, t0=t0, e=e):
            yield t0, [adapter.profile()] if e == 0 else []
            while not world.done:
                obs = world.step(GO)
                msgs = adapter.step(obs, t0 + obs.t)
                extras.ingest(msgs, t0 + obs.t)
                yield t0 + obs.t, msgs

        def on_sample(t, world=world, truth=truth):
            tr = world.truth()
            rx, ry = tr["robot"][0], tr["robot"][1]
            people = sorted(tr["people"].values(), key=lambda p: math.hypot(p[0] - rx, p[1] - ry))
            if people:
                truth.append((1.0 if people[0][3] == "careless" else 0.0, math.hypot(people[0][0] - rx,
                                                                                     people[0][1] - ry)))
            else:
                truth.append((-1.0, math.inf))
            return extras.sample(t)

        ep = episode_from_stream(f"{partition}{index}:{e}", adapter.profile(), stream(), STRIDE,
                                 meta={"lifetime": index, "episode": e, "partition": partition}, on_sample=on_sample)
        ep.meta["truth"] = np.asarray(truth)
        episodes.append(ep)
    return episodes


def nerva_trial(job: tuple[str, dict, str]) -> list:
    partition, trial, experience = job
    a = np.load(Path(experience) / f"trial_{trial['trial']:04d}.npz")
    adapter = NervaAdapter(Emitter(f"nerva-{partition}{trial['trial']}"))
    apm = APM(profile=adapter.profile())
    apm.ingest(adapter.profile(), now_s=0.0)
    extras = Extras(apm, np.random.SeedSequence(S.PROGRAMME_ENTROPY, spawn_key=(1, S.PARTITIONS[partition], 77,
                                                                                trial["trial"])))
    command = tuple(trial["command"])

    def stream():
        yield 0.0, [adapter.profile()]
        for i in range(len(a["t"])):
            t = float(a["t"][i])
            msgs = adapter.step(t, a["base_quat"][i], a["base_linvel"][i], a["base_angvel"][i], float(a["tau_sq"][i]),
                                command)
            extras.ingest(msgs, t)
            yield t, msgs

    ep = episode_from_stream(f"nerva-{partition}{trial['trial']}", adapter.profile(), stream(), STRIDE,
                             meta={**trial, "lifetime": trial["trial"], "partition": partition},
                             on_sample=extras.sample)
    ep.meta["truth"] = np.full((len(ep.times), 2), -1.0)
    return [ep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", choices=("wheeled", "nerva"), required=True)
    ap.add_argument("--partition", choices=sorted(S.PARTITIONS), required=True)
    ap.add_argument("--n", type=int, default=0, help="wheeled lifetimes")
    ap.add_argument("--experience", type=Path, help="NERVA trials directory (nerva_generate.py output)")
    ap.add_argument("--out", type=Path, default=Path("runs/E1"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    start = time.time()
    if args.embodiment == "wheeled":
        jobs: list = [(args.partition, i) for i in range(args.n)]
        fn = wheeled_lifetime
        source = {"seeds": [S.fingerprint(S.world_seed("E1", args.partition, i)) for i in range(args.n)]}
    else:
        assert args.experience is not None
        manifest = json.loads((args.experience / "manifest.json").read_text())
        jobs = [(args.partition, t, str(args.experience)) for t in manifest["trials"]]
        fn = nerva_trial
        source = {"experience_manifest_sha256": hashlib.sha256((args.experience / "manifest.json").read_bytes())
                  .hexdigest(), "policy_sha256": manifest["policy_sha256"]}
    with Pool(args.workers) as pool:
        episodes = [e for chunk in pool.map(fn, jobs, chunksize=2) for e in chunk]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.embodiment}_{args.partition}.pkl"
    path.write_bytes(pickle.dumps(episodes))
    counts = {k: int(sum(len(e.events.get(k, [])) for e in episodes)) for k in KINDS}
    info = {"embodiment": args.embodiment, "partition": args.partition, "episodes": len(episodes),
            "samples": int(sum(len(e.times) for e in episodes)), "event_counts": counts,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "wall_s": round(time.time() - start, 1),
            "source": source}
    (args.out / f"{args.embodiment}_{args.partition}.manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "source"}))


if __name__ == "__main__":
    main()
