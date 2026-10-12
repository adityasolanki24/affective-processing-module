"""H1: NERVA's expressive movement repertoire through the admitted H0 interface (prereg/H1_ADDENDUM.md).

Arms:
- X0: neutral.
- X1: six designed styles.
- X1m: the same styles speed-matched (tempo calibrated on V).
- X2: random intents.
- push: X0/X1 with a 0.30 m/s push.
- switch: within-episode style transitions.

Every trial approaches a target in one of 8 geometries and records motion features (`apm_express.features`),
stability and arrival.

    NERVA_ROOT=... PYTHONPATH="src;<NERVA>" <nerva-venv>/python scripts/h1_repertoire.py --partition P \
        --arm core --seeds 2 --interface results/H0/X/admitted_interface.json --out results/H1/pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def _load_h_seeds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("h_seeds", mod)
    spec.loader.exec_module(mod)
    return sys.modules["h_seeds"] if sys.modules["h_seeds"] is not mod else mod


S = _load_h_seeds()
sys.path.insert(0, str(HERE / "scripts"))

from h0_interface import interface  # noqa: E402

from apm_express.adapter import COMMAND_NAMES, D_FAR, D_NEAR, BipedExpressionAdapter  # noqa: E402
from apm_express.episode import EpisodeSpec, run_episode  # noqa: E402
from apm_express.features import episode_features, stability  # noqa: E402
from apm_express.intent import ExpressionIntent  # noqa: E402
from apm_express.safety import SafetyMonitor  # noqa: E402
from apm_express.styles import DESIGNED, NEUTRAL, random_intent, with_tempo  # noqa: E402

GEOMETRIES = [(d, b) for d in (0.9, 1.3) for b in (0.0, 0.7, -0.7, 1.6)]  # (distance m, bearing rad)
SHIFT_GEOMETRIES = [(1.8, 0.0), (1.8, 2.4), (0.7, -2.4), (1.3, 3.1)]  # partition S: farther / behind the robot
SECONDS = 60.0
SWITCH_PAIRS = [("cautious", "assertive"), ("assertive", "cautious"), ("hesitant", "energetic"),
                ("energetic", "hesitant"), ("relaxed", "curious"), ("curious", "relaxed")]
TEMPO_GRID = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)


@dataclass(frozen=True)
class Trial:
    arm: str
    style: str
    geometry: int
    index: int
    partition: str
    latency: bool
    intent: dict = field(default_factory=dict)
    push: tuple | None = None  # (time s, speed m/s, direction rad)
    switch_to: dict | None = None
    switch_t: float = 6.0
    config: str = "primary"


def run_trial(job: tuple[Trial, dict]) -> dict:
    trial, iface = job
    from apm_express.biped import NervaBiped

    geoms = SHIFT_GEOMETRIES if trial.partition == "S" else GEOMETRIES
    dist, bearing = geoms[trial.geometry]
    target = (dist * math.cos(bearing), dist * math.sin(bearing))
    seed = S.sim_seed("H1", trial.partition, trial.index)
    ch, limits = interface(iface)
    ctrl = NervaBiped(policy=iface["policy"], seed=seed, latency=trial.latency, head_channel=ch.head,
                      max_head_rate=1.5, max_pitch_rate=2.0)
    intent = ExpressionIntent(**trial.intent)
    intents = [(0.0, intent)]
    if trial.switch_to is not None:
        intents.append((trial.switch_t, ExpressionIntent(**trial.switch_to)))
    pushes = {}
    if trial.push is not None:
        t_push, speed, direction = trial.push
        pushes[int(round(t_push / ctrl.dt))] = (speed * math.cos(direction), speed * math.sin(direction))
    t0 = time.time()
    r = run_episode(ctrl, BipedExpressionAdapter(ch, seed=seed), SafetyMonitor(limits),
                    EpisodeSpec(SECONDS, target, intents, pushes=pushes))
    a = r["arrays"]
    final_intent = intents[-1][1]
    d_stop = D_FAR - (D_FAR - D_NEAR) * final_intent.proximity
    final_d = float(math.hypot(target[0] - a["x"][-1], target[1] - a["y"][-1]))
    moving = a["command"] != COMMAND_NAMES.index("rest")
    out = {"trial": asdict(trial), "sim_seed": seed, "target": target, "d_stop": d_stop, "final_distance": final_d,
           "arrived": bool(abs(final_d - d_stop) <= 0.15), **stability(a, r["completed"]), "trips": r["trips"],
           "trip_reasons": [x for _, x in r["trip_reasons"]][:3], "features": episode_features(a, target),
           "head_while_moving_max": float(np.abs(a["head_request"][moving]).max()) if moving.any() else 0.0,
           "wall_s": round(time.time() - t0, 2)}
    return out


def build(arm: str, partition: str, n_seeds: int, tempos: dict | None = None) -> list[Trial]:
    styles = {"neutral": NEUTRAL, **DESIGNED}
    geoms = SHIFT_GEOMETRIES if partition == "S" else GEOMETRIES
    out: list[Trial] = []
    base = {"core": 0, "matched": 100_000, "random": 200_000, "push": 300_000, "switch": 400_000,
            "calibrate": 500_000}[arm]
    k = 0

    def add(**kw):
        nonlocal k
        out.append(Trial(arm=arm, partition=partition, index=base + k, latency=bool(k % 2), **kw))
        k += 1

    if arm == "core":
        for name, it in styles.items():
            for g in range(len(geoms)):
                for _ in range(n_seeds):
                    add(style=name, geometry=g, intent=it.as_dict())
    elif arm == "matched":
        if tempos is None:
            raise SystemExit("the matched arm needs calibrated tempos (--tempos)")
        for name, it in DESIGNED.items():
            for g in range(len(geoms)):
                for _ in range(n_seeds):
                    add(style=name, geometry=g, intent=with_tempo(it, tempos[name]).as_dict())
    elif arm == "random":
        rng = np.random.default_rng(S.world_seed("H1", partition, 999_999))
        for i in range(n_seeds * 20):
            add(style=f"random_{i}", geometry=i % len(geoms), intent=random_intent(rng).as_dict())
    elif arm == "push":
        rng = np.random.default_rng(S.world_seed("H1", partition, 999_998))
        for name, it in styles.items():
            for g in range(len(geoms)):
                for _ in range(n_seeds):
                    add(style=name, geometry=g, intent=it.as_dict(),
                        push=(float(rng.uniform(4.0, 10.0)), 0.30, float(rng.uniform(-math.pi, math.pi))))
    elif arm == "switch":
        for a_name, b_name in SWITCH_PAIRS:
            for g in range(0, len(geoms), 2):
                for _ in range(n_seeds):
                    add(style=f"{a_name}->{b_name}", geometry=g, intent=DESIGNED[a_name].as_dict(),
                        switch_to=DESIGNED[b_name].as_dict())
    elif arm == "calibrate":
        for name, it in DESIGNED.items():
            for tempo in TEMPO_GRID:
                for g in (0, 5):
                    for _ in range(n_seeds):
                        add(style=f"{name}@{tempo}", geometry=g, intent=with_tempo(it, tempo).as_dict())
    else:
        raise ValueError(arm)
    return out


def _init():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import nerva.sim.open_duck  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["P", "V", "X", "S"])
    ap.add_argument("--arm", required=True, choices=["core", "matched", "random", "push", "switch", "calibrate"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--interface", type=Path, required=True)
    ap.add_argument("--tempos", type=Path)
    ap.add_argument("--config", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock, environment, nerva_state
    if a.partition in ("X", "S"):
        lock = confirmatory_lock(HERE / "prereg" / "H1_ADDENDUM.md")
    else:
        lock = {**environment(), **nerva_state(), "partition": a.partition, "exploratory": a.partition == "P"}
    adm = json.loads(a.interface.read_text(encoding="utf-8"))
    iface = adm["interface"] if a.config == "primary" else adm["secondary_interface"]
    if iface is None:
        raise SystemExit(f"no admitted {a.config} interface")
    tempos = json.loads(a.tempos.read_text(encoding="utf-8"))["tempos"] if a.tempos else None
    trials = build(a.arm, a.partition, a.seeds, tempos)
    start = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        rows = pool.map(run_trial, [(t, iface) for t in trials], chunksize=1)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.arm}_{a.config}.json"
    payload = {"arm": a.arm, "partition": a.partition, "config": a.config, "interface": iface, "lock": lock,
               "n_trials": len(rows), "wall_s": round(time.time() - start, 1), "rows": rows}
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"arm": a.arm, "trials": len(rows), "wall_s": payload["wall_s"],
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
