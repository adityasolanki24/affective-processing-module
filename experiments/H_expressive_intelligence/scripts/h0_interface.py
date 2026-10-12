"""H0: measurements of the expressive-action interface on NERVA's validated controllers.

Suites, all in simulation, NERVA policies frozen:
- switching:  random sequences of the seven validated commands, switching every `dwell` seconds. Monitor in
              shadow. Finds the minimum safe dwell.
- pitch:      E1′ only. e_pitch square waves (half-period T) or rate-limited ramps between −1 and +1, under four
              commands. Checks the safety of within-episode pitch changes and the pitch response.
- head:       static head_pitch / head_yaw offsets, ramped in over 1 s, at rest and walking forward, plus head-yaw
              sweeps. Monitor in shadow. Gives the envelope for an untrained head channel.
- fallback:   0.6 / 0.9 m/s pushes in 8 heading-relative directions during an expressive episode, with the monitor
              enforcing vs in shadow (paired).
- nominal:    adapter-driven approach episodes with fixed probe intents. Gives the monitor diagnostic distributions
              (joint margins, force saturation) and adapter behaviour.

    NERVA_ROOT=<nerva> PYTHONPATH=<apm>/src;<nerva> <nerva-venv>/python scripts/h0_interface.py --partition P \
        --suite switching --out results/H0/pilot --workers 3
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

from apm_express.adapter import (  # noqa: E402
    COMMAND_NAMES,
    D_FAR,
    D_NEAR,
    BipedChannels,
    BipedExpressionAdapter,
    MotionRequest,
)
from apm_express.episode import EpisodeSpec, run_episode  # noqa: E402
from apm_express.features import episode_features  # noqa: E402
from apm_express.intent import ExpressionIntent  # noqa: E402
from apm_express.safety import MonitorLimits, SafetyMonitor  # noqa: E402

DT = 0.02
FALL_TILT = 45.0
FORWARD = 0.074
PROBES = {  # fixed probe intents for the nominal suite (not the H1 styles)
    "neutral": ExpressionIntent(),
    "slow_pausing": ExpressionIntent(tempo=0.4, hesitation=0.8, proximity=0.2),
    "indirect_scanning": ExpressionIntent(directness=0.2, scanning=1.0, gaze=1.0, proximity=0.8),
    "averted_lean_back": ExpressionIntent(gaze=-1.0, lean=-1.0, tempo=0.7, head_elevation=-1.0),
    "lean_forward_close": ExpressionIntent(lean=1.0, proximity=1.0, head_elevation=1.0),
}


@dataclass(frozen=True)
class Trial:
    suite: str
    name: str
    policy: str
    index: int  # seed index within the partition
    latency: bool = False
    seconds: float = 30.0
    params: dict = field(default_factory=dict)


def _velocity_frame(a):
    """Heading-frame forward/lateral velocity and yaw rate per step."""
    c, s = np.cos(a["yaw"]), np.sin(a["yaw"])
    vfwd = c * a["vx_world"] + s * a["vy_world"]
    vlat = -s * a["vx_world"] + c * a["vy_world"]
    yaw_rate = np.gradient(np.unwrap(a["yaw"]), DT)
    return vfwd, vlat, yaw_rate


def _smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode="valid") if len(x) >= n else np.array([np.mean(x)])


def _base_metrics(result) -> dict:
    a = result["arrays"]
    tilt = a["tilt"]
    fell = bool((tilt > FALL_TILT).any() or not result["completed"])
    return {"fell": fell, "max_tilt": float(np.nanmax(tilt)), "trips": result["trips"],
            "trip_reasons": [r for _, r in result["trip_reasons"]][:5], "fallback_steps": result["fallback_steps"],
            "min_joint_margin": result["min_joint_margin"], "max_saturated_run": result["max_saturated_run"],
            "completed": result["completed"]}


def _controller(trial: Trial, seed: int, head: bool = False, start_yaw: float = 0.0):
    from apm_express.biped import NervaBiped
    return NervaBiped(policy=trial.policy, seed=seed, latency=trial.latency, head_channel=head, start_yaw=start_yaw)


def run_trial(trial: Trial) -> dict:
    seed = S.sim_seed("H0", trial.params["partition"], trial.index)
    rng = np.random.default_rng(S.world_seed("H0", trial.params["partition"], trial.index).spawn(1)[0])
    p = trial.params
    t0 = time.time()
    out = {"trial": asdict(trial), "sim_seed": seed}
    if trial.suite == "switching":
        ctrl = _controller(trial, seed)
        dwell = p["dwell"]
        override, t, cmd = [], 0.0, "rest"
        while t < trial.seconds:
            cmd = rng.choice([c for c in COMMAND_NAMES if c != cmd])
            override.append((t, t + dwell, MotionRequest(str(cmd), 0.0, (0.0, 0.0, 0.0, 0.0))))
            t = round(t + dwell, 10)
        spec = EpisodeSpec(trial.seconds, None, [(0.0, ExpressionIntent())], override=override)
        r = run_episode(ctrl, BipedExpressionAdapter(BipedChannels(min_dwell_s=0.0)), SafetyMonitor(), spec,
                        enforce=False)
        out.update(_base_metrics(r))
    elif trial.suite == "pitch":
        ctrl = _controller(trial, seed)
        cmd, mode = p["command"], p["mode"]
        n = int(round(trial.seconds / DT))
        ts = np.arange(n) * DT
        if mode.startswith("square"):
            half = float(mode.split("_")[1])
            e = np.where(((ts // half) % 2) == 0, -1.0, 1.0)
            e[ts < 2.0] = 0.0
        else:  # ramp at `rate` per second between -1 and +1 (triangle wave)
            rate = float(mode.split("_")[1])
            period = 4.0 / rate
            e = 1.0 - 4.0 * np.abs(((ts - 2.0) / period + 0.25) % 1.0 - 0.5)
            e[ts < 2.0] = 0.0
            e = np.clip(e, -1, 1)
        override = [(k * DT, (k + 1) * DT, MotionRequest(cmd, float(e[k]), (0.0, 0.0, 0.0, 0.0))) for k in range(n)]
        spec = EpisodeSpec(trial.seconds, None, [(0.0, ExpressionIntent())], override=override)
        r = run_episode(ctrl, BipedExpressionAdapter(), SafetyMonitor(), spec, enforce=False)
        out.update(_base_metrics(r))
        a = r["arrays"]
        mask = ts[: len(a["pitch"])] >= 5.0
        ee, pp = e[: len(a["pitch"])][mask], a["pitch"][mask]
        hi, lo = pp[ee > 0.5], pp[ee < -0.5]
        out["pitch_hi_minus_lo_rad"] = float(hi.mean() - lo.mean()) if len(hi) and len(lo) else None
        lags = range(0, 76, 5)
        corr = [float(np.corrcoef(ee[: len(ee) - L], pp[L:])[0, 1]) if np.std(ee) > 0 else 0.0 for L in lags]
        out["pitch_lag_s"] = float(list(lags)[int(np.nanargmax(corr))] * DT)
        out["pitch_corr_at_lag"] = float(np.nanmax(corr))
        vfwd, vlat, wz = _velocity_frame(a)
        out["mean_vfwd"], out["mean_vlat"], out["mean_yaw_rate"] = (float(vfwd[mask].mean()), float(vlat[mask].mean()),
                                                                   float(wz[mask].mean()))
    elif trial.suite == "head":
        ctrl = _controller(trial, seed, head=True)
        cmd = p["command"]
        n = int(round(trial.seconds / DT))
        ts = np.arange(n) * DT
        hp, hy, sweep = p.get("head_pitch", 0.0), p.get("head_yaw", 0.0), p.get("sweep", 0.0)
        ramp = np.clip((ts - 1.0) / 1.0, 0.0, 1.0)
        yaw = hy * ramp + sweep * ramp * np.sin(2 * math.pi * 0.3 * ts)
        override = [(k * DT, (k + 1) * DT, MotionRequest(cmd, 0.0, (0.0, float(hp * ramp[k]), float(yaw[k]), 0.0)))
                    for k in range(n)]
        spec = EpisodeSpec(trial.seconds, None, [(0.0, ExpressionIntent())], override=override)
        r = run_episode(ctrl, BipedExpressionAdapter(), SafetyMonitor(), spec, enforce=False)
        out.update(_base_metrics(r))
        a = r["arrays"]
        mask = (a["t"] >= 5.0) & (a["t"] < 20.0 + 1e-9)
        vfwd, vlat, wz = _velocity_frame(a)
        sf, sl, sw = _smooth(vfwd[mask]), _smooth(vlat[mask]), _smooth(wz[mask])
        disp = float(np.hypot(a["x"][mask][-1] - a["x"][mask][0], a["y"][mask][-1] - a["y"][mask][0]))
        out.update(mean_vfwd=float(vfwd[mask].mean()), lateral_rms=float(np.sqrt(np.mean(sl ** 2))),
                   yaw_rms=float(np.sqrt(np.mean(sw ** 2))), horizontal_rms=float(np.sqrt(np.mean(sf ** 2 + sl ** 2))),
                   displacement=disp, stationary_fraction=float(np.mean(np.abs(sf) < 0.01)),
                   face_yaw_mean=float(np.nanmean(a["face_yaw"][mask])),
                   face_elev_mean=float(np.nanmean(a["face_elev"][mask])),
                   head_yaw_joint_mean=float(a["head_joints"][mask, 2].mean()),
                   head_pitch_joint_mean=float(a["head_joints"][mask, 1].mean()),
                   mean_pitch=float(a["pitch"][mask].mean()))
    elif trial.suite in ("fallback", "fallback_b"):
        ch, limits = interface(p["channels"])
        ctrl = _controller(trial, seed, head=ch.head)
        push_k = int(round(p["push_t"] / DT))
        ang = math.radians(p["push_deg"])
        intent = PROBES[p["probe"]]
        # world-frame push; the robot starts facing +x and walks toward a target on the +x axis, so this is
        # approximately heading-relative
        spec = EpisodeSpec(trial.seconds, (1.6, 0.0), [(0.0, intent)],
                           pushes={push_k: (p["push"] * math.cos(ang), p["push"] * math.sin(ang))})
        ad = BipedExpressionAdapter(ch, seed=seed)
        r = run_episode(ctrl, ad, SafetyMonitor(limits), spec, enforce=p["enforce"])
        out.update(_base_metrics(r))
        a = r["arrays"]
        out["fallback_s"] = r["fallback_steps"] * DT
        fb = a["fallback"].astype(bool)
        out["nonneutral_during_fallback"] = int(np.count_nonzero(fb & ((a["command"] != COMMAND_NAMES.index("rest"))
                                                                 | (a["e_pitch"] != 0.0)
                                                                 | np.any(a["head_request"] != 0.0, axis=1))))
        out["final_distance"] = float(math.hypot(1.6 - a["x"][-1], a["y"][-1]))
    elif trial.suite == "nominal":
        ch, limits = interface(p["channels"])
        ctrl = _controller(trial, seed, head=ch.head, start_yaw=p["start_yaw"])
        intent = PROBES[p["probe"]]
        target = (p["distance"] * math.cos(p["bearing"]), p["distance"] * math.sin(p["bearing"]))
        ad = BipedExpressionAdapter(ch, seed=seed)
        r = run_episode(ctrl, ad, SafetyMonitor(limits), EpisodeSpec(trial.seconds, target, [(0.0, intent)]),
                        enforce=p.get("enforce", True))
        out.update(_base_metrics(r))
        a = r["arrays"]
        out["final_distance"] = float(math.hypot(target[0] - a["x"][-1], target[1] - a["y"][-1]))
        out["command_counts"] = np.bincount(a["command"], minlength=len(COMMAND_NAMES)).tolist()
        out["d_stop"] = D_FAR - (D_FAR - D_NEAR) * intent.proximity
        out["arrived"] = bool(abs(out["final_distance"] - out["d_stop"]) <= 0.15)
        moving = a["command"] != COMMAND_NAMES.index("rest")
        out["features"] = episode_features(a, target)
        out["head_while_moving_max"] = float(np.abs(a["head_request"][moving]).max()) if moving.any() else 0.0
    else:
        raise ValueError(trial.suite)
    out["wall_s"] = round(time.time() - t0, 2)
    return out


def interface(channels: dict) -> tuple[BipedChannels, MonitorLimits]:
    """The adapter channels and monitor limits from an admitted-interface record (h0_analyze.py)."""
    ch = BipedChannels(min_dwell_s=channels["min_dwell_s"], pitch=channels["pitch"],
                       pitch_rate=channels.get("pitch_rate", 1.0), head=channels["head"],
                       head_pitch_range=tuple(channels.get("head_pitch_range", (0.0, 0.0))),
                       head_yaw_range=tuple(channels.get("head_yaw_range", (0.0, 0.0))),
                       head_rate=channels.get("head_rate", 0.8), head_while_moving=False)
    limits = MonitorLimits(joint_margin_rad=channels.get("joint_margin_rad"),
                           saturation_steps=channels.get("saturation_steps"))
    return ch, limits


PILOT_CHANNELS = {"policy": "e1_prime", "min_dwell_s": 0.6, "pitch": True, "head": False}


def trials_for(suite: str, partition: str, n_seeds: int, channels: dict | None = None,
               secondary: dict | None = None) -> list[Trial]:
    """Pilot (P) grids as run on 2026-10-12; confirmatory (X) grids as fixed in prereg/H0_ADDENDUM.md."""
    x = partition == "X"
    given = channels is not None
    channels = channels or PILOT_CHANNELS
    out: list[Trial] = []
    base = {"partition": partition}
    idx = 0

    def add(**kw):
        nonlocal idx
        params = dict(base, **kw.pop("params", {}))
        index = kw.pop("index", idx)
        out.append(Trial(suite=suite, index=index, params=params, **kw))
        idx += 1

    if suite == "switching":
        for policy in ("neutral", "e1_prime"):
            for dwell in ((0.4, 0.6, 1.0) if x else (0.2, 0.4, 0.6, 1.0)):
                for latency in (False, True):
                    for s in range(n_seeds):
                        add(name=f"{policy}_d{dwell}_{'lat' if latency else 'nolat'}_{s}", policy=policy,
                            latency=latency, seconds=30.0, params={"dwell": dwell})
    elif suite == "pitch":
        modes = ("square_1", "ramp_1") if x else ("square_0.5", "square_1", "square_2", "square_4", "ramp_1")
        for cmd in ("forward", "turn_left", "left", "rest"):
            for mode in modes:
                for latency in (False, True):
                    for s in range(n_seeds):
                        add(name=f"{cmd}_{mode}_{'lat' if latency else 'nolat'}_{s}", policy="e1_prime",
                            latency=latency, seconds=30.0, params={"command": cmd, "mode": mode})
    elif suite == "head":
        points = [(0.0, 0.0)] + [(hp, 0.0) for hp in (-0.4, -0.2, 0.2, 0.4)] + \
                 [(0.0, hy) for hy in (-0.8, -0.4, 0.4, 0.8)] + [(hp, hy) for hp in (-0.2, 0.2) for hy in (-0.4, 0.4)]
        sweeps = [(0.0, a) for a in (0.2, 0.4)]
        if x:  # every corner of every candidate box, and yaw sweeps at every candidate pitch (H0 addendum)
            points = [(0.0, 0.0)] + [(hp, 0.0) for hp in (-0.4, -0.2, 0.2, 0.4)] + \
                     [(0.0, hy) for hy in (-0.8, -0.4, 0.4, 0.8)] + \
                     [(hp, hy) for hp in (-0.4, -0.2, 0.2, 0.4) for hy in (-0.8, -0.4, 0.4, 0.8)]
            sweeps = [(hp, a) for a in (0.4, 0.8) for hp in (-0.4, -0.2, 0.0, 0.2, 0.4)]
        walk_points = [(0.0, 0.0)] + [(hp, 0.0) for hp in (-0.2, 0.2)] + [(0.0, hy) for hy in (-0.4, 0.4)]
        for policy in ("neutral", "e1_prime"):
            for cmd in ("rest", "forward"):
                for hp, hy in (walk_points if x and cmd == "forward" else points):
                    for s in range(n_seeds):
                        add(name=f"{policy}_{cmd}_p{hp}_y{hy}_{s}", policy=policy, seconds=20.0,
                            params={"command": cmd, "head_pitch": hp, "head_yaw": hy})
                for hp, amp in (sweeps if not (x and cmd == "forward") else ()):
                    for s in range(n_seeds):
                        add(name=f"{policy}_{cmd}_p{hp}_sweep{amp}_{s}", policy=policy, seconds=20.0,
                            params={"command": cmd, "head_pitch": hp, "sweep": amp})
    elif suite == "fallback":
        for push in ((0.3, 0.6, 0.9) if x else (0.6, 0.9)):
            for deg in range(0, 360, 45):
                for enforce in (True, False):
                    for s in range(n_seeds):
                        # enforce and shadow arms share the seed index (paired)
                        add(name=f"push{push}_{deg}_{'enf' if enforce else 'shadow'}_{s}",
                            policy=channels["policy"], index=int(round(push * 10)) * 1000 + deg + s * 100_000,
                            seconds=20.0, params={"push": push, "push_deg": deg, "push_t": 6.0, "enforce": enforce,
                                                  "probe": "slow_pausing", "channels": channels})
    elif suite == "fallback_b":
        m_star = channels["push_star"]
        cells = [(m_star, "slow_pausing", True), (0.3, "slow_pausing", True), (0.3, "slow_pausing", False),
                 (0.3, "neutral", True)]
        for push, probe, enforce in cells:
            for deg in range(0, 360, 45):
                for s in range(n_seeds):
                    add(name=f"push{push}_{probe}_{deg}_{'enf' if enforce else 'shadow'}_{s}",
                        policy=channels["policy"], index=int(round(push * 100)) * 1000 + deg + s * 100_000,
                        seconds=20.0, params={"push": push, "push_deg": deg, "push_t": 6.0, "enforce": enforce,
                                              "probe": probe, "channels": channels})
    elif suite == "nominal":
        if given:  # the admitted primary interface, and the admitted secondary (E1') one if any
            configs = [channels] + ([secondary] if secondary else [])
        else:
            configs = [{**PILOT_CHANNELS, "policy": pol, "pitch": pol == "e1_prime"} for pol in ("neutral", "e1_prime")]
        for probe in PROBES:
            for ch in configs:
                policy = ch["policy"]
                for s in range(n_seeds):
                    add(name=f"{probe}_{policy}_{s}", policy=policy, seconds=60.0 if x else 45.0,
                        params={"probe": probe, "distance": 1.0 + 0.2 * (s % 3), "channels": ch,
                                "bearing": [0.0, 0.6, -0.6, 1.4][s % 4], "start_yaw": 0.0, "enforce": x})
    else:
        raise ValueError(suite)
    return out


def _init():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import nerva.sim.open_duck  # noqa: F401  warm import once per worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["P", "X"])
    ap.add_argument("--suite", required=True,
                    choices=["switching", "pitch", "head", "fallback", "nominal", "fallback_b"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--channels", type=Path, help="admitted-interface JSON (fallback and nominal suites on X)")
    ap.add_argument("--index-offset", type=int, default=0, help="fresh seed indices (H0b: 1,000,000)")
    ap.add_argument("--push-star", type=float, help="standing-validated push magnitude (fallback_b)")
    ap.add_argument("--addendum", default="H0_ADDENDUM.md")
    args = ap.parse_args()
    from apm_express.provenance import confirmatory_lock, environment, nerva_state
    if args.partition == "X":
        lock = confirmatory_lock(HERE / "prereg" / args.addendum)
    else:
        lock = {**environment(), **nerva_state(), "partition": "P", "exploratory": True}
    admitted = json.loads(args.channels.read_text(encoding="utf-8")) if args.channels else {}
    channels, secondary = admitted.get("interface"), admitted.get("secondary_interface")
    if channels is not None and args.push_star is not None:
        channels = {**channels, "push_star": args.push_star}
    if args.partition == "X" and args.suite in ("fallback", "nominal", "fallback_b") and channels is None:
        raise SystemExit("X fallback/nominal suites run on the admitted interface: pass --channels")
    trials = trials_for(args.suite, args.partition, args.seeds, channels, secondary)
    if args.index_offset:
        trials = [Trial(**{**asdict(t), "index": t.index + args.index_offset}) for t in trials]
    start = time.time()
    with Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(run_trial, trials, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.suite}.json"
    payload = {"suite": args.suite, "partition": args.partition, "lock": lock, "n_trials": len(rows),
               "wall_s": round(time.time() - start, 1), "rows": rows}
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"suite": args.suite, "trials": len(rows), "wall_s": payload["wall_s"],
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
