"""Generate NERVA simulator experience for APM learning (Milestone C): randomized pushes on the validated policy.

Each trial: NERVA's validated neutral candidate, 15 s, a random one of NERVA's seven neutral commands, action
latency on or off, and with probability 0.85 one horizontal push at a random time in [5, 10] s with speed
U(0.2, 1.4) m/s in a random world direction (NERVA's own training pushes span 0.1–1.0 m/s). Telemetry is saved
exactly as NERVA's simulator logs it; labels are derived later by the APM NERVA adapter's fixed detectors.
The push parameters are stored for analysis only and are never model inputs.

    PYTHONPATH=<APM>/src python examples/nerva/generate_experience.py --nerva <NERVA> --policy <candidate.onnx> \
        --trials 600 --out runs/nerva_experience
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

COMMANDS = [(0.0, 0.0, 0.0), (0.074, 0.0, 0.0), (-0.074, 0.0, 0.0), (0.0, 0.074, 0.0), (0.0, -0.074, 0.0),
            (0.0, 0.0, 0.6), (0.0, 0.0, -0.6)]
DT, SECONDS = 0.02, 15.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerva", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("runs/nerva_experience"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.nerva.resolve()))
    from nerva.analysis.motor_eval import deployment_metadata
    from nerva.interfaces import BehaviourCommand
    from nerva.sim.open_duck import SCENE_BACKLASH, OpenDuckSim, to_arrays

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    digest = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    manifest = {"policy_sha256": digest, "seed": args.seed, "dt_s": DT, "seconds": SECONDS, "trials": []}
    start = time.monotonic()
    for i in range(args.trials):
        command = COMMANDS[int(rng.integers(7))]
        latency = bool(rng.random() < 0.5)
        push = None
        if rng.random() < 0.85:
            push = {"t_s": float(rng.uniform(5, 10)), "speed_m_s": float(rng.uniform(0.2, 1.4)),
                    "direction_rad": float(rng.uniform(0, 2 * np.pi))}
        with contextlib.redirect_stdout(io.StringIO()):
            sim = OpenDuckSim(policy_path=args.policy, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                              init_joint_noise=0.02, seed=i, action_delay=latency)
        sim.set_neutral_motor_contract(deployment_metadata(digest))
        sim.set_behaviour(BehaviourCommand(*command))
        push_step = int(round(push["t_s"] / DT)) if push else -1
        log = []
        for k in range(int(SECONDS / DT)):
            if k == push_step:
                sim.push(push["speed_m_s"] * np.cos(push["direction_rad"]),
                         push["speed_m_s"] * np.sin(push["direction_rad"]))
            log.extend(sim.run(DT))
            if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
                break
        a = to_arrays(log)
        np.savez_compressed(args.out / f"trial_{i:04d}.npz", t=(np.arange(len(log)) + 1) * DT,
                            base_quat=a["base_quat"], base_linvel=a["base_linvel"], base_angvel=a["base_angvel"],
                            tau_sq=a["tau_sq"])
        manifest["trials"].append({"trial": i, "command": command, "latency": latency, "push": push,
                                   "steps": len(log)})
        if i % 50 == 49:
            (args.out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            print(f"{i + 1}/{args.trials} trials, {time.monotonic() - start:.0f} s", flush=True)
    manifest["wall_seconds"] = round(time.monotonic() - start, 1)
    (args.out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
