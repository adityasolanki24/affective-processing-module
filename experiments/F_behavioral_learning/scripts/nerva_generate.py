"""Fresh NERVA simulator experience for v0.2 (observe-only; NERVA's policy, safety logic and checkpoints untouched).

Same trial protocol as Milestone C (`examples/nerva/generate_experience.py`): validated neutral candidate, 15 s, one
of the seven neutral commands, action latency on/off, push with prob. 0.85 at U(5, 10) s. v0.2 seeding (AUDIT.md):
trial parameters from SeedSequence(2027, spawn_key=(partition,)), simulator seed 100,000 + 10,000·partition + i, so no
simulator noise stream of the C data (seeds 0–599) is reused. Partition S (shift) uses pushes of U(1.0, 1.8) m/s
instead of U(0.2, 1.4).

    python scripts/nerva_generate.py --partition T --trials 480 --out runs/E1/nerva/T --workers 4
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seeds import NERVA_V02_MASTER_SEED, NERVA_V02_SIM_SEED_OFFSET, PARTITIONS  # noqa: E402

COMMANDS = [(0.0, 0.0, 0.0), (0.074, 0.0, 0.0), (-0.074, 0.0, 0.0), (0.0, 0.074, 0.0), (0.0, -0.074, 0.0),
            (0.0, 0.0, 0.6), (0.0, 0.0, -0.6)]
DT, SECONDS = 0.02, 15.0
_ARGS: dict = {}


def trial_plan(partition: str, n: int) -> list[dict]:
    rng = np.random.default_rng(np.random.SeedSequence(NERVA_V02_MASTER_SEED, spawn_key=(PARTITIONS[partition],)))
    speed = (1.0, 1.8) if partition == "S" else (0.2, 1.4)
    plan = []
    for i in range(n):
        command = COMMANDS[int(rng.integers(7))]
        latency = bool(rng.random() < 0.5)
        push = None
        if rng.random() < 0.85:
            push = {"t_s": float(rng.uniform(5, 10)), "speed_m_s": float(rng.uniform(*speed)),
                    "direction_rad": float(rng.uniform(0, 2 * np.pi))}
        plan.append({"trial": i, "command": command, "latency": latency, "push": push,
                     "sim_seed": NERVA_V02_SIM_SEED_OFFSET + 10_000 * PARTITIONS[partition] + i})
    return plan


def _init(nerva: str, policy: str, out: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, nerva)
    _ARGS.update(nerva=nerva, policy=Path(policy), out=Path(out))


def run_trial(trial: dict) -> dict:
    from nerva.analysis.motor_eval import deployment_metadata
    from nerva.interfaces import BehaviourCommand
    from nerva.sim.open_duck import SCENE_BACKLASH, OpenDuckSim, to_arrays
    digest = hashlib.sha256(_ARGS["policy"].read_bytes()).hexdigest()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = OpenDuckSim(policy_path=_ARGS["policy"], scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                          init_joint_noise=0.02, seed=trial["sim_seed"], action_delay=trial["latency"])
    sim.set_neutral_motor_contract(deployment_metadata(digest))
    sim.set_behaviour(BehaviourCommand(*trial["command"]))
    push = trial["push"]
    push_step = int(round(push["t_s"] / DT)) if push else -1
    log = []
    for k in range(int(SECONDS / DT)):
        if k == push_step:
            speed, direction = push["speed_m_s"], push["direction_rad"]
            sim.push(speed * np.cos(direction), speed * np.sin(direction))
        log.extend(sim.run(DT))
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            break
    a = to_arrays(log)
    path = _ARGS["out"] / f"trial_{trial['trial']:04d}.npz"
    np.savez_compressed(path, t=(np.arange(len(log)) + 1) * DT, base_quat=a["base_quat"],
                        base_linvel=a["base_linvel"], base_angvel=a["base_angvel"], tau_sq=a["tau_sq"])
    return {**trial, "steps": len(log), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerva", default="/home/claude/NERVA")
    ap.add_argument("--policy", default="runs/nerva_policy/candidate.onnx")
    ap.add_argument("--partition", required=True, choices=sorted(PARTITIONS))
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.nerva, args.policy, str(args.out))) as pool:
        trials = pool.map(run_trial, trial_plan(args.partition, args.trials), chunksize=4)
    digest = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    manifest = {"partition": args.partition, "policy_sha256": digest,
                "master_seed": NERVA_V02_MASTER_SEED, "dt_s": DT, "seconds": SECONDS, "trials": trials,
                "wall_seconds": round(time.time() - start, 1)}
    (args.out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(args.partition, len(trials), manifest["wall_seconds"], "s")


if __name__ == "__main__":
    main()
