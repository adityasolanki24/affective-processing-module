"""Resource benchmark of the APM runtime and learned predictor on the machine it runs on.

Measures (on THIS machine; embedded boards are estimated from these numbers in docs/hardware.md, never measured):
- APM step latency and message-ingest throughput in a busy synthetic scene (10 tracked entities, 6 live hypotheses),
- Python heap allocated by a long-running APM with populated memory (tracemalloc), saved-state size,
- GRU (64 hidden) single-sample CPU inference latency, model size, shadow-update training throughput.

    python experiments/E_hardware/benchmark.py --out experiments/E_hardware/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from conftest import Robot  # noqa: E402  (synthetic sender; no robot data)


def busy_scene(apm, robot, t):
    apm.ingest(robot.self_state(t))
    apm.ingest(robot.goal(t))
    for k in range(10):
        apm.ingest(robot.entity(t, track=f"e{k}", identity=f"id{k % 4}", confidence=.8, distance=1 + k * .3,
                                closing=.2 * (k % 3)))
    for k in range(6):
        apm.ingest(robot.hypothesis(t, target=f"e{k}", hid=f"h{k}", p=.1 + .1 * k))
    return 1 + 1 + 10 + 6


def bench_runtime() -> dict:
    robot = Robot("bench")
    apm = robot.apm()
    step_ms, ingest_s, messages = [], 0.0, 0
    for i in range(3000):  # 150 s at 20 Hz
        t = 0.05 * (i + 1)
        t0 = time.perf_counter()
        messages += busy_scene(apm, robot, t)
        if i % 40 == 39:
            apm.ingest(robot.outcome(t, kind="collision", target=f"e{i % 10}", oid=f"o{i}"))
            messages += 1
        ingest_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        apm.step(t)
        step_ms.append(1000 * (time.perf_counter() - t0))
    tracemalloc.start()
    robot2 = Robot("bench2")
    apm2 = robot2.apm()
    for i in range(6000):  # 300 s with many encounters (tracks change every 2 s)
        t = 0.05 * (i + 1)
        apm2.ingest(robot2.self_state(t))
        apm2.ingest(robot2.entity(t, track=f"tr{i // 40}", identity=f"person-{(i // 40) % 50}", confidence=.9))
        if i % 40 == 20:
            apm2.ingest(robot2.outcome(t, kind="benign_contact", target=f"tr{i // 40}", oid=f"b{i}"))
        apm2.step(t)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    with tempfile.TemporaryDirectory() as d:
        apm2.save(Path(d) / "s.json")
        state_bytes = (Path(d) / "s.json").stat().st_size
    return {"step_ms_median": statistics.median(step_ms), "step_ms_p99": float(np.percentile(step_ms, 99)),
            "step_ms_max": max(step_ms), "ingest_messages_per_s": messages / ingest_s,
            "messages_per_tick": messages / 3000,
            "heap_mb_after_300s": current / 1e6, "heap_peak_mb": peak / 1e6, "saved_state_kb": state_bytes / 1e3,
            "entity_records": len(apm2.entities.records), "episodes": len(apm2.episodes.episodes)}


def bench_gru() -> dict:
    try:
        import torch

        from apm.learning.predictor import GRUPredictor
    except ImportError:
        return {"skipped": "torch not installed"}
    torch.set_num_threads(1)  # embedded-like: one core
    model = GRUPredictor(16, hidden=64, seed=0)
    x = np.random.default_rng(0).normal(size=(1, 20, 16)).astype(np.float32)
    for _ in range(50):
        model.predict(x)
    lat = []
    for _ in range(1000):
        t0 = time.perf_counter()
        model.predict(x)
        lat.append(1000 * (time.perf_counter() - t0))
    X = np.random.default_rng(1).normal(size=(4096, 20, 16)).astype(np.float32)
    y = (np.random.default_rng(2).uniform(size=4096) < .1).astype(np.float32)
    t0 = time.perf_counter()
    model.fit(X, y, epochs=2, early_stopping=False)
    train_s = time.perf_counter() - t0
    return {"parameters": model.parameters, "state_kb": model.state_bytes() / 1e3,
            "infer_ms_median_1thread": statistics.median(lat), "infer_ms_p99_1thread": float(np.percentile(lat, 99)),
            "train_samples_per_s_1thread": 2 * 4096 / train_s, "torch": torch.__version__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/E_hardware/results.json"))
    args = ap.parse_args()
    result = {"machine": {"platform": platform.platform(terse=True), "machine": platform.machine(),
                          "logical_cpus": os.cpu_count(), "python": platform.python_version()},
              "note": "development PC, not an embedded board", "runtime": bench_runtime(), "gru": bench_gru()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
