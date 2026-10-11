"""E0: integrity gate tests, throughput benchmarks and estimator calibration (pilot partition only).

    python experiments/F_behavioral_learning/scripts/e0_integrity.py
"""
from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm_research.dataset import DecisionRecorder  # noqa: E402
from apm_research.features import WIDTH  # noqa: E402
from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.policies import B0, B1, B2, Explore, Policy  # noqa: E402
from apm_research.predictor import ActionPredictor  # noqa: E402
from apm_research.provenance import ROOT, environment, source_hashes, write_json  # noqa: E402
from apm_testbeds.wheeled_ctrl import GO, SLOW, YIELD, WheeledCtrlWorld  # noqa: E402


class Fixed(Policy):
    def __init__(self, action):
        self.action = action
        self.name = f"always_{action}"

    def decide(self, t):
        return self.action


def main() -> None:
    out = {"environment": environment(), "sources": source_hashes()}
    start = time.time()
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_wheeled_ctrl.py",
                            "tests/test_v02_defects.py"], cwd=ROOT, capture_output=True, text=True)
    out["gate_tests"] = {"returncode": tests.returncode, "summary": tests.stdout.strip().splitlines()[-1]}
    print(out["gate_tests"], flush=True)

    # ---- throughput
    bench = {}
    pilot = S.seeds("E0", "P", 200)
    t0 = time.perf_counter()
    steps = 0
    for ss in pilot[:20]:
        w = WheeledCtrlWorld(ss, Lifetime(ss).population())
        while not w.done:
            w.step(GO)
            steps += 1
    bench["world_steps_per_s"] = steps / (time.perf_counter() - t0)
    rec = DecisionRecorder()
    t0 = time.perf_counter()
    for i, ss in enumerate(pilot[20:60]):
        run_lifetime(Lifetime(ss, n_episodes=5), Explore(i), recorder=rec)
    bench["explore_lifetime5_s"] = (time.perf_counter() - t0) / 40
    rec_v = DecisionRecorder()
    for i, ss in enumerate(pilot[60:75]):
        run_lifetime(Lifetime(ss, n_episodes=5), Explore(1000 + i), recorder=rec_v)
    X, A, c, s, p = rec.arrays()
    Xv, Av, cv, sv, pv = rec_v.arrays()
    t0 = time.perf_counter()
    model = ActionPredictor(WIDTH).fit(X, A, c, s, p, Xv, Av, cv, sv, pv, epochs=10, patience=100)
    bench["predictor_train_samples_per_s"] = 10 * len(X) / (time.perf_counter() - t0)
    bench["predictor_parameters"] = model.parameters
    bench["pilot_decisions"] = int(len(X))
    for name, policy in (("B0", B0()), ("B1", B1(model, 1.0, 0.5)), ("B2", B2(model, 1.0, 0.5, 1.0, 0.25))):
        t0 = time.perf_counter()
        for ss in pilot[100:104]:
            run_lifetime(Lifetime(ss, n_episodes=5), policy)
        bench[f"{name}_lifetime10_s"] = 2 * (time.perf_counter() - t0) / 4
    bench["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out["benchmarks"] = bench
    print(bench, flush=True)

    # ---- TTC estimator calibration and testbed behaviour (pilot)
    ps, ys, alarms = [], [], 0
    stats = {}
    for name, policy in (("always_GO", Fixed(GO)), ("always_SLOW", Fixed(SLOW)), ("always_YIELD", Fixed(YIELD))):
        rows = [run_lifetime(Lifetime(ss, n_episodes=2), policy) for ss in pilot[120:160]]
        stats[name] = {k: float(np.mean([r[k] for r in rows])) for k in
                       ("completion", "collision_rate", "idle_fraction", "J")}
        stats[name]["careless_share_of_person_collisions"] = float(
            sum(r["careless_collisions"] for r in rows) / max(1, sum(r["person_collisions"] for r in rows)))
    for ss in pilot[160:200]:
        w = WheeledCtrlWorld(ss, Lifetime(ss).population())
        per_step = []
        while not w.done:
            obs = w.step(GO)
            per_step.append((obs.t, {e.track: e.ttc_p for e in obs.entities if e.ttc_p > 0}))
        cols = [(c[0], c[1]) for c in w.log.collisions]
        for t, live in per_step:
            for track, p in live.items():
                ps.append(p)
                ys.append(float(any(track == ct and t < tc <= t + 1.0 for tc, ct in cols)))
            alarms += any(p >= 0.5 for p in live.values())
    ps_a, ys_a = np.array(ps), np.array(ys)
    hi = ps_a >= 0.5
    out["ttc_estimator"] = {"pairs": int(len(ps_a)), "steps": 40 * 1200, "alarm_step_fraction": alarms / (40 * 1200),
                            "precision_at_0.5_same_target_1s": float(ys_a[hi].mean()) if hi.any() else None,
                            "mean_p": float(ps_a.mean()), "event_rate": float(ys_a.mean()),
                            "brier": float(np.mean((ps_a - ys_a) ** 2))}
    out["fixed_action_behaviour"] = stats
    out["wall_s"] = time.time() - start
    sha = write_json(HERE / "results" / "E0" / "e0_results.json", out)
    print(out["ttc_estimator"], stats, sha, flush=True)


if __name__ == "__main__":
    main()
