"""G2 runner: belief memories in closed loop on the scenario families (apm_select.scenarios), 30-episode lifetimes.

Spec = a G1-style "regulated" spec plus `memory` {model, params, evidence}; `memory: {"model": "oracle"}` is the
evaluation-only Oracle reference (memory answers with programme F's converged per-disposition rates of the
disposition in force; never a candidate). Per lifetime the job returns behaviour totals and instrumentation, the
memory's encounter classification counts, end-of-episode memory snapshots for every identity, the dispositions in
force per episode, and the encounter log (end time, identity, confidence, class, reliable, p at encounter start,
true disposition) for predictive metrics.

    python scripts/g2_run.py --partition P --n 6 --systems configs/G2_pilot_specs.json --scenarios S1,S2,S3,S5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

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
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
from g_common import RESULT_KEYS, init, make_policy  # noqa: E402

from apm_research.harness import run_lifetime  # noqa: E402
from apm_select.instrument import Instrumented  # noqa: E402
from apm_select.memory import BeliefMemory  # noqa: E402
from apm_select.provenance import confirmatory_lock, environment, write_json  # noqa: E402
from apm_select.scenarios import EPISODES, SCENARIOS, SensorGaps, dispositions, make_lifetime  # noqa: E402

ORACLE_RATES = json.loads((HERE.parents[0] / "F_behavioral_learning" / "configs" / "oracle_rates.json")
                          .read_text())["rates_by_disposition"]
NAMES = [f"ped-{i}" for i in range(8)]


class OracleMemory(BeliefMemory):
    """Evaluation reference: p = converged rate of the identity's true disposition in the current episode."""

    def set_truth(self, truth: dict[str, str]) -> None:
        self.truth = truth

    def p_collision(self, identity, kind):
        if kind == "person" and identity in getattr(self, "truth", {}):
            return ORACLE_RATES[self.truth[identity]], 1.0
        return super().p_collision(identity, kind)


class Tracker:
    """Outer wrapper: per-episode ground truth to the Oracle reference and memory snapshots (evaluation only)."""

    def __init__(self, policy, lifetime, inner):
        self.policy, self.lt, self.inner = policy, lifetime, inner
        self.snapshots: list[dict] = []
        self.truth: list[dict] = []
        self.episode = 0

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.truth.append(dispositions(self.lt, self.episode))
        if isinstance(getattr(self.inner, "memory", None), OracleMemory):
            self.inner.memory.set_truth(self.truth[-1])
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.policy.observe(obs, t)

    def decide(self, t: float) -> int:
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        self.policy.end_episode(t)
        mem = self.inner.memory
        self.snapshots.append({n: [round(v, 5) for v in mem.p_collision(n, "person")] for n in NAMES
                               if n in self.truth[-1]})
        self.episode += 1


def run_job(job: dict) -> dict:
    spec, sc = job["spec"], job["scenario"]
    lt, roles = make_lifetime(sc, S.world_seed("G2", job["partition"], job["index"]), job.get("episodes", EPISODES))
    inner = make_policy(spec)
    if spec.get("memory", {}).get("model") == "oracle":
        base_begin = inner.begin_lifetime

        def begin_with_oracle():
            base_begin()
            inner.memory = OracleMemory("M0", {}, spec["memory"].get("evidence", "all"))
        inner.begin_lifetime = begin_with_oracle
    policy = Instrumented(SensorGaps(inner) if sc == "S9" else inner)
    tracker = Tracker(policy, lt, inner)
    r = run_lifetime(lt, tracker)
    mem = inner.memory
    truth_by_time = []
    for (t_end, ident, conf, klass, reliable, p0) in mem.log:
        e = min(len(tracker.truth) - 1, int(t_end // (lt.config.episode_s + 10.0)))
        truth_by_time.append([t_end, ident, conf, klass, reliable, p0, tracker.truth[e].get(ident) if ident else None,
                              e])
    return {"label": spec.get("label"), "config_id": spec.get("config_id"), "scenario": sc, "roles": roles,
            "index": job["index"], "partition": job["partition"],
            "totals": {k: float(r[k]) for k in RESULT_KEYS} | {k: float(v) for k, v in policy.stats.items()},
            "actions": r["actions"], "memory_counts": dict(mem.counts), "state_size": mem.state_size(),
            "probes": getattr(inner, "probes", 0),
            "snapshots": tracker.snapshots, "truth": tracker.truth, "encounters": truth_by_time}


def run_jobs(jobs: list[dict], workers: int, out_path: Path | None = None) -> list[dict]:
    out = []
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        with Pool(workers, initializer=init) as pool:
            for r in pool.imap_unordered(run_job, jobs, chunksize=1):
                out.append(r)
                if fh:
                    fh.write(json.dumps(r) + "\n")
    finally:
        if fh:
            fh.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--systems", type=Path, required=True)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G2"))
    ap.add_argument("--addendum", default="G2_ADDENDUM.md")
    args = ap.parse_args()
    provenance = (confirmatory_lock(HERE / "prereg" / args.addendum) if args.partition in ("X", "S")
                  else environment())
    start = time.time()
    systems = json.loads(args.systems.read_text())["systems"]
    systems = {k: {kk: vv for kk, vv in v.items() if kk not in ("label", "config_id")} for k, v in systems.items()}
    jobs = [{"spec": {**spec, "label": name}, "scenario": sc, "partition": args.partition, "index": args.first + i}
            for name, spec in systems.items() for sc in args.scenarios.split(",") for i in range(args.n)]
    out_dir = args.out / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_jobs(jobs, args.workers, out_dir / "lifetimes.jsonl")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "systems": systems, "n": args.n,
                                           "scenarios": args.scenarios.split(","), "jobs": len(res),
                                           "wall_s": round(time.time() - start, 1)})
    print("done", len(res), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
