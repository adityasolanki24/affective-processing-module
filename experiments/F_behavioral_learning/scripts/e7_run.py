"""E7: robustness and safety independence (MASTER_PREREGISTRATION.md §2 E7).

Each fault is injected between the simulator and the decision system (observation faults), into the system's own
components (predictor / APM / memory faults), or into the world configuration. The simulator, navigation and the
safety supervisor are never altered. B0 runs on the same faulted lifetimes as the reference (B0 ignores all inputs,
so observation faults leave it unchanged by construction). Criteria (1)–(3) are also unit-tested in
tests/test_e7_robustness.py; this runner measures (1) exceptions caught by the harness and (4) collision rates.

    python scripts/e7_run.py --partition P --n 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import seeds as S  # noqa: E402
from e2_common import _init, make_policy  # noqa: E402

from apm.adapters.base import Emitter  # noqa: E402
from apm.adapters.wheeled_ctrl import WheeledCtrlAdapter  # noqa: E402
from apm.memory.entities import EntityMemory  # noqa: E402
from apm_research.harness import EPISODE_GAP_S, Lifetime, run_episode, summarize  # noqa: E402
from apm_research.policies import AblatedAPM  # noqa: E402
from apm_research.provenance import confirmatory_lock, environment, write_json  # noqa: E402
from apm_testbeds.wheeled_ctrl import Identity, WorldConfig, child  # noqa: E402

EPISODES = 5
FAULTS = ("none", "missing_obs", "stale_obs", "sensor_outage", "conflicting_predictions", "wrong_identities",
          "corrupted_memory", "out_of_order_duplicated", "event_rate_10x", "miscalibrated_x3", "world_change",
          "apm_crash_restart", "model_failure", "delayed_inference", "uncertain_outcomes")


class FaultyPolicy:
    """Applies one fault to what a policy receives or to its components. Counts exceptions it lets through (the
    harness catches them and falls back to GO; criterion 1 asks how often that path was needed)."""

    def __init__(self, policy, fault: str, rng: np.random.Generator, tmp: Path):
        self.policy, self.fault, self.rng, self.tmp = policy, fault, rng, tmp
        self.exceptions = 0
        self.prev = None
        self.pending_action: int | None = None
        self.outage_until = -1.0

    # ------------------------------------------------------------------ lifecycle
    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        if self.fault == "corrupted_memory" and hasattr(self.policy, "memory"):
            self._corrupt_and_restore(t0)
        self.pending_action = None
        self._guard(self.policy.begin_episode, t0)

    def end_episode(self, t: float) -> None:
        self._guard(self.policy.end_episode, t)

    def _guard(self, fn, *args):
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001 - counted, then re-raised for the harness's own fallback
            self.exceptions += 1
            raise

    def _corrupt_and_restore(self, t0: float) -> None:
        """Save state, damage the file, restore through the guarded path (fresh state when unreadable)."""
        self.policy.memory.expire(t0)
        path = self.tmp / f"mem_{id(self)}.json"
        path.write_text(json.dumps(self.policy.memory.mem.to_dict()))
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])  # truncated write
        try:
            self.policy.memory.mem = EntityMemory.from_dict(json.loads(path.read_text()))
        except (ValueError, KeyError, TypeError):
            self.policy.memory.mem = EntityMemory(self.policy.memory.mem.detectable)
        if hasattr(self.policy, "apm"):
            apm_path = self.tmp / f"apm_{id(self)}.json"
            self.policy.apm.save(apm_path)
            apm_path.write_text(apm_path.read_text()[:100])
            from apm.core.runtime import load_or_new
            apm, _ = load_or_new(apm_path, self.policy.adapter.profile())
            fresh = AblatedAPM(profile=apm.profile, tonic=self.policy.tonic, memory_frames=self.policy.memory_frames)
            fresh.ingest(self.policy.adapter.profile(), now_s=t0)
            self.policy.apm = fresh

    # ------------------------------------------------------------------ observations
    def _transform(self, obs):
        f, rng = self.fault, self.rng
        if f == "sensor_outage":
            if obs.t >= self.outage_until + 13.0:
                self.outage_until = obs.t + 2.0
            if obs.t < self.outage_until:
                return [dataclasses.replace(obs, entities=())]
        if f == "conflicting_predictions":
            ents = tuple(dataclasses.replace(e, ttc_p=float(rng.random())) if rng.random() < 0.2 else e
                         for e in obs.entities)
            return [dataclasses.replace(obs, entities=ents)]
        if f == "miscalibrated_x3":
            return [dataclasses.replace(obs, entities=tuple(dataclasses.replace(e, ttc_p=min(1.0, 3 * e.ttc_p))
                                                            for e in obs.entities))]
        if f == "event_rate_10x":
            return [dataclasses.replace(obs, events=tuple(ev for ev in obs.events for _ in range(10)))]
        if f == "uncertain_outcomes":
            events = tuple(ev for ev in obs.events if ev[0] != "collision" or rng.random() < 0.7)
            people = [e for e in obs.entities if e.kind == "person"]
            if people and rng.random() < 0.002:  # spurious collision report (≈ 1 per 25 s)
                events += (("collision", people[int(rng.integers(len(people)))].track, False),)
            return [dataclasses.replace(obs, events=events)]
        if f == "missing_obs" and rng.random() < 0.2:
            return []
        if f == "stale_obs" and self.prev is not None and rng.random() < 0.3:
            return [dataclasses.replace(self.prev, events=obs.events)]  # stale state, events still arrive
        if f == "out_of_order_duplicated":
            out = [obs]
            if rng.random() < 0.1:
                out = [obs, obs]
            if self.prev is not None and rng.random() < 0.1:
                out = [obs, self.prev]  # an older observation arrives after a newer one
            return out
        return [obs]

    def observe(self, obs, t: float) -> None:
        if self.fault == "apm_crash_restart" and hasattr(self.policy, "apm") and self.rng.random() < 0.002:
            fresh = AblatedAPM(profile=self.policy.adapter.profile(), tonic=self.policy.tonic,
                               memory_frames=self.policy.memory_frames)
            fresh.ingest(self.policy.adapter.profile(), now_s=t)
            self.policy.apm = fresh
            self.policy.adapter = WheeledCtrlAdapter(Emitter("wheeled-ctrl-restarted"))
        if self.fault == "apm_crash_restart" and hasattr(self.policy, "memory") and not hasattr(self.policy, "apm") \
                and self.rng.random() < 0.002:
            self.policy.memory.mem = EntityMemory(self.policy.memory.mem.detectable)  # B1: memory process restart
        for o in self._transform(obs):
            self._guard(self.policy.observe, o, t)
        self.prev = obs

    def decide(self, t: float) -> int:
        if self.fault == "model_failure" and hasattr(self.policy, "predictor") and self.rng.random() < 0.1:
            self.exceptions += 1
            raise RuntimeError("injected model failure")
        a = self._guard(self.policy.decide, t)
        if self.fault == "delayed_inference":  # the decision computed now takes effect one period (0.5 s) later
            a, self.pending_action = (self.pending_action if self.pending_action is not None else 0), a
        return a


def lifetime_for(fault: str, partition: str, index: int) -> Lifetime:
    ss = S.world_seed("E7", partition, index)
    if fault == "wrong_identities":
        return Lifetime(ss, n_episodes=EPISODES, config=WorldConfig(identity_error=0.3, identity_ambiguous=0.2))
    if fault == "world_change":  # at episode 2 every disposition flips and low-friction patches double
        def sched(e, base):
            if e < 2:
                return base
            return [Identity(i.name, "careful" if i.disposition == "careless" else "careless") for i in base]
        return Lifetime(ss, n_episodes=EPISODES, schedule=sched)
    return Lifetime(ss, n_episodes=EPISODES)


def run_job(job: dict) -> dict:
    lt = lifetime_for(job["fault"], job["partition"], job["index"])
    tmp = Path(job["tmp"])
    policy = FaultyPolicy(make_policy(job["spec"]), job["fault"],
                          np.random.default_rng(child(S.world_seed("E7", job["partition"], job["index"]), 900)), tmp)
    policy.begin_lifetime()
    rows = []
    for e in range(EPISODES):
        if job["fault"] == "world_change" and e >= 2:
            lt.config = WorldConfig(patches=(2, 6))
        rows.append(run_episode(lt.world(e), policy, e * (lt.config.episode_s + EPISODE_GAP_S)))
    out = summarize(rows)
    for r in out["episodes"]:
        r.pop("collision_log", None)
    return {"fault": job["fault"], "system": job["spec"]["label"], "index": job["index"],
            "exceptions": policy.exceptions, "fallbacks": out["fallbacks"],
            "totals": {k: out[k] for k in ("collisions", "minutes", "goals_completed", "goals_issued", "idle",
                                           "periods", "severe", "interventions")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--faults", default=",".join(FAULTS))
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--tuned", type=Path, default=HERE / "configs" / "E2_tuned.json")
    ap.add_argument("--out", type=Path, default=Path("runs/E7"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    provenance = (confirmatory_lock(HERE / "prereg" / "E7_ADDENDUM.md") if args.partition in ("X", "S")
                  else environment())
    start = time.time()
    tuned = json.loads(args.tuned.read_text())["best"]
    specs = {"B0": {"system": "B0", "label": "B0"},
             "B1": {"system": "B1", "lam_c": tuned["B1"]["lam_c"], "lam_s": tuned["B1"]["lam_s"], "label": "B1"},
             "B2": {"system": "B2", **{k: tuned["B2"][k] for k in ("lam_c", "lam_s", "k_pad", "k_t")},
                    "label": "B2"}}
    out_dir = args.out / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [{"fault": f, "partition": args.partition, "index": i, "spec": specs[s], "tmp": str(out_dir)}
            for f in args.faults.split(",") for s in specs for i in range(args.n)]
    jobs.sort(key=lambda j: j["spec"]["system"] != "B2")
    from multiprocessing import Pool
    with Pool(args.workers, initializer=_init, initargs=(args.predictor,)) as pool, \
            open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            fh.write(json.dumps(r, default=float) + "\n")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "specs": specs, "n": args.n,
                                           "episodes": EPISODES, "wall_s": round(time.time() - start, 1)})
    print("done", len(jobs), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
