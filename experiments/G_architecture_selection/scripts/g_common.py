"""Shared programme-G helpers: system specs → policies, environment conditions, instrumented lifetime jobs.

A spec is a JSON dict. `kind`:
  "b0"         always GO
  "regulated"  B1 + regulator (`regulator` ∈ static | ema | hysteresis | bayes | learned, `params`), lam_c, lam_s
  "b2v"        B2 variant (`mode` ∈ pad | appraisal | filter), lam_c, lam_s, k, k_t, tau, c
Optional `trace` (list of (m, detour bonus, yield bonus)) replays a recorded modulation (negative controls).

Conditions (environment variants; the policy is never told which one is in force):
  nominal      the F/E2 world (WorldConfig defaults, population 6, 40% careless)
  miscal_x3    the robot-side TTC estimates reach the policy multiplied by 3 (clipped to 1), as in F's E7
  dense        4–6 people present per episode from a population of 8
  sparse       1–2 people present per episode
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.policies import B0  # noqa: E402
from apm_research.predictor import EnsemblePredictor  # noqa: E402
from apm_select.instrument import Instrumented  # noqa: E402
from apm_select.regulators import B2Variant, Regulated, make_regulator  # noqa: E402
from apm_testbeds.wheeled_ctrl import WorldConfig  # noqa: E402

PREDICTOR = "runs/E2/predictor.json"   # programme F's frozen predictor (SHA-256 0735d854…be314), unchanged
PREDICTOR_SHA256 = "0735d85487ac031cccbb171745ec8e3d5daf6f111d5b3daadc3401b2ed4be314"
_STATE: dict = {}


def init(predictor_path: str | None = PREDICTOR) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    if predictor_path:
        _STATE["predictor"] = EnsemblePredictor.from_dict(json.loads(Path(predictor_path).read_text()))


def make_policy(spec: dict):
    kind = spec["kind"]
    if kind == "b0":
        return B0()
    pred = _STATE["predictor"]
    trace = [tuple(x) for x in spec["trace"]] if spec.get("trace") else None
    if kind == "regulated":
        return Regulated(pred, spec["lam_c"], spec["lam_s"], make_regulator(spec["regulator"], spec.get("params", {})),
                         trace=trace, name=spec.get("label", "R"))
    if kind == "b2v":
        return B2Variant(pred, spec["lam_c"], spec["lam_s"], spec["k"], spec["k_t"], mode=spec["mode"],
                         tau=spec.get("tau", 1.0), c=spec.get("c", 0.0), trace=trace, name=spec.get("label", "B2"))
    raise ValueError(kind)


class MiscalibratedInput:
    """Passes the policy every observation with ttc_p × 3 (clipped); everything else unchanged."""

    def __init__(self, policy, factor: float = 3.0):
        self.policy, self.factor = policy, factor

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        ents = tuple(dataclasses.replace(e, ttc_p=min(1.0, self.factor * e.ttc_p)) for e in obs.entities)
        self.policy.observe(dataclasses.replace(obs, entities=ents), t)

    def decide(self, t: float) -> int:
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        self.policy.end_episode(t)


CONDITIONS = ("nominal", "miscal_x3", "dense", "sparse")


def lifetime_for(experiment: str, partition: str, index: int, n_episodes: int, condition: str = "nominal") -> Lifetime:
    ss = S.world_seed(experiment, partition, index)
    if condition == "dense":
        return Lifetime(ss, n_episodes=n_episodes, config=WorldConfig(people=(4, 6)), n_people=8)
    if condition == "sparse":
        return Lifetime(ss, n_episodes=n_episodes, config=WorldConfig(people=(1, 2)))
    if condition in ("nominal", "miscal_x3"):
        return Lifetime(ss, n_episodes=n_episodes)
    raise ValueError(condition)


RESULT_KEYS = ("minutes", "goals_completed", "goals_issued", "timeouts", "goal_time_sum", "collisions",
               "person_collisions", "careless_collisions", "careful_collisions", "obstacle_collisions", "severe",
               "slips", "interventions", "distance", "progress", "idle", "periods", "fallbacks", "J")


def run_job(job: dict) -> dict:
    """job: spec, experiment, partition, index, n_episodes, condition, keep_readouts."""
    spec = job["spec"]
    cond = job.get("condition", "nominal")
    lt = lifetime_for(job["experiment"], job["partition"], job["index"], job["n_episodes"], cond)
    inner = make_policy(spec)
    policy = Instrumented(MiscalibratedInput(inner) if cond == "miscal_x3" else inner)
    r = run_lifetime(lt, policy)
    out = {"label": spec.get("label", spec["kind"]), "index": job["index"], "partition": job["partition"],
           "condition": cond, "config_id": spec.get("config_id"),
           "totals": {k: float(r[k]) for k in RESULT_KEYS} | {k: float(v) for k, v in policy.stats.items()},
           "actions": r["actions"]}
    if job.get("keep_readouts") and hasattr(inner, "readouts"):
        out["readouts"] = [list(map(float, x)) for x in inner.readouts]
    return out


def run_jobs(jobs: list[dict], workers: int, out_path: Path | None = None, predictor: str = PREDICTOR) -> list[dict]:
    """Runs jobs in a process pool; streams results to `out_path` (jsonl, readouts stripped) if given."""
    results = []
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        with Pool(workers, initializer=init, initargs=(predictor,)) as pool:
            for r in pool.imap_unordered(run_job, jobs, chunksize=1):
                results.append(r)
                if fh:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != "readouts"}) + "\n")
    finally:
        if fh:
            fh.close()
    return results
