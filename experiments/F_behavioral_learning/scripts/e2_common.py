"""Shared E2–E4 helpers: system construction from JSON specs and parallel lifetime execution."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.policies import B0, B1, B2  # noqa: E402
from apm_research.predictor import EnsemblePredictor  # noqa: E402

_STATE: dict = {}


def _init(predictor_path: str | None) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    if predictor_path:
        _STATE["predictor"] = EnsemblePredictor.from_dict(json.loads(Path(predictor_path).read_text()))


def make_policy(spec: dict):
    kind = spec["system"]
    if kind == "B0":
        return B0()
    pred = _STATE["predictor"]
    if kind == "B1":
        return B1(pred, spec["lam_c"], spec["lam_s"], oracle_rates=spec.get("oracle_rates"))
    if kind == "B2":
        return B2(pred, spec["lam_c"], spec["lam_s"], spec["k_pad"], spec["k_t"], tonic=spec.get("tonic", True),
                  memory_frames=spec.get("memory_frames", True), persistent=spec.get("persistent", True),
                  trace=spec.get("trace"), name=spec.get("name", "B2"))
    raise ValueError(kind)


def lifetime_for(experiment: str, partition: str, index: int, n_episodes: int, **kwargs) -> Lifetime:
    return Lifetime(S.world_seed(experiment, partition, index), n_episodes=n_episodes, **kwargs)


def _job(job: dict) -> dict:
    spec = job["spec"]
    lt = lifetime_for(job["experiment"], job["partition"], job["index"], job["n_episodes"])
    if "oracle_by_disposition" in spec:  # Oracle: this lifetime's identities → true rate of their disposition
        rates = spec["oracle_by_disposition"]
        spec = {**spec, "oracle_rates": {i.name: rates[i.disposition] for i in lt.population()}}
    policy = make_policy(spec)
    result = run_lifetime(lt, policy)
    result["episodes"] = [{k: v for k, v in e.items() if k != "collision_log"} | {"collision_log": e["collision_log"]}
                          for e in result["episodes"]]
    out = {"label": job["spec"].get("label", job["spec"]["system"]), "index": job["index"],
           "partition": job["partition"], "result": result}
    if hasattr(policy, "readouts"):
        out["readouts"] = policy.readouts
    return out


def run_jobs(jobs: list[dict], predictor_path: str | None, workers: int = 4) -> list[dict]:
    with Pool(workers, initializer=_init, initargs=(predictor_path,)) as pool:
        return pool.map(_job, jobs, chunksize=1)
