"""I0 pilot (partition P only): how much does one encounter tell the robot about a person's disposition?

Runs G's decision-A rule with F's cumulative memory (A0 + M0_all) and records, per person encounter, what the robot
itself observed: collision or not, G's encounter class, and a behavioural statistic computed from robot-side
observations only. Ground-truth dispositions are attached for *evaluation only*.

Behavioural statistic (robot-observable):
- Over frames in which the robot moves (|v| > 0.15 m/s) and the person is within 1.5 m, compute the person's own
  radial velocity toward the robot, u = closing − v·cos(bearing). Closing speed and bearing come from `Seen`; the
  robot's own speed comes from `Obs`.
- Summaries: the number of such frames, the mean of u, and the minimum gap.

    python experiments/I_autonomous_revision/scripts/i0_evidence_pilot.py --n 24 --workers 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
G = REPO / "experiments" / "G_architecture_selection"
sys.path.insert(0, str(G / "scripts"))

from g_common import init, make_policy  # noqa: E402

from apm_research.harness import run_lifetime  # noqa: E402
from apm_select.scenarios import dispositions, make_lifetime  # noqa: E402
from apm_testbeds.wheeled_ctrl import ACTIONS  # noqa: E402


def _seeds():
    spec = importlib.util.spec_from_file_location("i_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


S = _seeds()
A0 = {**json.loads((G / "configs" / "G6_systems.json").read_text())["rules"]["A0"],
      "memory": {"model": "M0", "evidence": "all"}}
NEAR_M, MOVING = 1.5, 0.15


class Recorder:
    """Wraps a policy; records robot-observable per-encounter statistics. Truth is attached only for evaluation."""

    def __init__(self, policy, lifetime):
        self.policy, self.lt = policy, lifetime
        self.episode = -1
        self.open: dict[str, dict] = {}
        self.done: list[dict] = []

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.episode += 1
        self.truth = dispositions(self.lt, self.episode)
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.policy.observe(obs, t)
        coll = {ev[1] for ev in obs.events if ev[0] == "collision"}
        seen = set()
        for e in obs.entities:
            if e.kind != "person":
                continue
            seen.add(e.track)
            r = self.open.setdefault(e.track, {"frames": 0, "u_sum": 0.0, "u_pos": 0, "min_gap": 9.0,
                                               "identity": None, "conf": 0.0, "collision": False,
                                               "actions": defaultdict(int), "episode": self.episode, "last": t})
            if e.identity is not None and e.identity_confidence >= r["conf"]:
                r["identity"], r["conf"] = e.identity, e.identity_confidence
            r["min_gap"] = min(r["min_gap"], e.distance)
            r["last"] = t
            if e.distance < NEAR_M:
                r["actions"][ACTIONS[obs.action]] += 1
                if abs(obs.v) > MOVING:
                    u = e.closing - obs.v * math.cos(e.bearing)
                    r["frames"] += 1
                    r["u_sum"] += u
                    r["u_pos"] += u > 0
            if e.track in coll:
                r["collision"] = True
        for track in [k for k, r in self.open.items() if k not in seen and t - r["last"] > 1.5]:
            self._close(track)

    def _close(self, track: str) -> None:
        r = self.open.pop(track)
        r["truth"] = self.truth.get(r["identity"]) if r["identity"] else None
        r["actions"] = dict(r["actions"])
        r["u_mean"] = r["u_sum"] / r["frames"] if r["frames"] else None
        self.done.append(r)

    def decide(self, t: float) -> int:
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        for track in list(self.open):
            self._close(track)
        self.policy.end_episode(t)


def run_job(job: dict) -> list[dict]:
    lt, _ = make_lifetime(job["scenario"], S.world_seed("I0", "P", job["index"]))
    rec = Recorder(make_policy(A0), lt)
    run_lifetime(lt, rec)
    for r in rec.done:
        r.update(scenario=job["scenario"], index=job["index"])
    return rec.done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--scenarios", default="S3,S1,S2")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "I0" / "P" / "encounters.jsonl")
    a = ap.parse_args()
    jobs = [{"scenario": sc, "index": i} for sc in a.scenarios.split(",") for i in range(a.n)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with Pool(a.workers, initializer=init) as pool, open(a.out, "w", encoding="utf-8") as fh:
        for rows in pool.imap_unordered(run_job, jobs):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("done", len(jobs), "lifetimes", round(time.time() - start, 1), "s ->", a.out)


if __name__ == "__main__":
    main()
