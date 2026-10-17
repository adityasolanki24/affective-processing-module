"""Collect the decision read-out bank (`apm_transfer.readout`) from A0 runs on J1-T (S3), training partition only.

Records the decision-time feature vector and the regulator multiplier whenever the nearest person is reliably
identified and within 1.5 m, subsampled at random to `--size` vectors.

    python experiments/J_validation_transfer/scripts/j1_bank.py --n 40 --size 1000 --out <bank.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j_common import A0, SEEDS, build, init  # noqa: E402

from apm_research.harness import run_lifetime  # noqa: E402
from apm_revise.scenarios import make_lifetime  # noqa: E402


class Collector:
    def __init__(self, policy):
        self.policy = policy
        self.rows: list[tuple[list[float], float]] = []

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.policy.observe(obs, t)

    def end_episode(self, t: float) -> None:
        self.policy.end_episode(t)

    def decide(self, t: float) -> int:
        a = self.policy.decide(t)
        obs = self.policy.state.last_obs
        people = sorted((e for e in obs.entities if e.kind == "person"), key=lambda e: e.distance) if obs else []
        if people and people[0].distance < 1.5 and people[0].identity is not None \
                and people[0].identity_confidence >= 0.5 and self.policy.last_features is not None:
            self.rows.append(([float(v) for v in self.policy.last_features], float(self.policy.regulator.multiplier())))
        return a


def job(i: int) -> list:
    lt, _ = make_lifetime("S3", SEEDS.world_seed("J1", "T", i))
    c = Collector(build({**A0, "memory": {"model": "M0", "evidence": "all"}}))
    run_lifetime(lt, c)
    return c.rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with Pool(a.workers, initializer=init) as pool:
        rows = [r for rs in pool.map(job, range(a.n)) for r in rs]
    rng = np.random.default_rng(20261017)
    pick = rng.choice(len(rows), size=min(a.size, len(rows)), replace=False)
    out = {"source": f"A0 on J1-T S3 lifetimes 0-{a.n - 1}", "available": len(rows),
           "features": [rows[i][0] for i in sorted(pick)], "multipliers": [rows[i][1] for i in sorted(pick)]}
    a.out.write_text(json.dumps(out), encoding="utf-8")
    print("bank", len(out["features"]), "of", len(rows))


if __name__ == "__main__":
    main()
