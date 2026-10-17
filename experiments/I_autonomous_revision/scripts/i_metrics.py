"""Programme I metrics (definitions fixed in MASTER_PREREGISTRATION.md §5).

- **Belief adaptation** (S1, S2; each switched identity): G2's definition, unchanged. The identity is established if
  at the end of episode 14 its memory p lies on the pre-switch side of the median p of the identities present. Time =
  episodes from 15 until the end-of-episode p first lies on the post-switch side (1 = adapted by the end of the switch
  episode). Never → censored at 15. Kaplan–Meier median and restricted mean (RMST, horizon 15).
- **Behavioural adaptation** (S1): share of the switched identities' encounters in episodes 15–29 that the robot
  avoided (G's class: a careful action while the person was within 1.5 m). Lower is better: they are careful now.
- **Safety:** avoidable collisions per minute (G9 definition: collisions after the first simulation step), careless
  collisions per minute.
- **Efficiency:** goal completion.
- **Evidence:** share of encounters that are censored, i.e. carry no evidence (the passive logger's dense
  coding, identical for every system).
- **Prediction:** per-encounter Brier of memory p at encounter start vs collision, over encounters of reliable
  identities (all systems that log p).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apm_select.survival import km_bootstrap

SWITCH, HORIZON = 15, 15


def load(paths: list[Path]) -> dict:
    data: dict = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            data[r["label"]][r["scenario"]].append(r)
    return data


def adaptation_items(r: dict) -> list[tuple[float, bool]]:
    to = r["roles"]["to"]
    items = []
    for ident in r["roles"]["switched"]:
        def side(e, ident=ident):
            snap = r["snapshots"][e]
            if ident not in snap:
                return None
            med = float(np.median([v[0] for v in snap.values()]))
            p = snap[ident][0]
            return p > med if to == "careless" else p < med
        pre = side(SWITCH - 1)
        if pre is None or pre:
            continue
        t = next((e - SWITCH + 1 for e in range(SWITCH, len(r["snapshots"])) if side(e)), None)
        items.append((float(t), True) if t is not None else (float(HORIZON), False))
    return items


def adaptation(rows: list[dict], n_boot: int = 2000) -> dict:
    return km_bootstrap([adaptation_items(r) for r in rows], HORIZON, n_boot)


def behavioural_avoidance(rows: list[dict]) -> dict:
    """Per lifetime: share of switched identities' post-switch encounters avoided (S1)."""
    vals = []
    for r in rows:
        ids = set(r["roles"].get("switched", []))
        enc = [e for e in r["encounters"] if e[0] >= SWITCH and e[2] in ids]
        if enc:
            vals.append(float(np.mean([e[7] == "avoided" for e in enc])))
    return {"mean": float(np.mean(vals)) if vals else None, "per_lifetime": vals}


def rate(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r["totals"][key] / max(1e-9, r["totals"]["minutes"]) for r in rows])


def completion(rows: list[dict]) -> np.ndarray:
    return np.array([r["totals"]["goals_completed"] / max(1.0, r["totals"]["goals_issued"]) for r in rows])


def censored_share(rows: list[dict]) -> float:
    enc = [e for r in rows for e in r["encounters"] if e[2] is not None and e[3] >= 0.5]
    return float(np.mean([e[8] is None for e in enc])) if enc else float("nan")


def summary(rows: list[dict]) -> dict:
    return {"lifetimes": len(rows),
            "avoidable_per_min": float(rate(rows, "avoidable_collisions").mean()),
            "careless_per_min": float(rate(rows, "careless_collisions").mean()),
            "completion": float(completion(rows).mean()),
            "censored_share": censored_share(rows),
            "probes_per_lifetime": float(np.mean([r["probes"] for r in rows]))}


def ranking_auroc(r: dict, episode: int = -1) -> float:
    """F's E3 criterion (i): per lifetime, memory p of the identities present at the end of `episode` vs their true
    disposition (careless = positive). NaN if only one disposition is present."""
    from apm.learning.evaluation import auroc
    snap, truth = r["snapshots"][episode], r["truth"][episode]
    ids = [k for k in snap if k in truth]
    return auroc([snap[k][0] for k in ids], [truth[k] == "careless" for k in ids])


def discrimination(rows: list[dict]) -> float:
    vals = [x for x in (ranking_auroc(r) for r in rows) if not np.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")
