"""Right-censored adaptation times: Kaplan–Meier median and restricted mean, cluster bootstrap over lifetimes."""
from __future__ import annotations

import numpy as np


def km(times: np.ndarray, events: np.ndarray, horizon: float) -> tuple[float, float]:
    """(median, restricted mean up to `horizon`) of a KM curve; median = inf if S(t) never drops to 0.5."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    if len(times) == 0:
        return float("nan"), float("nan")
    order = np.argsort(times, kind="stable")
    t, e = times[order], events[order]
    s, surv, grid = 1.0, [], []
    n = len(t)
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = int(e[i:j].sum())
        at_risk = n - i
        if d:
            s *= 1.0 - d / at_risk
            grid.append(t[i])
            surv.append(s)
        i = j
    median = next((g for g, v in zip(grid, surv, strict=True) if v <= 0.5), float("inf"))
    # restricted mean = area under S(t) on [0, horizon]
    area, last_t, last_s = 0.0, 0.0, 1.0
    for g, v in zip(grid, surv, strict=True):
        if g > horizon:
            break
        area += last_s * (g - last_t)
        last_t, last_s = g, v
    area += last_s * (horizon - last_t)
    return float(median), float(area)


def km_bootstrap(groups: list[list[tuple[float, bool]]], horizon: float, n_boot: int = 10_000,
                 seed: int = 0) -> dict:
    """groups: per lifetime, a list of (time, event) items. Resamples lifetimes."""
    flat = [x for g in groups for x in g]
    if not flat:
        return {"median": None, "rmst": None, "items": 0, "lifetimes": len(groups)}
    t = np.array([x[0] for x in flat])
    e = np.array([x[1] for x in flat])
    med, rmst = km(t, e, horizon)
    rng = np.random.default_rng(seed)
    meds, rms = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        items = [x for i in pick for x in groups[i]]
        if not items:
            continue
        m, r = km(np.array([x[0] for x in items]), np.array([x[1] for x in items]), horizon)
        meds.append(min(m, 1e9))
        rms.append(r)
    meds, rms = np.array(meds), np.array(rms)
    return {"median": med, "median_ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "p_median_above_6": float(np.mean(meds > 6)), "rmst": rmst,
            "rmst_ci95": [float(np.percentile(rms, 2.5)), float(np.percentile(rms, 97.5))],
            "items": int(len(flat)), "events": int(e.sum()), "censored": int((~e).sum()), "lifetimes": len(groups)}
