"""J1 metrics: programme I's adaptation evaluation with calibration frozen on separate data, and stricter measures.

All thresholds come from a calibration file computed on J1-V lifetimes (`calibrate`). The calibration is frozen in
an addendum before the J1-X test lifetimes exist.

Belief measures (per system, on the system's own p scale):
- `calibrated`: programme I's adaptation definition, with the frozen threshold θ = midpoint of μ_careless and
  μ_careful.
- `decisive`: established as above; adapted only when the end-of-episode p is beyond the midpoint between θ and the
  post-switch disposition's μ, for ≥ 2 consecutive episodes (or at the final episode). Forgetting toward the prior
  cannot satisfy it unless the prior itself lies that deep.
- `belief_error`: |p − μ(disposition in force)| / |μ_careless − μ_careful|, per episode, for switched identities.
- `collision_logloss`: log loss and Brier of the memory's p at encounter start vs the encounter's collision outcome, for
  switched identities after the switch (every memory; sparse outcome).
- `dense_logp`: mean log predictive probability of the observed dense outcome category, for switched identities
  before and after the switch (type memories only).

Decision measure:
- `decision`: per episode, the share of near frames (person within 1.5 m) in which the robot used a careful action
  toward the switched identity, as a 3-episode rolling mean. Threshold = midpoint of the calibration means for
  careless and careful identities. Adapted when the rolling mean crosses to the post-switch side (Kaplan–Meier, horizon
  15).
"""
from __future__ import annotations

import numpy as np

from apm_select.survival import km_bootstrap

SWITCH, HORIZON, CAL_FROM, EPISODES = 15, 15, 10, 30
SWITCH_T = SWITCH * 70.0  # episodes of 60 s plus a 10 s gap (apm_research.harness)
NEAR, CAREFUL = 11, 12  # extended encounter-row columns (j_common)


def calibrate(rows_s3: list[dict]) -> dict:
    car, caf = [], []
    dc, df = [0.0, 0.0], [0.0, 0.0]  # [careful frames, near frames] for careless / careful identities
    for r in rows_s3:
        for snap, truth in zip(r["snapshots"][CAL_FROM:], r["truth"][CAL_FROM:], strict=True):
            for k, v in snap.items():
                (car if truth.get(k) == "careless" else caf).append(v[0])
        for e in r["encounters"]:
            if e[0] < CAL_FROM or e[2] is None or e[3] < 0.5 or e[10] is None or not e[NEAR]:
                continue
            acc = dc if e[10] == "careless" else df
            acc[0] += e[CAREFUL]
            acc[1] += e[NEAR]
    mu_c, mu_f = float(np.mean(car)), float(np.mean(caf))
    d_c, d_f = dc[0] / max(1.0, dc[1]), df[0] / max(1.0, df[1])
    return {"mu_careless": mu_c, "mu_careful": mu_f, "threshold": (mu_c + mu_f) / 2, "gap": mu_c - mu_f,
            "dec_careless": d_c, "dec_careful": d_f, "dec_threshold": (d_c + d_f) / 2, "dec_gap": d_c - d_f,
            "lifetimes": len(rows_s3)}


def _careless_side(p: float, cal: dict) -> bool:
    return (1.0 if cal["gap"] >= 0 else -1.0) * (p - cal["threshold"]) > 0


def _items(r: dict, cal: dict, decisive: bool) -> list[tuple[float, bool]]:
    to = r["roles"]["to"]
    mu_post = cal["mu_careless"] if to == "careless" else cal["mu_careful"]
    deep = (cal["threshold"] + mu_post) / 2
    sign_post = 1.0 if mu_post >= cal["threshold"] else -1.0
    items = []
    for ident in r["roles"]["switched"]:
        def p(e, ident=ident):
            snap = r["snapshots"][e]
            return snap[ident][0] if ident in snap else None

        def post(e):
            v = p(e)
            if v is None:
                return False
            if decisive:
                return sign_post * (v - deep) > 0
            c = _careless_side(v, cal)
            return c if to == "careless" else not c
        p14 = p(SWITCH - 1)
        if p14 is None:
            continue
        c14 = _careless_side(p14, cal)
        if c14 if to == "careless" else not c14:  # already on the post-switch side: not established
            continue
        n = len(r["snapshots"])
        t = None
        for e in range(SWITCH, n):
            if post(e) and (not decisive or e == n - 1 or post(e + 1)):
                t = e - SWITCH + 1
                break
        items.append((float(t), True) if t is not None else (float(HORIZON), False))
    return items


def adaptation(rows: list[dict], cal: dict, decisive: bool = False, n_boot: int = 2000) -> dict:
    groups = [_items(r, cal, decisive) for r in rows]
    res = km_bootstrap(groups, HORIZON, n_boot)
    res["established"] = sum(len(g) for g in groups)
    res["switched_total"] = sum(len(r["roles"]["switched"]) for r in rows)
    return res


def belief_error(rows: list[dict], cal: dict) -> dict:
    gap = abs(cal["gap"]) or float("nan")
    curve, count = np.zeros(EPISODES), np.zeros(EPISODES)
    for r in rows:
        for e, (snap, truth) in enumerate(zip(r["snapshots"], r["truth"], strict=True)):
            for ident in r["roles"]["switched"]:
                if ident in snap and truth.get(ident):
                    mu = cal["mu_careless"] if truth[ident] == "careless" else cal["mu_careful"]
                    curve[e] += abs(snap[ident][0] - mu) / gap
                    count[e] += 1
    c = curve / np.maximum(1, count)
    return {"curve": c.tolist(), "post_mean": float(c[SWITCH:].mean()), "pre_mean": float(c[CAL_FROM:SWITCH].mean())}


def collision_logloss(rows: list[dict]) -> dict:
    ll, br = [], []
    for r in rows:
        ids = set(r["roles"]["switched"])
        for t, ident, p0, klass in r.get("p_log", []):
            if ident in ids and p0 is not None and t >= SWITCH_T:
                y = 1.0 if klass == "collision" else 0.0
                q = min(max(p0, 1e-6), 1 - 1e-6)
                ll.append(-(y * np.log(q) + (1 - y) * np.log(1 - q)))
                br.append((q - y) ** 2)
    return {"n": len(ll), "logloss": float(np.mean(ll)) if ll else None, "brier": float(np.mean(br)) if br else None}


def dense_logp(rows: list[dict]) -> dict:
    pre, post = [], []
    for r in rows:
        ids = set(r["roles"]["switched"])
        for t, ident, lp in r.get("logp_log", []):
            if ident in ids:
                (post if t >= SWITCH_T else pre).append(lp)
    return {"pre": float(np.mean(pre)) if pre else None, "post": float(np.mean(post)) if post else None,
            "n_post": len(post)}


def decision_items(r: dict, cal: dict) -> list[tuple[float, bool]]:
    to = r["roles"]["to"]
    sign = 1.0 if cal["dec_gap"] >= 0 else -1.0  # careless side: more careful actions (if gap > 0)
    items = []
    for ident in r["roles"]["switched"]:
        num, den = np.zeros(EPISODES), np.zeros(EPISODES)
        for e in r["encounters"]:
            if e[2] == ident and e[3] >= 0.5 and e[NEAR]:
                num[e[0]] += e[CAREFUL]
                den[e[0]] += e[NEAR]

        def careless_side(ep, num=num, den=den):
            lo = max(0, ep - 2)
            d = den[lo:ep + 1].sum()
            return None if d == 0 else sign * (num[lo:ep + 1].sum() / d - cal["dec_threshold"]) > 0
        pre = careless_side(SWITCH - 1)
        if pre is None or (pre if to == "careless" else not pre):
            continue
        t = None
        for ep in range(SWITCH, EPISODES):
            c = careless_side(ep)
            if c is not None and (c if to == "careless" else not c):
                t = ep - SWITCH + 1
                break
        items.append((float(t), True) if t is not None else (float(HORIZON), False))
    return items


def decision_adaptation(rows: list[dict], cal: dict, n_boot: int = 2000) -> dict:
    groups = [decision_items(r, cal) for r in rows]
    res = km_bootstrap(groups, HORIZON, n_boot)
    res["established"] = sum(len(g) for g in groups)
    return res
