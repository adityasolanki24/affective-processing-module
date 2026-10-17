"""H2 power simulation (prereg/H2_PROTOCOL.md §5). Simulated *design* data only: no participant data exist.

Model: logit P(hit) = logit(p_style) + u_participant + v_clip, with u ~ N(0, sd_p²) and v ~ N(0, sd_c²).

Design per speed condition: each participant sees `per_style` clips of each style, drawn from `clips_per_style`
distinct clips. Each of the 6 styles is tested one-sided against chance with a t-test on per-participant hit
proportions (a conservative stand-in for the mixed model), Holm across styles. Power = P(a given style is detected).

    python scripts/h2_power.py --n 60 --p 0.30
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

CHANCE = 1.0 / 7.0


def _t_sf(t: float, df: int) -> float:
    """One-sided upper-tail p-value of Student's t via the regularized incomplete beta (numpy-only)."""
    x = df / (df + t * t)
    # continued fraction for the regularized incomplete beta I_x(a, b), a = df/2, b = 1/2
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, itmax=200, eps=3e-14):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    ibeta = front * betacf(a, b, x)
    p_two_sided_tail = ibeta  # P(|T| > |t|)
    return p_two_sided_tail / 2.0 if t > 0 else 1.0 - p_two_sided_tail / 2.0


def simulate(n: int, p: float, sd_p: float, sd_c: float, per_style: int, clips_per_style: int, sims: int,
             seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    logit = math.log(p / (1 - p))
    detected = np.zeros(6)
    all_six = 0
    for _ in range(sims):
        u = rng.normal(0, sd_p, n)
        pvals = []
        for _ in range(6):
            v = rng.normal(0, sd_c, clips_per_style)
            clips = rng.integers(0, clips_per_style, (n, per_style))
            eta = logit + u[:, None] + v[clips]
            hits = rng.random((n, per_style)) < 1 / (1 + np.exp(-eta))
            prop = hits.mean(axis=1)
            sd = prop.std(ddof=1)
            t = (prop.mean() - CHANCE) / (sd / math.sqrt(n)) if sd > 0 else (np.inf if prop.mean() > CHANCE else 0)
            pvals.append(0.0 if t == np.inf else _t_sf(float(t), n - 1))
        order = np.argsort(pvals)
        reject = np.zeros(6, dtype=bool)
        for rank, i in enumerate(order):  # Holm step-down
            if pvals[i] <= 0.05 / (6 - rank):
                reject[i] = True
            else:
                break
        detected += reject
        all_six += reject.all()
    return {"n": n, "p": p, "sd_participant": sd_p, "sd_clip": sd_c, "trials_per_style_per_participant": per_style,
            "clips_per_style": clips_per_style, "sims": sims, "power_per_style": float(detected.mean() / sims),
            "power_all_six": all_six / sims}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--p", type=float, default=0.30)
    ap.add_argument("--sd-p", type=float, default=0.5)
    ap.add_argument("--sd-c", type=float, default=0.5)
    ap.add_argument("--per-style", type=int, default=3)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--sims", type=int, default=2000)
    a = ap.parse_args()
    print(json.dumps(simulate(a.n, a.p, a.sd_p, a.sd_c, a.per_style, a.clips, a.sims)))


if __name__ == "__main__":
    main()
