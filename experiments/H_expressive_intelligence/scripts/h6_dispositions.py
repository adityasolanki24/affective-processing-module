"""H6: do persistent, revisable expressive dispositions emerge? (prereg/H6_ADDENDUM.md)

Long lifetimes (96 encounters) are two physically measured X streams joined end to end:

| world | first half | second half |
|---|---|---|
| stable_risky | unpredictable i | unpredictable i+1 |
| stable_safe | safe i | safe i+1 |
| changing | unpredictable i | safe i |
| reverse | safe i | unpredictable i |

The disposition D(k) is the expressed caution κ_e toward an entity of a kind never encountered (D1), so neither
identity nor kind knowledge can explain it. It is read every 4 encounters.

    python scripts/h6_dispositions.py --root results/H34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h34_analyze import _mean_ci, kappa_e  # noqa: E402
from h34_states import TUNE_GRID, learned_model, load_streams, make  # noqa: E402

from apm_express.states import Probe, intent_of  # noqa: E402
from apm_select.survival import km  # noqa: E402

D1 = Probe("D1", "D", "plain")
WORLDS = {
    "stable_risky": ("unpredictable", "unpredictable"),
    "stable_safe": ("safe", "safe"),
    "changing": ("unpredictable", "safe"),
    "reverse": ("safe", "unpredictable"),
}
REPEAT_KS = (60, 72, 84, 96)


def icc1(m: np.ndarray) -> float:
    """ICC(1): one-way random effects, rows = robots, columns = repeated measurements."""
    n, k = m.shape
    grand = m.mean()
    msb = k * ((m.mean(1) - grand) ** 2).sum() / (n - 1)
    msw = ((m - m.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock

    lock = confirmatory_lock(HERE / "prereg" / "H6_ADDENDUM.md")
    xs = load_streams(a.root / "X" / "experience.json", 0.04)
    by = {(s["history"], s["index"]): s["obs"] for s in xs}
    n = 1 + max(i for _, i in by)
    tuning = json.loads((a.root / "tuning.json").read_text(encoding="utf-8"))
    learned = json.loads((a.root / "learned.json").read_text(encoding="utf-8"))
    systems = {"S0_none": lambda: make("S0_none"), "S1_fixed": lambda: make("S1_fixed")}
    for fam in TUNE_GRID:
        systems[fam] = lambda fam=fam: make(fam, tuning[fam]["selected"])
    for kind, name in (("gru", "S5_gru"), ("fastslow", "S6_fastslow"), ("gru_mem", "S7_gru_mem")):
        systems[name] = lambda kind=kind, name=name: learned_model(
            a.root, kind, learned[kind]["selected"], a.seeds, name
        )
    out = {"lock": lock, "systems": {}}
    for name, factory in systems.items():
        curves = {w: [] for w in WORLDS}
        for w, (h1, h2) in WORLDS.items():
            for i in range(n):
                stream = by[(h1, i)] + by[(h2, (i + 1) % n if h1 == h2 else i)]
                m = factory()
                m.reset()
                c = []
                for k, ob in enumerate(stream):
                    m.observe(ob)
                    if (k + 1) % 4 == 0:
                        c.append(kappa_e(intent_of(m, D1).as_dict()))
                curves[w].append(c)
        ks = list(range(4, 97, 4))
        arr = {w: np.array(v) for w, v in curves.items()}
        diff48 = arr["stable_risky"][:, ks.index(48)] - arr["stable_safe"][:, ks.index(48)]
        diff96 = arr["stable_risky"][:, -1] - arr["stable_safe"][:, -1]
        rep = np.concatenate([arr["stable_risky"], arr["stable_safe"]])[:, [ks.index(k) for k in REPEAT_KS]]
        risky_level, safe_level = arr["stable_risky"][:, -1].mean(), arr["stable_safe"][:, -1].mean()
        mid = (risky_level + safe_level) / 2
        rev, acq = [], []
        for c in arr["changing"]:
            after = [(k - 48, v) for k, v in zip(ks, c, strict=True) if k > 48]
            hit = next((t for t, v in after if v <= mid), None)
            rev.append((float(hit if hit is not None else 48), hit is not None))
        for c in arr["reverse"]:
            after = [(k - 48, v) for k, v in zip(ks, c, strict=True) if k > 48]
            hit = next((t for t, v in after if v >= mid), None)
            acq.append((float(hit if hit is not None else 48), hit is not None))
        out["systems"][name] = {
            "world_effect_k48": _mean_ci(diff48),
            "world_effect_k96": _mean_ci(diff96),
            "icc_stable_worlds": icc1(rep) if np.std(rep) > 0 else 0.0,
            "revision_km_median": km(np.array([t for t, _ in rev]), np.array([e for _, e in rev]), 48.0)[0],
            "revised_fraction": float(np.mean([e for _, e in rev])),
            "acquisition_km_median": km(np.array([t for t, _ in acq]), np.array([e for _, e in acq]), 48.0)[0],
            "levels": {"stable_risky": float(risky_level), "stable_safe": float(safe_level)},
            "mean_curves": {w: v.mean(0).round(4).tolist() for w, v in arr.items()},
        }
        r = out["systems"][name]
        r["shows_revisable_disposition"] = bool(
            r["world_effect_k96"]["mean"] > 0.10
            and r["world_effect_k96"]["ci"][0] > 0
            and r["icc_stable_worlds"] >= 0.5
            and r["revision_km_median"] <= 48
            and r["revised_fraction"] >= 0.8
        )
        print(
            name,
            {
                k: (v if not isinstance(v, dict) else v.get("mean", ""))
                for k, v in r.items()
                if k not in ("mean_curves", "levels")
            },
            flush=True,
        )
    (a.root / "X" / "h6_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
