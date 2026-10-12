"""H8: integrated evaluation and architecture selection (prereg/H8_ADDENDUM.md).

    <nerva-venv> python scripts/h34_experience.py --partition X --lifetimes 30 --index-offset 2000 \
        --addendum H8_ADDENDUM.md --out results/H34/X8
    python scripts/h8.py eval    --root results/H34      # intent level + dispositions + compute
    <nerva-venv> python scripts/h8.py probes --root results/H34
    python scripts/h8.py analyze --root results/H34
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

PROBES_PHYS = {  # name -> (probe key, target (x, y))
    "a_B1_plain": ("B1_plain", (1.0, 0.0)),
    "b_B3_marked": ("B3_marked", (1.3 * math.cos(-0.7), 1.3 * math.sin(-0.7))),
    "c_B1_shift": ("B1_plain", (1.8 * math.cos(2.4), 1.8 * math.sin(2.4))),
}
PHYS_SYSTEMS = ("B0", "I", "A0_L1")


def _probes():
    from h34_states import PROBES

    from apm_express.states import Probe

    return {**PROBES, "B3_marked": Probe("B3", "B", "marked")}


class L1Expression:
    """A0 memory (M_id) + the H5 L1 MLP mapping (ensemble mean), no separate developmental latent."""

    name = "A0_L1"
    fixed_intent = None

    def __init__(self, mappers):
        from apm_express.states import EntityMemory

        self.mappers, self.mem, self.counts = mappers, EntityMemory(alpha=0.02), {}
        self.EntityMemory = EntityMemory

    def reset(self):
        self.mem, self.counts = self.EntityMemory(alpha=0.02), {}

    def observe(self, ob):
        self.mem.observe(ob)
        self.counts[ob.identity] = self.counts.get(ob.identity, 0) + 1

    def readout(self, probe):
        return self.mem.readout(probe)

    def express(self, probe):
        import torch
        from h5_mapping import HI, LO, mem_features

        from apm_express.intent import ExpressionIntent

        f = torch.tensor(mem_features(self.mem, probe, self.counts))[None, None, None]
        with torch.no_grad():
            out = np.mean([m.head(f).sigmoid().numpy().reshape(-1) for m in self.mappers], axis=0)
        return ExpressionIntent.from_array(LO + (HI - LO) * out)


def express(model, probe):
    from apm_express.states import intent_of

    return model.express(probe) if hasattr(model, "express") else intent_of(model, probe)


def systems(root: Path) -> dict:
    from h34_states import learned_model

    from apm_express.states import EntityMemory, ScalarEMA

    learned = {}
    for f in sorted(root.glob("learned_*.json")):
        learned.update(json.loads(f.read_text(encoding="utf-8")))
    out = {
        "B0": lambda: EntityMemory(alpha=0.02),
        "I": lambda: EntityMemory(context=True, changepoint=True, alpha=0.07),
        "I_minus_context": lambda: EntityMemory(context=False, changepoint=True, alpha=0.07),
        "I_minus_changepoint": lambda: EntityMemory(context=True, changepoint=False, alpha=0.07),
        "I_minus_global": lambda: EntityMemory(context=True, changepoint=True, alpha=0.0),
        "global_only": lambda: ScalarEMA(alpha=0.07),
        "S6_fastslow": lambda: learned_model(root, "fastslow", learned["fastslow"]["selected"], 5, "S6_fastslow"),
    }
    mappers = l1_mappers(root)
    out["A0_L1"] = lambda: L1Expression(mappers)
    return out


def l1_mappers(root: Path):
    """Retrain H5's selected L1 configuration deterministically (seeds 0-4) on the same T/V data."""
    from h5_mapping import THRESHOLD, lifetime_data, train
    from h34_states import load_streams

    h5 = json.loads((root / "X" / "h5_eval.json").read_text(encoding="utf-8"))["training"]["L1"]["selected"]["cfg"]
    rng = np.random.default_rng(0)
    td = [lifetime_data(s, rng) for s in load_streams(root / "T" / "experience.json", THRESHOLD)]
    vd = [lifetime_data(s, rng) for s in load_streams(root / "V" / "experience.json", THRESHOLD)]
    return [train("L1", td, vd, seed, **h5)[0] for seed in range(5)]


def _course(model, obs, history, probes):
    from apm_express.develop import N_ENCOUNTERS, jolt_probability

    model.reset()
    course = []
    for k, ob in enumerate(obs):
        model.observe(ob)
        if (k + 1) % 4 == 0:
            snap = {}
            for pn, pr in probes.items():
                truth = (
                    jolt_probability(history, pr.identity, pr.zone, min(k + 1, N_ENCOUNTERS - 1))
                    if pr.kind != "D"
                    else None
                )
                snap[pn] = {"intent": express(model, pr).as_dict(), "truth": truth}
            course.append({"k": k + 1, "probes": snap})
    return course


def evaluate(root: Path) -> dict:
    from h34_states import load_streams

    from apm_express.states import Obs

    probes = _probes()
    streams = load_streams(root / "X8" / "experience.json", 0.04)
    by = {(s["history"], s["index"]): s["obs"] for s in streams}
    idx = sorted({i for _, i in by})
    rows, disp, compute = [], {}, {}
    for name, factory in systems(root).items():
        m = factory()
        for s in streams:
            rows.append(
                {
                    "system": name,
                    "control": "none",
                    "history": s["history"],
                    "index": s["index"],
                    "course": _course(m, s["obs"], s["history"], probes),
                    "reset_intents": None,
                }
            )
        for i in idx:  # pooled shuffle (H4b control) for the risky_B / safe pair
            risky, safe = by[("risky_B", i)], by[("safe", i)]
            pooled = [(o.y, o.magnitude) for o in risky + safe]
            perm = np.random.default_rng(i).permutation(len(pooled))
            pooled = [pooled[j] for j in perm]
            pr_ = [Obs(o.identity, o.kind, o.zone, *pooled[k]) for k, o in enumerate(risky)]
            ps_ = [Obs(o.identity, o.kind, o.zone, *pooled[len(risky) + k]) for k, o in enumerate(safe)]
            for h, obs in (("risky_B", pr_), ("safe", ps_)):
                rows.append(
                    {
                        "system": name,
                        "control": "pooled_shuffle",
                        "history": h,
                        "index": i,
                        "course": _course(m, obs, h, {"B1_plain": probes["B1_plain"]}),
                        "reset_intents": None,
                    }
                )
        # dispositions (H6 design): D1 caution over joined streams
        from h34_analyze import kappa_e

        worlds = {
            "stable_risky": ("unpredictable", "unpredictable"),
            "stable_safe": ("safe", "safe"),
            "changing": ("unpredictable", "safe"),
            "reverse": ("safe", "unpredictable"),
        }
        curves = {}
        for w, (h1, h2) in worlds.items():
            cs = []
            for j, i in enumerate(idx):
                stream = by[(h1, i)] + by[(h2, idx[(j + 1) % len(idx)] if h1 == h2 else i)]
                m.reset()
                c = []
                for k, ob in enumerate(stream):
                    m.observe(ob)
                    if (k + 1) % 4 == 0:
                        c.append(kappa_e(express(m, probes["D1_unknown"]).as_dict()))
                cs.append(c)
            curves[w] = cs
        disp[name] = curves
        # compute cost
        s0 = streams[0]["obs"]
        m.reset()
        t = time.perf_counter()
        for ob in s0:
            m.observe(ob)
        t_obs = (time.perf_counter() - t) / len(s0)
        t = time.perf_counter()
        for _ in range(50):
            express(m, probes["B1_plain"])
        compute[name] = {"us_per_observe": t_obs * 1e6, "us_per_expression": (time.perf_counter() - t) / 50 * 1e6}
    return {"rows": rows, "dispositions": disp, "compute": compute}


def probes_physical(root: Path, workers: int = 3) -> dict:
    from multiprocessing import Pool

    from h0_interface import _init

    from apm_express.provenance import confirmatory_lock

    lock = confirmatory_lock(HERE / "prereg" / "H8_ADDENDUM.md")
    iface = json.loads((HERE / "results/H0/X/admitted_interface.json").read_text(encoding="utf-8"))["interface"]
    ev = json.loads((root / "X8" / "h8_eval.json").read_text(encoding="utf-8"))
    jobs = []
    for r in ev["rows"]:
        if r["control"] != "none" or r["system"] not in PHYS_SYSTEMS or r["history"] not in ("risky_B", "safe"):
            continue
        for pname, (pkey, target) in PROBES_PHYS.items():
            jobs.append(
                {
                    "system": r["system"],
                    "history": r["history"],
                    "index": r["index"],
                    "probe": pname,
                    "intent": r["course"][-1]["probes"][pkey]["intent"],
                    "iface": iface,
                    "target": target,
                }
            )
    with Pool(workers, initializer=_init) as pool:
        rows = pool.map(_run, jobs, chunksize=1)
    return {"lock": lock, "rows": rows}


def _run(job: dict) -> dict:
    import h34_probes

    h34_probes.TARGET = tuple(job["target"])
    out = h34_probes.run(job)
    out["probe"] = job["probe"]
    return out


def analyze(root: Path) -> dict:
    from h6_dispositions import icc1
    from h34_analyze import (
        LDA,
        NON_SPEED,
        _end,
        _matrix,
        _mean_ci,
        crossing_times,
        e1_from,
        e1_points,
        kappa_e,
        paired_diff,
    )

    from apm_select.survival import km

    ev = json.loads((root / "X8" / "h8_eval.json").read_text(encoding="utf-8"))
    rows = ev["rows"]
    names = list(dict.fromkeys(r["system"] for r in rows))
    keys = sorted({(r["history"], r["index"]) for r in rows if r["control"] == "none"})
    base = {s: [r for r in rows if r["system"] == s and r["control"] == "none"] for s in names}
    pts = {s: e1_points(base[s]) for s in names}
    res: dict = {"systems": {}}
    for s in names:
        b = base[s]
        safe_l = float(np.mean([kappa_e(_end(r, "B1_plain")["intent"]) for r in b if r["history"] == "safe"]))
        risky_l = float(np.mean([kappa_e(_end(r, "B1_plain")["intent"]) for r in b if r["history"] == "risky_B"]))
        rev = crossing_times(b, "change_B", False, safe_l, risky_l)
        ce = [
            abs(kappa_e(_end(r, p)["intent"]) - _end(r, p)["truth"])
            for r in b
            for p in ("B1_plain", "B1_marked", "A1_plain", "C1_plain", "B3_novel")
        ]
        pooled = [r for r in rows if r["system"] == s and r["control"] == "pooled_shuffle"]
        ks = list(range(4, 97, 4))
        c = {w: np.array(v) for w, v in ev["dispositions"][s].items()}
        diff96 = c["stable_risky"][:, -1] - c["stable_safe"][:, -1]
        rep = np.concatenate([c["stable_risky"], c["stable_safe"]])[:, [ks.index(k) for k in (60, 72, 84, 96)]]
        mid = (c["stable_risky"][:, -1].mean() + c["stable_safe"][:, -1].mean()) / 2
        drev = []
        for cv in c["changing"]:
            hit = next((k - 48 for k, v in zip(ks, cv, strict=True) if k > 48 and v <= mid), None)
            drev.append((float(hit if hit is not None else 48), hit is not None))
        icc = icc1(rep) if np.std(rep) > 1e-12 else float("nan")
        dk = km(np.array([t for t, _ in drev]), np.array([e for _, e in drev]), 48.0)[0]
        de = _mean_ci(diff96)
        res["systems"][s] = {
            "E1": e1_from(pts[s], keys),
            "CE": _mean_ci(ce),
            "CE_values": ce,
            "E2": paired_diff(b, "B1_plain", "risky_B", "safe"),
            "E2_pooled_shuffle": paired_diff(pooled, "B1_plain", "risky_B", "safe"),
            "revision_median": km(np.array([t for t, _ in rev]), np.array([e for _, e in rev]), 24.0)[0],
            "revised_fraction": float(np.mean([e for _, e in rev])),
            "E6": paired_diff(b, "B3_novel", "risky_B", "safe"),
            "E7": _mean_ci(
                [
                    kappa_e(_end(r, "B1_marked")["intent"]) - kappa_e(_end(r, "B1_plain")["intent"])
                    for r in b
                    if r["history"] == "zone_B"
                ]
            ),
            "E7_values": [
                kappa_e(_end(r, "B1_marked")["intent"]) - kappa_e(_end(r, "B1_plain")["intent"])
                for r in b
                if r["history"] == "zone_B"
            ],
            "disposition": {
                "effect_k96": de,
                "icc": icc,
                "revision_median": dk,
                "revised_fraction": float(np.mean([e for _, e in drev])),
                "meets_h6": bool(
                    de["mean"] > 0.10
                    and de["ci"][0] > 0
                    and icc >= 0.5
                    and dk <= 48
                    and np.mean([e for _, e in drev]) >= 0.8
                ),
            },
            "compute": ev["compute"][s],
        }
    # paired bootstrap comparisons vs B0
    rng = np.random.default_rng(0)

    def boot_diff(s, metric):
        d = []
        for _ in range(2000):
            samp = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            d.append(e1_from(pts[s], samp) - e1_from(pts["B0"], samp) if metric == "E1" else 0)
        return float(np.percentile(d, 5))

    for s in names:
        r, r0 = res["systems"][s], res["systems"]["B0"]
        e2_d = [a - b for a, b in zip(_e2_list(base[s]), _e2_list(base["B0"]), strict=True)]
        ce_d = np.array(r["CE_values"]) - np.array(r0["CE_values"])
        e7_d = np.array(r["E7_values"]) - np.array(r0["E7_values"])
        r["vs_B0"] = {
            "E1_diff_lower95": boot_diff(s, "E1") if s != "B0" else 0.0,
            "E2_diff": _mean_ci(e2_d),
            "CE_diff": _mean_ci(ce_d),
            "E7_diff": _mean_ci(e7_d),
        }
    integ, B0, L1 = res["systems"]["I"], res["systems"]["B0"], res["systems"]["A0_L1"]
    e2_lower = integ["vs_B0"]["E2_diff"]["ci"][0]
    better = {
        "revision": bool(
            integ["revision_median"] < B0["revision_median"] and integ["revised_fraction"] > B0["revised_fraction"]
        ),
        "context": bool(integ["vs_B0"]["E7_diff"]["ci"][0] > 0),
        "disposition": bool(integ["disposition"]["meets_h6"] and not B0["disposition"]["meets_h6"]),
    }
    res["rule1_I_replaces_B0"] = {
        "E1_noninferior": bool(integ["vs_B0"]["E1_diff_lower95"] > -0.05),
        "E2_noninferior": bool(e2_lower > -0.05),
        "better_on": better,
        "replaces": bool(integ["vs_B0"]["E1_diff_lower95"] > -0.05 and e2_lower > -0.05 and any(better.values())),
    }
    res["rule2_learned_mapping"] = {
        "CE_diff_ci": L1["vs_B0"]["CE_diff"]["ci"],
        "E1_not_lower": bool(L1["E1"] >= B0["E1"]),
        "E2_not_lower": bool(L1["E2"]["mean"] >= B0["E2"]["mean"]),
        "E4_not_slower": bool(not (L1["revision_median"] > B0["revision_median"])),
        "replaces_lookup": bool(
            L1["vs_B0"]["CE_diff"]["ci"][1] < 0
            and L1["E1"] >= B0["E1"]
            and L1["E2"]["mean"] >= B0["E2"]["mean"]
            and not (L1["revision_median"] > B0["revision_median"])
        ),
    }
    abl = res["systems"]
    res["rule3_ablations"] = {
        "context_necessary": bool(_ci_pos(abl["I"]["E7_values"], abl["I_minus_context"]["E7_values"])),
        "changepoint_necessary": bool(
            abl["I_minus_changepoint"]["revision_median"] > abl["I"]["revision_median"]
            or abl["I_minus_changepoint"]["revised_fraction"] < abl["I"]["revised_fraction"]
        ),
        "global_necessary": bool(
            abl["I"]["disposition"]["meets_h6"] and not abl["I_minus_global"]["disposition"]["meets_h6"]
        ),
        "memory_necessary": bool(
            abl["global_only"]["E1"] < abl["I"]["E1"] - 0.05
            and abl["global_only"]["E2"]["mean"] < abl["I"]["E2"]["mean"] - 0.05
        ),
    }
    phys_path = root / "X8" / "h8_physical.json"
    if phys_path.exists():
        prow = json.loads(phys_path.read_text(encoding="utf-8"))["rows"]
        ph = {}
        for s in PHYS_SYSTEMS:
            for pname in PROBES_PHYS:
                rs = [r for r in prow if r["system"] == s and r["probe"] == pname]
                if not rs:
                    continue
                X, y = _matrix(rs, NON_SPEED), [r["history"] for r in rs]
                idx = np.array([r["index"] for r in rs])
                corr = []
                for i in sorted(set(idx)):
                    t = idx == i
                    mdl = LDA().fit(X[~t], [yy for yy, tt in zip(y, t, strict=True) if not tt])
                    corr += [p == q for p, q in zip(mdl.predict(X[t]), np.array(y)[t], strict=True)]
                ph[f"{s}/{pname}"] = {
                    "n": len(rs),
                    "falls": sum(r["fell"] for r in rs),
                    "trips": sum(r["trips"] > 0 for r in rs),
                    "arrival": float(np.mean([r["arrived"] for r in rs])),
                    "history_classification": float(np.mean(corr)),
                }
        res["physical"] = ph
    for s in res["systems"].values():
        s.pop("CE_values")
        s.pop("E7_values")
    return res


def _e2_list(rows):
    from h34_analyze import _end, kappa_e

    by = {(r["history"], r["index"]): r for r in rows}
    return [
        kappa_e(_end(by[("risky_B", i)], "B1_plain")["intent"]) - kappa_e(_end(by[("safe", i)], "B1_plain")["intent"])
        for (h, i) in sorted(by)
        if h == "risky_B"
    ]


def _ci_pos(a, b) -> bool:
    from h34_analyze import _mean_ci

    return _mean_ci(np.array(a) - np.array(b))["ci"][0] > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "probes", "analyze"])
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    (a.root / "X8").mkdir(parents=True, exist_ok=True)
    if a.mode == "eval":
        import torch

        from apm_express.provenance import confirmatory_lock

        torch.set_num_threads(1)
        res = {"lock": confirmatory_lock(HERE / "prereg" / "H8_ADDENDUM.md"), **evaluate(a.root)}
        (a.root / "X8" / "h8_eval.json").write_text(json.dumps(res), encoding="utf-8")
    elif a.mode == "probes":
        res = probes_physical(a.root)
        (a.root / "X8" / "h8_physical.json").write_text(json.dumps(res), encoding="utf-8")
    else:
        res = analyze(a.root)
        (a.root / "X8" / "h8_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "systems"}, indent=1, default=float))


if __name__ == "__main__":
    main()
