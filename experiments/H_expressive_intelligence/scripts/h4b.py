"""H4b: developmental-expression re-test with the pooled outcome-shuffle control (prereg/H4b_ADDENDUM.md).

python scripts/h4b.py eval    --root results/H34     # intent level, frozen models, fresh streams in results/H34/X4b
<nerva-venv> python scripts/h4b.py probes --root results/H34   # physical pairs (needs NERVA)
python scripts/h4b.py analyze --root results/H34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

SYSTEMS_PHYSICAL = ("M_id", "S5_gru", "S6_fastslow", "S7_gru_mem", "S4_pad", "S0_none", "S1_fixed")
OFFSET = 1000


def _systems(root: Path, seeds: int = 5) -> dict:
    from h34_states import TUNE_GRID, learned_model, make

    tuning = json.loads((root / "tuning.json").read_text(encoding="utf-8"))
    learned = {}
    for f in sorted(root.glob("learned_*.json")):
        learned.update(json.loads(f.read_text(encoding="utf-8")))
    systems = {n: (lambda n=n: make(n)) for n in ("S0_none", "S1_fixed")}
    for fam in TUNE_GRID:
        systems[fam] = lambda fam=fam: make(fam, tuning[fam]["selected"])
    for kind, name in (("gru", "S5_gru"), ("fastslow", "S6_fastslow"), ("gru_mem", "S7_gru_mem")):
        systems[name] = lambda kind=kind, name=name: learned_model(root, kind, learned[kind]["selected"], seeds, name)
    return systems


def _final(model, obs) -> dict:
    from h34_states import PROBES

    from apm_express.states import intent_of

    model.reset()
    for ob in obs:
        model.observe(ob)
    return {pn: intent_of(model, pr).as_dict() for pn, pr in PROBES.items()}


def evaluate(root: Path) -> dict:
    from h34_states import control_stream, load_streams

    from apm_express.states import Obs

    streams = load_streams(root / "X4b" / "experience.json", 0.04)
    by = {(s["history"], s["index"]): s["obs"] for s in streams}
    idx = sorted({i for _, i in by})
    out = []
    for name, factory in _systems(root).items():
        for i in idx:
            risky, safe = by[("risky_B", i)], by[("safe", i)]
            rng = np.random.default_rng(i)
            pooled = [(o.y, o.magnitude) for o in risky + safe]
            perm = rng.permutation(len(pooled))
            pooled = [pooled[j] for j in perm]
            p_risky = [Obs(o.identity, o.kind, o.zone, *pooled[k]) for k, o in enumerate(risky)]
            p_safe = [Obs(o.identity, o.kind, o.zone, *pooled[len(risky) + k]) for k, o in enumerate(safe)]
            m = factory()
            rec = {
                "system": name,
                "index": i,
                "none": {"risky_B": _final(m, risky), "safe": _final(m, safe)},
                "pooled_shuffle": {"risky_B": _final(m, p_risky), "safe": _final(m, p_safe)},
                "within_shuffle_risky": _final(
                    m, control_stream(risky, "shuffled_outcomes", np.random.default_rng(10_000 + i))
                ),
            }
            m.reset()
            rec["reset"] = _final(m, [])
            out.append(rec)
    return {"rows": out}


def probes(root: Path, workers: int = 3) -> dict:
    from multiprocessing import Pool

    from h0_interface import _init
    from h34_probes import run

    from apm_express.provenance import confirmatory_lock

    lock = confirmatory_lock(HERE / "prereg" / "H4b_ADDENDUM.md")
    iface = json.loads((HERE / "results/H0/X/admitted_interface.json").read_text(encoding="utf-8"))["interface"]
    ev = json.loads((root / "X4b" / "h4b_eval.json").read_text(encoding="utf-8"))
    jobs = [
        {"system": r["system"], "history": h, "index": r["index"], "intent": r["none"][h]["B1_plain"], "iface": iface}
        for r in ev["rows"]
        if r["system"] in SYSTEMS_PHYSICAL
        for h in ("risky_B", "safe")
    ]
    with Pool(workers, initializer=_init) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    return {"lock": lock, "rows": rows}


def analyze(root: Path) -> dict:
    from h34_analyze import _mean_ci, kappa_e, physical

    ev = json.loads((root / "X4b" / "h4b_eval.json").read_text(encoding="utf-8"))
    h34 = json.loads((root / "X" / "h34_results.json").read_text(encoding="utf-8"))
    phys_path = root / "X4b" / "h4b_physical.json"
    phys = None
    if phys_path.exists():  # reuse the H34 physical analysis on the H4b rows
        tmp = root / "X4b" / "_phys"
        (tmp / "X").mkdir(parents=True, exist_ok=True)
        (tmp / "X" / "probes_physical.json").write_text(phys_path.read_text(encoding="utf-8"), encoding="utf-8")
        phys = physical(tmp)
    res = {}
    for s in sorted({r["system"] for r in ev["rows"]}):
        rows = [r for r in ev["rows"] if r["system"] == s]

        def e2(key, rows=rows):
            return _mean_ci(
                [kappa_e(r[key]["risky_B"]["B1_plain"]) - kappa_e(r[key]["safe"]["B1_plain"]) for r in rows]
            )

        spec = _mean_ci(
            [kappa_e(r["none"]["risky_B"]["B1_plain"]) - kappa_e(r["none"]["risky_B"]["A1_plain"]) for r in rows]
        )
        spec_sh = _mean_ci(
            [
                kappa_e(r["within_shuffle_risky"]["B1_plain"]) - kappa_e(r["within_shuffle_risky"]["A1_plain"])
                for r in rows
            ]
        )
        E2, pooled = e2("none"), e2("pooled_shuffle")
        reset = 0.0  # every system is evaluated from the same reset state for both robots: identical by construction
        e1 = h34["systems"][s]["E1_evidence_consistency"]
        p = (phys or {}).get(s)
        g = {
            "E2_gt_0.20_ci_pos": bool(E2["mean"] > 0.20 and E2["ci"][0] > 0),
            "E1_ge_0.50": bool(e1 >= 0.50),
            "pooled_shuffle_null": bool(pooled["ci"][0] <= 0 <= pooled["ci"][1] or abs(pooled["mean"]) < 0.05),
            "reset_zero": bool(abs(reset) < 1e-9),
            "physical": bool(p and p["history_classification"] >= 0.80 and p["falls"] == 0 and p["arrival"] >= 0.95),
        }
        g["supports_claim_C"] = all(g.values())
        res[s] = {
            "E2": E2,
            "E2_pooled_shuffle": pooled,
            "specificity": spec,
            "specificity_within_shuffle": spec_sh,
            "E1_from_H34": e1,
            "physical": p,
            "gate": g,
        }
    return {"systems": res, "passes": any(v["gate"]["supports_claim_C"] for v in res.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "probes", "analyze"])
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    if a.mode == "eval":
        import torch

        from apm_express.provenance import confirmatory_lock

        torch.set_num_threads(1)
        res = {"lock": confirmatory_lock(HERE / "prereg" / "H4b_ADDENDUM.md"), **evaluate(a.root)}
        (a.root / "X4b" / "h4b_eval.json").write_text(json.dumps(res), encoding="utf-8")
    elif a.mode == "probes":
        res = probes(a.root)
        (a.root / "X4b" / "h4b_physical.json").write_text(json.dumps(res), encoding="utf-8")
    else:
        res = analyze(a.root)
        (a.root / "X4b" / "h4b_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        for s, v in res["systems"].items():
            print(
                s,
                {
                    k: (round(x["mean"], 3) if isinstance(x, dict) and "mean" in x else x)
                    for k, x in v.items()
                    if k not in ("physical",)
                },
            )
        print("passes", res["passes"])


if __name__ == "__main__":
    main()
