"""J1: programme I's I1 re-evaluated on new test seeds, with calibration frozen beforehand (MASTER_PREREGISTRATION §J1).

    python j1_analyze.py calibrate --runs <J1-V lifetimes> --bank <bank.json> --out configs/J1_calibration.json
    python j1_analyze.py test --runs <J1-X lifetimes> --calibration configs/J1_calibration.json --bank <bank.json>
        --out results/J1/J1_X.json

Per system on the test lifetimes:
- adaptation in S1, S2 and S10: frozen-calibrated, decisive, decision read-out, and observed decisions;
- programme I's original test-calibrated form, for comparison only;
- belief error, collision log loss, and dense predictive log-likelihood;
- S3 discrimination; normalised false blame (frozen gap);
- safety and completion vs A0 (G's paired bootstrap).

Then the preregistered criteria.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "I_autonomous_revision" / "scripts"))

import j1_metrics as M  # noqa: E402
from i_analyze import paired  # noqa: E402
from i_metrics import adaptation_cal, boot_mean, calibration, false_blame, load, ranking_auroc  # noqa: E402

from apm_select.stats import holm, p_beyond, strip  # noqa: E402
from apm_select.survival import km_bootstrap  # noqa: E402

ADAPT = ("S1", "S2", "S10")
DECAY_CAL_FROM = "R_dense"  # the decay-only control is R_dense until the switch: it uses R_dense's calibration
NI_COLLISION, NI_COMPLETION = 0.10, 0.02
_RO = {}


def _readout(bank_path: str):
    if "ro" not in _RO:
        from j_common import A0, init
        init()
        import g_common

        from apm_transfer.readout import DecisionReadout
        b = json.loads(Path(bank_path).read_text())
        _RO["ro"] = DecisionReadout(g_common._STATE["predictor"], A0["lam_c"], A0["lam_s"], b["features"],
                                    b["multipliers"])
    return _RO["ro"]


def readout_calibration(args: tuple) -> tuple[str, dict]:
    name, rows, bank = args
    ro = _readout(bank)
    car, caf = [], []
    for r in rows:
        for snap, truth in zip(r["snapshots"][M.CAL_FROM:], r["truth"][M.CAL_FROM:], strict=True):
            for k, v in snap.items():
                (car if truth.get(k) == "careless" else caf).append((v[0], v[1]))
    rng = np.random.default_rng(0)
    car = [car[i] for i in rng.choice(len(car), min(600, len(car)), replace=False)]
    caf = [caf[i] for i in rng.choice(len(caf), min(600, len(caf)), replace=False)]
    sc, sf = float(np.mean(ro.careful_shares(car))), float(np.mean(ro.careful_shares(caf)))
    return name, {"ro_careless": sc, "ro_careful": sf, "ro_threshold": (sc + sf) / 2, "ro_gap": sc - sf}


def readout_adaptation(args: tuple) -> tuple[str, dict]:
    """Kaplan–Meier adaptation of the decision read-out of switched identities (end-of-episode beliefs)."""
    name, by_sc, cal, bank = args
    ro = _readout(bank)
    out = {}
    for sc in ADAPT:
        groups, curve = [], np.zeros(M.EPISODES)
        count = np.zeros(M.EPISODES)
        for r in by_sc.get(sc, []):
            to = r["roles"]["to"]
            sign = 1.0 if cal["ro_gap"] >= 0 else -1.0
            items = []
            for ident in r["roles"]["switched"]:
                eps = [e for e in range(M.CAL_FROM, M.EPISODES) if ident in r["snapshots"][e]]
                shares = dict(zip(eps, ro.careful_shares([tuple(r["snapshots"][e][ident]) for e in eps]),
                                  strict=True))
                for e, s in shares.items():
                    curve[e] += s
                    count[e] += 1

                def careless_side(e, shares=shares, sign=sign):
                    return None if e not in shares else sign * (shares[e] - cal["ro_threshold"]) > 0
                pre = careless_side(M.SWITCH - 1)
                if pre is None or (pre if to == "careless" else not pre):
                    continue
                t = next((e - M.SWITCH + 1 for e in range(M.SWITCH, M.EPISODES)
                          if careless_side(e) is not None and (careless_side(e) if to == "careless"
                                                              else not careless_side(e))), None)
                items.append((float(t), True) if t is not None else (float(M.HORIZON), False))
            groups.append(items)
        res = km_bootstrap(groups, M.HORIZON, 2000)
        res["established"] = sum(len(g) for g in groups)
        res["curve"] = (curve / np.maximum(1, count)).tolist()
        out[sc] = res
    return name, out


def cmd_calibrate(a) -> None:
    data = load(a.runs)
    cal = {name: M.calibrate(by_sc["S3"]) for name, by_sc in data.items()}
    with Pool(min(8, len(data))) as pool:
        for name, ro in pool.map(readout_calibration, [(n, data[n]["S3"], str(a.bank)) for n in data]):
            cal[name] |= ro
    if DECAY_CAL_FROM in cal:
        for name in data:
            if name.startswith("R_decay"):
                cal[name] = {**cal[DECAY_CAL_FROM], "note": f"copied from {DECAY_CAL_FROM} (§J1)"}
    a.out.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    for n, c in cal.items():
        print(f"{n:9} theta {c['threshold']:.4f} gap {c['gap']:+.4f} ro {c['ro_careless']:.3f}/{c['ro_careful']:.3f} "
              f"obs-dec {c['dec_careless']:.3f}/{c['dec_careful']:.3f}")


def adaptation_ok(x: dict | None) -> bool:
    return bool(x and x.get("median") is not None and x["median"] <= 6 and x["median_ci95"][1] <= 10)


def cmd_test(a) -> None:
    data = load(a.runs)
    cal = json.loads(a.calibration.read_text())
    res: dict = {"systems": {}, "comparisons": {}}
    with Pool(min(8, len(data))) as pool:
        ro_res = dict(pool.map(readout_adaptation, [(n, {sc: data[n].get(sc, []) for sc in ADAPT}, cal[n],
                                                      str(a.bank)) for n in data]))
    for name, by_sc in data.items():
        c = cal[name]
        s: dict = {"calibration": c}
        for sc in ADAPT:
            rows = by_sc.get(sc, [])
            s[sc] = {"calibrated": M.adaptation(rows, c), "decisive": M.adaptation(rows, c, decisive=True),
                     "decision_readout": ro_res[name][sc], "decision_observed": M.decision_adaptation(rows, c),
                     "test_calibrated_programme_I": adaptation_cal(rows, calibration(by_sc["S3"])),
                     "belief_error": M.belief_error(rows, c), "collision_logloss": M.collision_logloss(rows),
                     "dense_logp": M.dense_logp(rows)}
        s["discrimination_S3"] = boot_mean([ranking_auroc(r) for r in by_sc["S3"]])
        gap = abs(c["gap"])
        s["false_blame_normalised"] = boot_mean([None if x is None or gap < 1e-12 else x / gap
                                                 for x in (false_blame(r) for r in by_sc["S5"])])
        res["systems"][name] = s
    allrows = {n: [r for rs in by_sc.values() for r in rs] for n, by_sc in data.items()}
    for name in data:
        if name != "A0":
            res["comparisons"][f"{name}_vs_A0"] = paired(allrows[name], allrows["A0"])

    def criteria(name: str) -> dict:
        s, cmp_ = res["systems"][name], res["comparisons"][f"{name}_vs_A0"]
        p3 = p_beyond(cmp_["avoidable_collision_rate"]["rel"], NI_COLLISION, "below")
        p4 = p_beyond(cmp_["completion"]["abs"], -NI_COMPLETION, "above")
        adj = holm({"C3": p3, "C4": p4})
        disc, fb = s["discrimination_S3"], s["false_blame_normalised"]
        out = {"C1_frozen_calibrated": {sc: adaptation_ok(s[sc]["calibrated"]) for sc in ADAPT},
               "C1b_decisive": {sc: adaptation_ok(s[sc]["decisive"]) for sc in ADAPT},
               "C2_discrimination": bool(disc.get("estimate") is not None and disc["estimate"] >= 0.80
                                         and disc["ci95"][0] >= 0.70),
               "C3_safety_NI": {"p_holm": adj["C3"], "pass": bool(adj["C3"] < 0.05)},
               "C4_completion_NI": {"p_holm": adj["C4"], "pass": bool(adj["C4"] < 0.05)},
               "C5_false_blame": bool(fb.get("estimate") is not None and abs(fb["estimate"]) < 0.25
                                      and max(abs(fb["ci95"][0]), abs(fb["ci95"][1])) < 0.5)}
        out["pass"] = bool(all(out["C1_frozen_calibrated"].values()) and all(out["C1b_decisive"].values())
                           and out["C2_discrimination"] and out["C3_safety_NI"]["pass"]
                           and out["C4_completion_NI"]["pass"] and out["C5_false_blame"])
        return out
    res["criteria"] = {n: criteria(n) for n in data if n not in ("A0", "Oracle")}
    decay = [n for n in data if n.startswith("R_decay")]
    res["validity_decay_control_fails_decisive"] = {
        n: {sc: not adaptation_ok(res["systems"][n][sc]["decisive"]) for sc in ADAPT} for n in decay}
    valid = all(all(v.values()) for v in res["validity_decay_control_fails_decisive"].values())
    res["J1_verdict"] = ("I1 survives" if valid and res["criteria"]["R_dense"]["pass"] else
                         "metric invalid (decay-only control adapts)" if not valid else "I1 does not survive")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(strip(res), indent=1, default=float), encoding="utf-8")
    print(res["J1_verdict"])
    for n, s in res["systems"].items():
        print(f"{n:9}", " | ".join(
            f"{sc}: cal {s[sc]['calibrated']['median']} dec {s[sc]['decisive']['median']} "
            f"ro {s[sc]['decision_readout']['median']} err {s[sc]['belief_error']['post_mean']:.2f}" for sc in ADAPT),
            f"| AUROC {s['discrimination_S3']['estimate']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "test"))
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cmd_calibrate(a) if a.mode == "calibrate" else cmd_test(a)


if __name__ == "__main__":
    main()
