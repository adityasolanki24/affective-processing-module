"""J2 analysis (MASTER_PREREGISTRATION §3).

    python j2_analyze.py calibrate --closed <J2-V closed> --policy <J2-V policy> --out configs/J2_calibration.json
    python j2_analyze.py test --closed <J2-X closed> --policy <J2-X policy> --calibration configs/J2_calibration.json
        --out results/J2/J2_X.json

**calibrate.** For each (condition, system) and each policy-group observer, from S3: the frozen belief calibration
(μ_careless, μ_careful, θ). It also performs the §3.3 selection: of EU_T, EA and EC, the model with the highest mean
dense predictive log-likelihood over all J2-V closed-loop lifetimes, a self-supervised objective.

**test.** Per (condition, system):
- adaptation (frozen-calibrated and decisive; S1, S2, S10);
- S3 AUROC and careful-misclassification. Identities with novel (erratic) behaviour are excluded and reported
  separately;
- normalised false blame (S5);
- avoidable collisions and completion vs A0 in the same condition;
- dense predictive log-likelihood;
- ECE of p at encounter start vs collision.

Then the §3.3 survival criteria for EU_I and for the selected model. The policy group gets belief metrics per
driver × memory. Finally the conditional-information analysis: plug-in I(Z; C), I(Z; C | action band) and
I(Z; C | context band) per driver, with a bootstrap over lifetimes. Z is the true disposition (evaluation only).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "I_autonomous_revision" / "scripts"))

import j1_metrics as M  # noqa: E402
from i_analyze import paired  # noqa: E402
from i_metrics import boot_mean, false_blame  # noqa: E402

from apm.learning.evaluation import auroc  # noqa: E402
from apm_select.stats import p_beyond, strip  # noqa: E402

CANDIDATES = ("EU_T", "EA", "EC")
FAMILIES_ADAPT = ("S1", "S2", "S10")
NI_COLLISION, NI_COMPLETION = 0.10, 0.02


def load(paths: list[Path]) -> dict:
    """data[condition][system][scenario] -> rows."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            data[r.get("condition", "nominal")][r["label"]][r["scenario"]].append(r)
    return data


def _excluded(r: dict) -> set:
    return set(r["roles"].get("erratic", []))


def calibration_ex(rows: list[dict]) -> dict:
    """j1_metrics.calibrate without erratic identities (they have no single disposition)."""
    clean = []
    for r in rows:
        ex = _excluded(r)
        if not ex:
            clean.append(r)
            continue
        clean.append({**r, "snapshots": [{k: v for k, v in s.items() if k not in ex} for s in r["snapshots"]],
                      "encounters": [e for e in r["encounters"] if e[2] not in ex]})
    return M.calibrate(clean)


def cmd_calibrate(a) -> None:
    closed, policy = load(a.closed), load(a.policy)
    cal: dict = {"closed": {}, "policy": {}}
    for cond, systems in closed.items():
        cal["closed"][cond] = {s: calibration_ex(by_sc["S3"]) for s, by_sc in systems.items()}
    for _cond, systems in policy.items():
        for s, by_sc in systems.items():
            cal["policy"][s] = calibration_ex(by_sc["S3"])
    logp = {s: float(np.mean([x for cond in closed.values() for r in cond.get(s, {}).get("S3", [])
                              for x in r.get("pred_logp", [])])) for s in ("EU_I", *CANDIDATES)}
    cal["selection"] = {"objective": "mean dense predictive log-likelihood, all J2-V closed-loop lifetimes",
                        "values": logp, "selected": max(CANDIDATES, key=lambda s: logp[s])}
    a.out.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print("selected", cal["selection"]["selected"], {k: round(v, 4) for k, v in logp.items()})


def disc_and_misclass(rows: list[dict], cal: dict) -> tuple[dict, dict]:
    aucs, mis = [], []
    sign = 1.0 if cal["gap"] >= 0 else -1.0
    for r in rows:
        ex = _excluded(r)
        snap, truth = r["snapshots"][-1], r["truth"][-1]
        ids = [k for k in snap if k in truth and k not in ex]
        if len({truth[k] for k in ids}) == 2:
            aucs.append(auroc([snap[k][0] for k in ids], [truth[k] == "careless" for k in ids]))
        careful = [k for k in ids if truth[k] == "careful"]
        if careful:
            mis.append(float(np.mean([sign * (snap[k][0] - cal["threshold"]) > 0 for k in careful])))
    return boot_mean(aucs), boot_mean(mis)


def ece(rows: list[dict]) -> float:
    bins = np.zeros((10, 3))
    for r in rows:
        for _t, _ident, p0, klass in r.get("p_log", []):
            if p0 is None:
                continue
            b = min(9, int(p0 * 10))
            bins[b] += (1, p0, 1.0 if klass == "collision" else 0.0)
    n = bins[:, 0].sum()
    return float(sum(abs(c[1] - c[2]) for c in bins if c[0]) / max(1.0, n))


def per_system(by_sc: dict, cal: dict) -> dict:
    s: dict = {}
    for sc in FAMILIES_ADAPT:
        rows = by_sc.get(sc, [])
        if rows:
            s[sc] = {"calibrated": M.adaptation(rows, cal), "decisive": M.adaptation(rows, cal, decisive=True)}
    if by_sc.get("S3"):
        s["auroc_S3"], s["careful_misclassification"] = disc_and_misclass(by_sc["S3"], cal)
        s["dense_logp"] = float(np.mean([x for r in by_sc["S3"] for x in r.get("pred_logp", [])] or [np.nan]))
        s["ece"] = ece(by_sc["S3"])
    if by_sc.get("S5"):
        gap = abs(cal["gap"])
        s["false_blame_normalised"] = boot_mean([None if x is None or gap < 1e-12 else x / gap
                                                 for x in (false_blame(r) for r in by_sc["S5"])])
    return s


def mutual_information(rows: list[dict], col: int | None, n_boot: int = 500) -> dict:
    """Plug-in I(Z; C | band) in nats (band column `col`, or unconditional), bootstrap over lifetimes."""
    def mi(sample: list[dict]) -> float:
        cnt: dict = defaultdict(float)
        for r in sample:
            for e in r["encounters"]:
                z, c = e[10], e[8]
                if z not in ("careless", "careful") or c is None or e[3] < 0.5:
                    continue
                b = (e[col] or 0) if col is not None else 0
                cnt[(b, z, c)] += 1
        total = sum(cnt.values())
        out = 0.0
        for b in {k[0] for k in cnt}:
            nb = sum(v for k, v in cnt.items() if k[0] == b)
            for (bb, z, c), v in cnt.items():
                if bb != b:
                    continue
                pz = sum(w for k, w in cnt.items() if k[0] == b and k[1] == z) / nb
                pc = sum(w for k, w in cnt.items() if k[0] == b and k[2] == c) / nb
                out += (v / total) * np.log((v / nb) / (pz * pc))
        return float(out)
    est = mi(rows)
    rng = np.random.default_rng(0)
    draws = [mi([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    return {"estimate": est, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def cmd_test(a) -> None:
    closed, policy = load(a.closed), load(a.policy)
    cal = json.loads(a.calibration.read_text())
    sel = cal["selection"]["selected"]
    res: dict = {"selected": sel, "closed": {}, "policy": {}, "survival": {}, "information": {}}
    for cond, systems in closed.items():
        res["closed"][cond] = {}
        allrows = {s: [r for rs in by_sc.values() for r in rs] for s, by_sc in systems.items()}
        for s, by_sc in systems.items():
            out = per_system(by_sc, cal["closed"][cond][s])
            if s != "A0":
                c = paired(allrows[s], allrows["A0"])
                out["vs_A0"] = {"avoidable_rel": c["avoidable_collision_rate"]["rel"],
                                "completion_abs": c["completion"]["abs"],
                                "p_ni_collision": p_beyond(c["avoidable_collision_rate"]["rel"], NI_COLLISION, "below"),
                                "p_ni_completion": p_beyond(c["completion"]["abs"], -NI_COMPLETION, "above")}
            res["closed"][cond][s] = out
    for name in ("EU_I", sel):
        per_cond = {}
        for cond, systems in res["closed"].items():
            s = systems[name]
            per_cond[cond] = {
                "auroc_ok": bool((s["auroc_S3"].get("estimate") or 0) >= 0.70),
                "collision_ni": bool(s["vs_A0"]["p_ni_collision"] < 0.05),
                "completion_ni": bool(s["vs_A0"]["p_ni_completion"] < 0.05),
                "misclass_ok": bool((s["careful_misclassification"].get("estimate") or 1) <= 0.20),
                "decisive_S1_S2": bool(all(s.get(sc, {}).get("decisive", {}).get("median") is not None
                                           and s[sc]["decisive"]["median"] <= 6 for sc in ("S1", "S2")))}
        every = all(v["auroc_ok"] and v["collision_ni"] and v["completion_ni"] and v["misclass_ok"]
                    for v in per_cond.values())
        share = float(np.mean([v["decisive_S1_S2"] for v in per_cond.values()]))
        res["survival"][name] = {"per_condition": per_cond, "every_condition_ok": every,
                                 "decisive_share": share, "promoted": bool(every and share >= 0.80),
                                 "failing": [c for c, v in per_cond.items() if not (v["auroc_ok"] and v["collision_ni"]
                                             and v["completion_ni"] and v["misclass_ok"])]}
    for _cond, systems in policy.items():
        for s, by_sc in systems.items():
            res["policy"][s] = per_system(by_sc, cal["policy"][s])
    groups = {"A0 (closed, nominal)": closed["nominal"]["EU_I"]["S3"]}
    for d in ("B0", "cautious", "R1"):
        key = f"{d}|EU_I"
        if key in next(iter(policy.values()), {}):
            groups[d] = next(iter(policy.values()))[key]["S3"]
    for g, rows in groups.items():
        res["information"][g] = {"I(Z;C)": mutual_information(rows, None),
                                 "I(Z;C|action)": mutual_information(rows, 13),
                                 "I(Z;C|context)": mutual_information(rows, 14)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(strip(res), indent=1, default=float), encoding="utf-8")
    for name, v in res["survival"].items():
        print(name, "promoted" if v["promoted"] else "not promoted", "failing:", v["failing"],
              "decisive share", round(v["decisive_share"], 2))
    for g, v in res["information"].items():
        print(g, {k: round(x["estimate"], 4) for k, x in v.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "test"))
    ap.add_argument("--closed", type=Path, nargs="+", required=True)
    ap.add_argument("--policy", type=Path, nargs="+", required=True)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cmd_calibrate(a) if a.mode == "calibrate" else cmd_test(a)


if __name__ == "__main__":
    main()
