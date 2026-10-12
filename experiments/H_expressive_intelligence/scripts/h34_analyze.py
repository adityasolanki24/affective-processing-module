"""H3/H4 analysis (prereg/H34_ADDENDUM.md §5, §7, §8). Numpy only.

    python scripts/h34_analyze.py --root results/H34    -> results/H34/X/h34_results.json

"Expressed caution" κ_e is measured from the **intent the robot expressed**, not from internal numbers. It is the
projection of the intent onto the relaxed → cautious direction, 0 = relaxed and 1 = cautious. This makes systems
with different internal variables (PAD, fixed styles, memories, GRUs) comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h1_analyze import LDA, NON_SPEED, _matrix  # noqa: E402

from apm_express.intent import ExpressionIntent  # noqa: E402
from apm_express.states import CAUTIOUS, RELAXED  # noqa: E402
from apm_select.survival import km  # noqa: E402

TRUTH_PROBES = ("B1_plain", "B1_marked", "A1_plain", "C1_plain", "B3_novel")
N_BOOT = 10_000
_R, _C = RELAXED.as_array(), CAUTIOUS.as_array()
_DIR = _C - _R


def kappa_e(intent: dict) -> float:
    return float(np.dot(ExpressionIntent(**intent).as_array() - _R, _DIR) / np.dot(_DIR, _DIR))


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    vals = np.asarray(x)[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _end(row, probe):
    return row["course"][-1]["probes"][probe]


def e1_points(rows) -> dict:
    """(history, index) -> list of (κ_e, truth) at lifetime end."""
    out = {}
    for r in rows:
        out[(r["history"], r["index"])] = [(kappa_e(_end(r, p)["intent"]), _end(r, p)["truth"]) for p in TRUTH_PROBES]
    return out


def e1_from(points: dict, keys) -> float:
    pts = [x for k in keys for x in points[k]]
    return spearman([p[0] for p in pts], [p[1] for p in pts])


def paired_diff(rows, probe, h1, h0, field="intent", reset=False) -> dict:
    by = {(r["history"], r["index"]): r for r in rows}
    diffs = []
    for (h, i), r in by.items():
        if h != h1 or (h0, i) not in by:
            continue
        if reset:
            a, b = r["reset_intents"][probe], by[(h0, i)]["reset_intents"][probe]
        else:
            a, b = _end(r, probe)[field], _end(by[(h0, i)], probe)[field]
        diffs.append(kappa_e(a) - kappa_e(b))
    return _mean_ci(diffs)


def _mean_ci(x) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(0)
    boot = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return {
        "mean": float(x.mean()),
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": int(len(x)),
    }


def crossing_times(rows, history, up: bool, safe_level: float, risky_level: float) -> list[tuple[float, bool]]:
    """Encounters after the switch at k = 24 until κ_e(B1/plain) crosses the midpoint (revision or acquisition)."""
    out = []
    for r in rows:
        if r["history"] != history:
            continue
        course = {c["k"]: kappa_e(c["probes"]["B1_plain"]["intent"]) for c in r["course"]}
        start = course.get(24, np.nan)
        target = (start + (safe_level if not up else risky_level)) / 2
        t, event = 24.0, False
        for k in sorted(c for c in course if c > 24):
            if (course[k] >= target) if up else (course[k] <= target):
                t, event = float(k - 24), True
                break
            t = float(k - 24)
        out.append((t, event))
    return out


def physical(root: Path) -> dict | None:
    path = root / "X" / "probes_physical.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    out = {}
    for system in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == system]
        X = _matrix(rs, NON_SPEED)
        y = [r["history"] for r in rs]
        idx = np.array([r["index"] for r in rs])
        correct = []
        for i in sorted(set(idx)):  # leave-one-pair-out
            test = idx == i
            if len(set(np.array(y)[~test])) < 2:
                continue
            model = LDA().fit(X[~test], [yy for yy, t in zip(y, test, strict=True) if not t])
            correct += [p == t for p, t in zip(model.predict(X[test]), np.array(y)[test], strict=True)]
        feats = {}
        for f in (
            "progress_speed",
            "pauses_per_m",
            "final_distance",
            "face_elev_rest",
            "face_yaw_rest_std",
            "straightness",
        ):
            d = [
                next(r for r in rs if r["index"] == i and r["history"] == "risky_B")["features"][f]
                - next(r for r in rs if r["index"] == i and r["history"] == "safe")["features"][f]
                for i in sorted(set(idx))
            ]
            feats[f] = _mean_ci(d)
        out[system] = {
            "n": len(rs),
            "falls": sum(r["fell"] for r in rs),
            "trips": sum(r["trips"] > 0 for r in rs),
            "arrival": float(np.mean([r["arrived"] for r in rs])),
            "history_classification": float(np.mean(correct)) if correct else float("nan"),
            "feature_difference_risky_minus_safe": feats,
        }
    return out


def analyze(root: Path) -> dict:
    ev = json.loads((root / "X" / "states_eval.json").read_text(encoding="utf-8"))
    rows = ev["rows"]
    systems = sorted({r["system"] for r in rows})
    base = {s: [r for r in rows if r["system"] == s and r["control"] == "none"] for s in systems}
    keys = sorted({(r["history"], r["index"]) for r in rows})
    points = {s: e1_points(base[s]) for s in systems}
    res: dict = {"systems": {}}
    for s in systems:
        b = base[s]
        safe_level = float(np.mean([kappa_e(_end(r, "B1_plain")["intent"]) for r in b if r["history"] == "safe"]))
        risky_level = float(np.mean([kappa_e(_end(r, "B1_plain")["intent"]) for r in b if r["history"] == "risky_B"]))
        rev = crossing_times(b, "change_B", False, safe_level, risky_level)
        acq = crossing_times(b, "late_B", True, safe_level, risky_level)
        ctrl = {
            c: [r for r in rows if r["system"] == s and r["control"] == c]
            for c in ("shuffled_outcomes", "random_identity")
        }
        zone = [
            kappa_e(_end(r, "B1_marked")["intent"]) - kappa_e(_end(r, "B1_plain")["intent"])
            for r in b
            if r["history"] == "zone_B"
        ]
        res["systems"][s] = {
            "E1_evidence_consistency": e1_from(points[s], keys),
            "E2_history_effect": paired_diff(b, "B1_plain", "risky_B", "safe"),
            "E3_controls": {
                "shuffled_outcomes": paired_diff(ctrl["shuffled_outcomes"], "B1_plain", "risky_B", "safe"),
                "random_identity": paired_diff(ctrl["random_identity"], "B1_plain", "risky_B", "safe"),
                "state_reset": paired_diff(b, "B1_plain", "risky_B", "safe", reset=True),
            },
            "E4_revision_km_median": km(np.array([t for t, _ in rev]), np.array([e for _, e in rev]), 24.0)[0],
            "E4_revised_fraction": float(np.mean([e for _, e in rev])) if rev else float("nan"),
            "E5_acquisition_km_median": km(np.array([t for t, _ in acq]), np.array([e for _, e in acq]), 24.0)[0],
            "E6_kind_generalisation": paired_diff(b, "B3_novel", "risky_B", "safe"),
            "E7_context": _mean_ci(zone),
            "E8_disposition_unknown_kind": paired_diff(b, "D1_unknown", "unpredictable", "safe"),
            "levels": {"safe": safe_level, "risky_B": risky_level},
        }
    # H3 selection: smallest system non-inferior to the best E1 (bootstrap over lifetimes)
    order = [
        "S0_none",
        "S1_fixed",
        "S2_scalar",
        "S3_small",
        "M_id",
        "M_ctx",
        "M_id_cp",
        "S4_pad",
        "S5_gru",
        "S6_fastslow",
        "S7_gru_mem",
    ]
    cands = [s for s in order if s in systems]
    e1 = {s: res["systems"][s]["E1_evidence_consistency"] for s in cands}
    best = max(cands, key=lambda s: e1[s])
    rng = np.random.default_rng(0)
    boots = {s: [] for s in cands}
    for _ in range(1000):  # 1,000 resamples (each needs a Spearman per system)
        sample = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals = {s: e1_from(points[s], sample) for s in cands}
        for s in cands:
            boots[s].append(vals[s] - vals[best])
    ni = {s: float(np.percentile(boots[s], 5)) for s in cands}
    selected = next(s for s in cands if ni[s] > -0.05)
    beats = {}
    for s in ("S4_pad", "S5_gru", "S6_fastslow", "S7_gru_mem", "M_ctx", "M_id_cp"):
        if s in cands and "M_id" in cands:
            diff = np.array(boots[s]) - np.array(boots["M_id"])
            r_s, r_m = res["systems"][s], res["systems"]["M_id"]
            beats[s] = {
                "E1_diff_ci_low": float(np.percentile(diff, 2.5)),
                "not_worse_E2_E6": r_s["E2_history_effect"]["mean"] >= r_m["E2_history_effect"]["mean"]
                and r_s["E6_kind_generalisation"]["mean"] >= r_m["E6_kind_generalisation"]["mean"],
                "not_worse_E4": not (r_s["E4_revision_km_median"] > r_m["E4_revision_km_median"]),
            }
            beats[s]["beats_M_id"] = bool(
                beats[s]["E1_diff_ci_low"] > 0 and beats[s]["not_worse_E2_E6"] and beats[s]["not_worse_E4"]
            )
    res["H3"] = {
        "E1": e1,
        "best": best,
        "ni_lower_vs_best": ni,
        "selected_smallest_noninferior": selected,
        "beats_A0_memory_baseline": beats,
    }
    phys = physical(root)
    res["physical"] = phys
    gate = {}
    for s in systems:
        r = res["systems"][s]
        p = (phys or {}).get(s)
        e2 = r["E2_history_effect"]
        sh = r["E3_controls"]["shuffled_outcomes"]
        gate[s] = {
            "E2_gt_0.20_ci_pos": bool(e2["mean"] > 0.20 and e2["ci"][0] > 0),
            "E1_ge_0.50": bool(r["E1_evidence_consistency"] >= 0.50),
            "shuffled_null": bool(sh["ci"][0] <= 0 <= sh["ci"][1] or abs(sh["mean"]) < 0.05),
            "reset_zero": bool(abs(r["E3_controls"]["state_reset"]["mean"]) < 1e-9),
            "physical": bool(p and p["history_classification"] >= 0.80 and p["falls"] == 0 and p["arrival"] >= 0.95),
        }
        gate[s]["supports_claim_C"] = all(gate[s].values())
    res["H4_gate"] = {
        "per_system": gate,
        "passes": any(v["supports_claim_C"] for v in gate.values()),
        "design_check_S0_S1_zero": all(
            abs(res["systems"][s]["E2_history_effect"]["mean"]) < 1e-9 for s in ("S0_none", "S1_fixed") if s in systems
        ),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    res = analyze(a.root)
    (a.root / "X" / "h34_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"H3": res["H3"], "H4_gate": res["H4_gate"]}, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
