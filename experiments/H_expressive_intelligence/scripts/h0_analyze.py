"""H0 analysis: interface admission and the H0 gate, exactly as fixed in prereg/H0_ADDENDUM.md.

    python scripts/h0_analyze.py admit --x results/H0/X        -> results/H0/X/admitted_interface.json
    python scripts/h0_analyze.py gate  --x results/H0/X        -> results/H0/X/gate.json
    python scripts/h0_analyze.py reproduce --x results/H0 --nerva <NERVA> -> results/H0/reproduction_check.json

Numpy only; runs in the APM venv.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

DWELLS = (0.4, 0.6, 1.0)
REST_RULES = {"horizontal_rms": 0.02, "yaw_rms": 0.15, "displacement": 0.10}  # NERVA's rest rules
JOINT_MARGIN_RAD = 0.02
SATURATION_STEPS = 25
PRIMARY = "neutral"
SECONDARY = "e1_prime"
D_ARRIVE_TOL = 0.15
HEAD_EPS = 0.02


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(rows) -> bool:
    return all(not r["fell"] and r["trips"] == 0 for r in rows)


def admit_dwell(switching: dict) -> dict:
    by = collections.defaultdict(list)
    for r in switching["rows"]:
        by[(r["trial"]["policy"], r["trial"]["params"]["dwell"])].append(r)
    out = {}
    for policy in (PRIMARY, SECONDARY):
        clean = {d: _clean(by[(policy, d)]) for d in DWELLS}
        admitted = [d for i, d in enumerate(DWELLS) if all(clean[e] for e in DWELLS[i:])]
        out[policy] = {
            "per_dwell": {
                str(d): {
                    "n": len(by[(policy, d)]),
                    "falls": sum(r["fell"] for r in by[(policy, d)]),
                    "trips": sum(r["trips"] > 0 for r in by[(policy, d)]),
                }
                for d in DWELLS
            },
            "min_dwell_s": min(admitted) if admitted else None,
        }
    return out


def _rest_point_ok(rows) -> bool:
    return bool(rows) and all(
        not r["fell"] and r["trips"] == 0 and all(r[k] <= v for k, v in REST_RULES.items()) for r in rows
    )


def admit_head(head: dict, policy: str) -> dict:
    rest = collections.defaultdict(list)
    walk = collections.defaultdict(list)
    for r in head["rows"]:
        t = r["trial"]
        if t["policy"] != policy:
            continue
        p = t["params"]
        key = (p.get("head_pitch", 0.0), p.get("head_yaw", 0.0), p.get("sweep", 0.0))
        (rest if p["command"] == "rest" else walk)[key].append(r)
    ok = {k: _rest_point_ok(v) for k, v in rest.items()}

    def box_ok(pmax: float, ymax: float) -> bool:
        """Every tested-grid point inside the box passes: axes, all corners, and yaw sweeps (amplitude <= ymax) at
        every pitch within the box. A grid point that was not run counts as a failure."""
        ps = [p for p in (-0.4, -0.2, 0.0, 0.2, 0.4) if abs(p) <= pmax + 1e-9]
        ys = [y for y in (-0.8, -0.4, 0.0, 0.4, 0.8) if abs(y) <= ymax + 1e-9]
        statics = [(p, y, 0.0) for p in ps for y in ys if (p, y) != (0.0, 0.0)]
        dynamics = [(p, 0.0, a) for p in ps for a in (0.4, 0.8) if a <= ymax + 1e-9]
        return all(ok.get(k, False) for k in statics + dynamics)

    boxes = [(p, y) for p in (0.0, 0.2, 0.4) for y in (0.0, 0.4, 0.8) if (p, y) != (0.0, 0.0) and box_ok(p, y)]
    pitch_max, yaw_max = max(boxes, key=lambda b: (b[1] / 0.8 + b[0] / 0.4, b[1])) if boxes else (0.0, 0.0)
    zero = walk.get((0.0, 0.0, 0.0), [])
    v0 = np.mean([r["mean_vfwd"] for r in zero]) if zero else float("nan")
    walk_ok = {}
    for k, rows in walk.items():
        if k == (0.0, 0.0, 0.0):
            continue
        ratio = np.mean([r["mean_vfwd"] for r in rows]) / v0 if v0 > 0.01 else float("nan")
        walk_ok[str(k)] = {
            "speed_ratio": float(ratio),
            "yaw_rms_max": float(max(r["yaw_rms"] for r in rows)),
            "falls": sum(r["fell"] for r in rows),
            "ok": bool(0.75 <= ratio <= 1.33 and all(r["yaw_rms"] <= 0.2 and not r["fell"] for r in rows)),
        }
    return {
        "rest_points": {str(k): {"ok": v, "n": len(rest[k])} for k, v in sorted(ok.items())},
        "yaw_max": yaw_max,
        "pitch_max": pitch_max,
        "admitted_boxes": [list(b) for b in boxes],
        "head_rate": 0.75,  # rad/s; passing sweeps peak at 0.75 (0.4 rad) and 1.5 rad/s (0.8 rad)
        "walking_points": walk_ok,
        "head_while_moving_admitted": bool(walk_ok) and all(v["ok"] for v in walk_ok.values()),
    }


def admit_pitch(pitch: dict) -> dict:
    rows = pitch["rows"]
    return {
        "n": len(rows),
        "falls": sum(r["fell"] for r in rows),
        "trips": sum(r["trips"] > 0 for r in rows),
        "admitted": _clean(rows),
    }


def admit(x: Path) -> dict:
    dwell = admit_dwell(_load(x / "switching.json"))
    head_doc = _load(x / "head.json")
    head = {p: admit_head(head_doc, p) for p in (PRIMARY, SECONDARY)}
    pitch = admit_pitch(_load(x / "pitch.json"))
    d = dwell[PRIMARY]["min_dwell_s"]
    hp = head[PRIMARY]
    interface = {
        "policy": PRIMARY,
        "min_dwell_s": d,
        "pitch": False,
        "head": hp["yaw_max"] > 0 or hp["pitch_max"] > 0,
        "head_pitch_range": [-hp["pitch_max"], hp["pitch_max"]],
        "head_yaw_range": [-hp["yaw_max"], hp["yaw_max"]],
        "head_rate": hp["head_rate"],
        "joint_margin_rad": JOINT_MARGIN_RAD,
        "saturation_steps": SATURATION_STEPS,
    }
    secondary = None
    d2 = dwell[SECONDARY]["min_dwell_s"]
    if d2 is not None and pitch["admitted"]:
        h2 = head[SECONDARY]
        secondary = {
            **interface,
            "policy": SECONDARY,
            "min_dwell_s": max(d, d2) if d else d2,
            "pitch": True,
            "pitch_rate": 1.0,
            "head": h2["yaw_max"] > 0 or h2["pitch_max"] > 0,
            "head_pitch_range": [-h2["pitch_max"], h2["pitch_max"]],
            "head_yaw_range": [-h2["yaw_max"], h2["yaw_max"]],
            "head_rate": h2["head_rate"],
        }
    return {
        "dwell": dwell,
        "head": head,
        "pitch": pitch,
        "interface": interface if d is not None else None,
        "secondary_interface": secondary,
    }


def _paired_fraction(rows, probe, feature, sign, reference="neutral") -> float:
    """Fraction of seed indices where probe's feature differs from the reference probe's in direction `sign`."""
    by = collections.defaultdict(dict)
    for r in rows:
        name = r["trial"]["name"]
        pr, s = r["trial"]["params"]["probe"], name.rsplit("_", 1)[1]
        by[s][pr] = r["features"][feature]
    pairs = [(v[probe], v[reference]) for v in by.values() if probe in v and reference in v]
    return float(np.mean([sign * (a - b) > 0 for a, b in pairs])) if pairs else 0.0


def gate(x: Path) -> dict:
    adm = _load(x / "admitted_interface.json")
    fb = _load(x / "fallback.json")["rows"]
    nom_all = _load(x / "nominal.json")["rows"]
    nominal = {p: [r for r in nom_all if r["trial"]["policy"] == p] for p in (PRIMARY, SECONDARY)}
    gate_reproduction = _load(x.parent / "reproduction_check.json")

    # (b) dwell
    b = adm["interface"] is not None and adm["interface"]["min_dwell_s"] is not None
    # (c) channel families realised in the nominal suite (primary): >= 95% of paired seeds in the designed direction
    rows = nominal[PRIMARY]
    fam = {
        "temporal (pauses)": _paired_fraction(rows, "slow_pausing", "pauses_per_m", +1),
        "spatial: path": _paired_fraction(rows, "indirect_scanning", "straightness", -1),
        "spatial: proximity": _paired_fraction(rows, "lean_forward_close", "final_distance", -1),
        "spatial: orientation": _paired_fraction(rows, "averted_lean_back", "final_bearing_abs", +1),
    }
    if adm["interface"] and adm["interface"]["head"]:
        fam["posture: head scanning"] = _paired_fraction(rows, "indirect_scanning", "face_yaw_rest_std", +1)
        if adm["interface"]["head_pitch_range"][1] > 0:
            fam["posture: head elevation"] = _paired_fraction(
                rows, "lean_forward_close", "face_elev_rest", +1, reference="averted_lean_back"
            )
    realised = {k: v >= 0.95 for k, v in fam.items()}
    families = {
        "temporal": realised["temporal (pauses)"],
        "spatial": any(v for k, v in realised.items() if k.startswith("spatial")),
        "posture": any(v for k, v in realised.items() if k.startswith("posture")),
    }
    c = sum(families.values()) >= 2
    # (d) monitor and fallback
    by = collections.defaultdict(dict)
    for r in fb:
        p = r["trial"]["params"]
        by[(p["push"], p["push_deg"], r["trial"]["name"].rsplit("_", 1)[1])][p["enforce"]] = r
    low = [v[True] for k, v in by.items() if k[0] == 0.3]
    high = [v for k, v in by.items() if k[0] > 0.3]
    enf_only = sum(v[True]["fell"] and not v[False]["fell"] for v in high)
    shadow_only = sum(v[False]["fell"] and not v[True]["fell"] for v in high)
    neutral_ok = all(v[True]["nonneutral_during_fallback"] == 0 for v in by.values())
    d = {
        "falls_at_0.3_enforced": sum(r["fell"] for r in low),
        "n_0.3": len(low),
        "falls_high_enforced": sum(v[True]["fell"] for v in high),
        "falls_high_shadow": sum(v[False]["fell"] for v in high),
        "n_high_pairs": len(high),
        "discordant_enforced_only": enf_only,
        "discordant_shadow_only": shadow_only,
        "fallback_always_neutral": neutral_ok,
    }
    d_pass = d["falls_at_0.3_enforced"] == 0 and enf_only <= shadow_only and neutral_ok

    # (e) nominal
    def nominal_summary(rs):
        per = collections.defaultdict(list)
        for r in rs:
            per[r["trial"]["params"]["probe"]].append(r)
        return {
            "falls": sum(r["fell"] for r in rs),
            "trips": sum(r["trips"] > 0 for r in rs),
            "n": len(rs),
            "arrival_rate": {k: float(np.mean([r["arrived"] for r in v])) for k, v in per.items()},
            "head_while_moving_max": max((r["head_while_moving_max"] for r in rs), default=0.0),
        }

    e_sum = nominal_summary(rows)
    e = (
        e_sum["falls"] == 0
        and e_sum["trips"] == 0
        and all(v >= 0.95 for v in e_sum["arrival_rate"].values())
        and e_sum["head_while_moving_max"] <= HEAD_EPS + 1e-9
    )
    sec = nominal_summary(nominal[SECONDARY]) if nominal[SECONDARY] else None
    sec_ok = (
        adm["secondary_interface"] is not None
        and sec is not None
        and sec["falls"] == 0
        and sec["trips"] == 0
        and all(v >= 0.95 for v in sec["arrival_rate"].values())
    )
    a = bool(gate_reproduction.get("all_reproduced"))
    out = {
        "a_reproduction": a,
        "b_dwell_admitted": b,
        "c_channel_families": {"fractions": fam, "families": families, "pass": c},
        "d_monitor_fallback": {**d, "pass": d_pass},
        "e_nominal": {**e_sum, "pass": e},
        "secondary_e1_prime": {"nominal": sec, "admitted_for_h1": sec_ok},
        "gate_passes": bool(a and b and c and d_pass and e),
    }
    return out


def push_star(rest: dict, policy: str = PRIMARY) -> dict:
    """Largest standing push magnitude with no fall in any direction, at it and at every smaller magnitude."""
    by = collections.defaultdict(list)
    for r in rest["rows"]:
        if r["policy"] == policy:
            by[r["magnitude"]].append(r["fell"])
    star = 0.0
    for m in sorted(by):
        if any(by[m]):
            break
        star = m
    return {
        "falls_by_magnitude": {str(m): int(sum(v)) for m, v in sorted(by.items())},
        "n_by_magnitude": {str(m): len(v) for m, v in sorted(by.items())},
        "push_star": star,
    }


def gate_b(xb: Path, x: Path) -> dict:
    """H0b gate (prereg/H0b_ADDENDUM.md): (a), (b) carried from H0; (c), (d'), (e') on the H0b runs."""
    h0 = _load(x / "gate.json")
    rest = push_star(_load(xb / "rest_push.json"))
    rest_secondary = push_star(_load(xb / "rest_push.json"), SECONDARY)
    fb = _load(xb / "fallback_b.json")["rows"]
    nom_all = _load(xb / "nominal.json")["rows"]
    rows = [r for r in nom_all if r["trial"]["policy"] == PRIMARY]
    fam = {
        "temporal (pauses)": _paired_fraction(rows, "slow_pausing", "pauses_per_m", +1),
        "spatial: path": _paired_fraction(rows, "indirect_scanning", "straightness", -1),
        "spatial: proximity": _paired_fraction(rows, "lean_forward_close", "final_distance", -1),
        "spatial: orientation": _paired_fraction(rows, "averted_lean_back", "final_bearing_abs", +1),
        "posture: head scanning": _paired_fraction(rows, "indirect_scanning", "face_yaw_rest_std", +1),
        "posture: head elevation": _paired_fraction(
            rows, "lean_forward_close", "face_elev_rest", +1, reference="averted_lean_back"
        ),
    }
    realised = {k: v >= 0.95 for k, v in fam.items()}
    families = {
        "temporal": realised["temporal (pauses)"],
        "spatial": any(v for k, v in realised.items() if k.startswith("spatial")),
        "posture": any(v for k, v in realised.items() if k.startswith("posture")),
    }
    c = sum(families.values()) >= 2

    def cell(push, probe, enforce):
        return [
            r
            for r in fb
            if r["trial"]["params"]["push"] == push
            and r["trial"]["params"]["probe"] == probe
            and r["trial"]["params"]["enforce"] == enforce
        ]

    star = rest["push_star"]
    at_star = cell(star, "slow_pausing", True)
    enf, sh = cell(0.3, "slow_pausing", True), cell(0.3, "slow_pausing", False)
    pairs = {(r["trial"]["params"]["push_deg"], r["trial"]["name"].rsplit("_", 1)[1]): r for r in sh}
    enf_only = sum(
        r["fell"] and not pairs[(r["trial"]["params"]["push_deg"], r["trial"]["name"].rsplit("_", 1)[1])]["fell"]
        for r in enf
    )
    sh_only = sum(
        not r["fell"] and pairs[(r["trial"]["params"]["push_deg"], r["trial"]["name"].rsplit("_", 1)[1])]["fell"]
        for r in enf
    )
    neutral_probe = cell(0.3, "neutral", True)
    d = {
        "rest_push": rest,
        "rest_push_secondary": rest_secondary,
        "falls_at_push_star_enforced": sum(r["fell"] for r in at_star),
        "n_push_star": len(at_star),
        "falls_0.3_expressive_enforced": sum(r["fell"] for r in enf),
        "falls_0.3_expressive_shadow": sum(r["fell"] for r in sh),
        "falls_0.3_neutral_probe_enforced": sum(r["fell"] for r in neutral_probe),
        "n_0.3_each": len(enf),
        "discordant_enforced_only": enf_only,
        "discordant_shadow_only": sh_only,
        "fallback_always_neutral": all(
            r["nonneutral_during_fallback"] == 0 for r in fb if r["trial"]["params"]["enforce"]
        ),
    }
    d_pass = (
        star >= 0.10 and d["falls_at_push_star_enforced"] == 0 and enf_only <= sh_only and d["fallback_always_neutral"]
    )
    per = collections.defaultdict(list)
    for r in rows:
        per[r["trial"]["params"]["probe"]].append(r)
    e_sum = {
        "falls": sum(r["fell"] for r in rows),
        "trips": sum(r["trips"] > 0 for r in rows),
        "n": len(rows),
        "arrival_rate": {k: float(np.mean([r["arrived"] for r in v])) for k, v in per.items()},
        "head_while_moving_max": max((r["head_while_moving_max"] for r in rows), default=0.0),
    }
    e = (
        e_sum["falls"] == 0
        and e_sum["trips"] == 0
        and all(v >= 0.95 for v in e_sum["arrival_rate"].values())
        and e_sum["head_while_moving_max"] <= HEAD_EPS + 1e-9
    )
    sec_rows = [r for r in nom_all if r["trial"]["policy"] == SECONDARY]
    sec = (
        {
            "falls": sum(r["fell"] for r in sec_rows),
            "trips": sum(r["trips"] > 0 for r in sec_rows),
            "arrival_rate": {
                k: float(np.mean([r["arrived"] for r in sec_rows if r["trial"]["params"]["probe"] == k])) for k in per
            },
        }
        if sec_rows
        else None
    )
    sec_ok = bool(
        sec and sec["falls"] == 0 and sec["trips"] == 0 and all(v >= 0.95 for v in sec["arrival_rate"].values())
    )
    return {
        "a_reproduction": h0["a_reproduction"],
        "b_dwell_admitted": h0["b_dwell_admitted"],
        "c_channel_families": {"fractions": fam, "families": families, "pass": c},
        "d_standing_push_and_fallback": {**d, "pass": d_pass},
        "e_nominal": {**e_sum, "pass": e},
        "secondary_e1_prime": {"nominal": sec, "admitted_for_h1": sec_ok},
        "gate_passes": bool(h0["a_reproduction"] and h0["b_dwell_admitted"] and c and d_pass and e),
    }


def _compare(ours: Path, theirs: Path, summary: bool = True) -> dict:
    ta, tb = _load(ours / "trials.json"), _load(theirs / "trials.json")
    same = sum(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(ta, tb, strict=False))
    out = {"trials_identical": same, "n_trials": [len(ta), len(tb)]}
    if summary:
        sa, sb = _load(ours / "summary.json"), _load(theirs / "summary.json")
        timing = {"wall_seconds"}  # wall-clock run time, not a result (DEVIATIONS 2026-10-12)
        sa = {k: v for k, v in sa.items() if k not in timing}
        sb = {k: v for k, v in sb.items() if k not in timing}
        out["summary_identical"] = json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
        out["ours_passes"] = sa.get("gate_passes", sa.get("passes"))
    out["identical"] = same == len(ta) == len(tb) and out.get("summary_identical", True)
    return out


def reproduce(h0: Path, nerva: Path) -> dict:
    """NERVA's gate (both candidates) and E1' style evaluation, reproduced with NERVA's own runners."""
    lc = nerva / "experiments" / "locomotion_curriculum"
    res = {
        "gate_turn_translation": _compare(
            h0 / "reproduce_gate_turn_translation", lc / "results_turn_translation" / "gate"
        ),
        "gate_e1_prime": _compare(h0 / "reproduce_gate_e1_prime", lc / "results_e1prime" / "gate"),
        "e1prime_style": _compare(h0 / "reproduce_e1prime_style", lc / "results_e1prime" / "style"),
    }
    res["all_reproduced"] = all(v["identical"] for v in res.values())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["admit", "gate", "reproduce", "gate_b", "push_star"])
    ap.add_argument("--xb", type=Path, help="H0b results directory")
    ap.add_argument("--x", type=Path)
    ap.add_argument("--nerva", type=Path)
    args = ap.parse_args()
    if args.mode == "admit":
        res = admit(args.x)
        path = args.x / "admitted_interface.json"
    elif args.mode == "gate":
        res = gate(args.x)
        path = args.x / "gate.json"
    elif args.mode == "gate_b":
        res = gate_b(args.xb, args.x)
        path = args.xb / "gate_b.json"
    elif args.mode == "push_star":
        res = push_star(_load(args.xb / "rest_push.json"))
        path = args.xb / "push_star.json"
    else:
        res = reproduce(args.x, args.nerva)
        path = args.x / "reproduction_check.json"
    path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
