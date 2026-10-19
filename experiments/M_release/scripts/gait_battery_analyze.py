"""Analyse the T1 gait battery and apply the frozen promotion rule (REMEDIATION_PROTOCOL.md).

    python experiments/M_release/scripts/gait_battery_analyze.py --runs <dirs with *.jsonl> --out gait_battery.json

Per run, from the recorded physics at 50 Hz:
- tracking: forward speed along the torso heading (sway is not progress) and yaw rate, in each commanded
  segment after a 2 s transient;
- stance slip: horizontal speed of a foot in contact (> 5 N) at both ends of a sample;
- swing clearance p95, stride time, left/right stride asymmetry;
- tilt p95, joint acceleration RMS, torque proxy (mean Σ τ²);
- stop distance and settle time after each walk → stop transition (speed below 0.02 m/s);
- peak tilt in the 2 s after each push.
Pairs are matched on (environment, scenario, seed).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = {"speed": "speed_", "turn": "turn_", "startstop": "startstop", "push": "push_"}


def load(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0])["meta"], [json.loads(x) for x in lines[1:]]


def heading_axes(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat.T
    return np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)], axis=-1)


def metrics(meta: dict, rows: list[dict]) -> dict:
    t = np.array([r["t"] for r in rows])
    out: dict = {"fell": meta["fell"], "fall_t": meta["fall_t"], "observed_s": float(t[-1]) if len(t) else 0.0}
    if len(rows) < 10:
        return out
    dt = np.diff(t)
    cmd = np.array([r["cmd"] for r in rows])
    quat = np.array([r["quat"] for r in rows])
    lin = np.array([r["lin"] for r in rows])
    ang = np.array([r["ang"] for r in rows])
    fwd = np.sum(lin * heading_axes(quat), axis=1)
    feet = np.array([r["feet"] for r in rows])
    force = np.linalg.norm(np.array([r["force"] for r in rows]), axis=2)
    contact = force > 5.0
    foot_v = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) / dt[:, None]
    stance = contact[1:] & contact[:-1]
    out["stance_slip_m_s"] = float(foot_v[stance].mean()) if stance.any() else None
    clear, strides = [], []
    for f in range(2):
        z = feet[:, f, 2]
        if contact[:, f].any() and (~contact[:, f]).any():
            clear.append(float(np.percentile(z[~contact[:, f]] - np.median(z[contact[:, f]]), 95)))
        down = t[1:][contact[1:, f] & ~contact[:-1, f]]
        iv = np.diff(down)
        iv = iv[(iv > 0.2) & (iv < 3.0)]
        strides.append(float(np.median(iv)) if len(iv) else None)
    out["swing_clearance_m"] = float(np.mean(clear)) if clear else None
    out["stride_s"] = float(np.mean([s for s in strides if s])) if any(strides) else None
    out["stride_asymmetry_s"] = abs(strides[0] - strides[1]) if all(strides) else None
    out["tilt_p95_deg"] = float(np.percentile([r["tilt"] for r in rows], 95))
    dq = np.array([r["dq"] for r in rows])
    out["joint_accel_rms"] = float(np.sqrt(np.mean((np.diff(dq, axis=0) / dt[:, None]) ** 2)))
    eff = np.array([r["effort"] for r in rows], float)
    out["torque_proxy"] = float(np.nanmean(np.nansum(eff ** 2, axis=1))) if np.isfinite(eff).any() else None
    # tracking per commanded segment (after a 2 s transient)
    v_err, w_err = [], []
    for (v, w) in {tuple(c) for c in cmd.tolist()}:
        if v == 0 and w == 0:
            continue
        on = np.all(cmd == [v, w], axis=1)
        idx = np.flatnonzero(on)
        if len(idx) == 0:
            continue
        steady = on & (t >= t[idx[0]] + 2.0)
        if steady.sum() < 25:
            continue
        v_err.append(abs(float(fwd[steady].mean()) - v))
        if w != 0:
            w_err.append(abs(float(ang[steady, 2].mean()) - w))
    out["speed_error"] = float(np.mean(v_err)) if v_err else None
    out["yaw_error"] = float(np.mean(w_err)) if w_err else None
    # stops: walk -> zero command
    root = np.array([r["root"] for r in rows])[:, :2]
    speed = np.linalg.norm(lin[:, :2], axis=1)
    moving_cmd = np.any(cmd != 0, axis=1)
    stop_d, settle = [], []
    for i in np.flatnonzero(moving_cmd[:-1] & ~moving_cmd[1:]) + 1:
        j = i
        while j < len(t) and not moving_cmd[j] and speed[j] > 0.02:
            j += 1
        if j < len(t) and not moving_cmd[j]:
            stop_d.append(float(np.linalg.norm(root[j] - root[i])))
            settle.append(float(t[j] - t[i]))
    out["stop_distance_m"] = float(np.mean(stop_d)) if stop_d else None
    out["settle_s"] = float(np.mean(settle)) if settle else None
    peaks = []
    for i, r in enumerate(rows):
        if r.get("push"):
            win = (t >= t[i]) & (t <= t[i] + 2.0)
            peaks.append(float(np.max(np.array([x["tilt"] for x in rows])[win])))
    out["push_peak_tilt_deg"] = float(np.mean(peaks)) if peaks else None
    return out


def family(scenario: str) -> str:
    return next(k for k, p in FAMILIES.items() if scenario.startswith(p))


def ratio_median(pairs: list[tuple[float, float]]) -> float | None:
    r = [c / b for b, c in pairs if b and c is not None and b is not None and b > 0]
    return float(np.median(r)) if r else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs: dict[tuple, dict] = {}
    for d in a.runs:
        for f in sorted(d.rglob("*_7200[0-9].jsonl")):
            meta, rows = load(f)
            runs[(meta["env"], meta["scenario"], meta["seed"], meta["controller"])] = metrics(meta, rows)
    keys = sorted({k[:3] for k in runs})
    pairs = [(k, runs.get((*k, "v1")), runs.get((*k, "candidate"))) for k in keys]
    pairs = [(k, b, c) for k, b, c in pairs if b is not None and c is not None]
    res: dict = {"runs": len(runs), "pairs": len(pairs)}
    # falls by condition
    falls = defaultdict(lambda: [0, 0, 0])
    for (env, sc, _), b, c in pairs:
        f = falls[(env, sc)]
        f[0] += 1
        f[1] += int(b["fell"])
        f[2] += int(c["fell"])
    res["falls"] = {f"{e}/{s}": {"runs": n, "v1": fb, "candidate": fc} for (e, s), (n, fb, fc) in sorted(falls.items())}
    nominal = [k for k in falls if not k[1].startswith("push") or k[1] in ("push_walk_0.2", "push_walk_0.3",
                                                                          "push_stand_0.3")]
    rule1 = all(falls[k][2] <= falls[k][1] for k in falls) and all(falls[k][2] == 0 for k in nominal)
    # quality
    q = {}
    for m in ("stance_slip_m_s", "joint_accel_rms", "tilt_p95_deg", "torque_proxy", "swing_clearance_m",
              "stride_asymmetry_s"):
        allp = [(b[m], c[m]) for _, b, c in pairs if b.get(m) is not None and c.get(m) is not None]
        fam = {fa: ratio_median([(b[m], c[m]) for (e, s, _), b, c in pairs if family(s) == fa
                                 and b.get(m) is not None and c.get(m) is not None]) for fa in FAMILIES}
        q[m] = {"median_v1": float(np.median([x for x, _ in allp])) if allp else None,
                "median_candidate": float(np.median([y for _, y in allp])) if allp else None,
                "median_paired_ratio": ratio_median(allp), "family_ratio": fam}
    res["quality"] = q
    core = ("stance_slip_m_s", "joint_accel_rms", "tilt_p95_deg")
    rule2 = all((q[m]["median_paired_ratio"] or 9) < 1.0
                and all(r <= 1.10 for r in q[m]["family_ratio"].values() if r is not None) for m in core)
    # tracking per speed and turn
    track: dict = {}
    for sc in sorted({k[1] for k in keys}):
        bs = [b["speed_error"] for (e, s, _), b, c in pairs if s == sc and b.get("speed_error") is not None]
        cs = [c["speed_error"] for (e, s, _), b, c in pairs if s == sc and c.get("speed_error") is not None]
        bw = [b["yaw_error"] for (e, s, _), b, c in pairs if s == sc and b.get("yaw_error") is not None]
        cw = [c["yaw_error"] for (e, s, _), b, c in pairs if s == sc and c.get("yaw_error") is not None]
        track[sc] = {"speed_error_v1": float(np.mean(bs)) if bs else None,
                     "speed_error_candidate": float(np.mean(cs)) if cs else None,
                     "yaw_error_v1": float(np.mean(bw)) if bw else None,
                     "yaw_error_candidate": float(np.mean(cw)) if cw else None}
    res["tracking"] = track
    rule3 = all((t["speed_error_candidate"] is None or t["speed_error_v1"] is None or
                 t["speed_error_candidate"] <= max(0.03, t["speed_error_v1"] + 0.02)) for sc, t in track.items()
                if sc.startswith("speed_")) and \
        all((t["yaw_error_candidate"] is None or t["yaw_error_v1"] is None or
             t["yaw_error_candidate"] <= t["yaw_error_v1"] + 0.1) for t in track.values())
    stops = {m: {"median_v1": float(np.median(v)) if (v := [b[m] for _, b, c in pairs if b.get(m) is not None])
                 else None,
                 "median_candidate": float(np.median(w)) if (w := [c[m] for _, b, c in pairs if c.get(m) is not None])
                 else None} for m in ("stop_distance_m", "settle_s", "push_peak_tilt_deg")}
    res["stops_and_pushes"] = stops
    rule4 = all((s["median_candidate"] or 0) <= 1.2 * (s["median_v1"] or 1e9) for m, s in stops.items()
                if m in ("stop_distance_m", "settle_s"))
    res["rules"] = {"1_falls": rule1, "2_quality": rule2, "3_tracking": rule3, "4_stops": rule4}
    res["promote"] = all(res["rules"].values())
    res["per_run"] = {"/".join(map(str, k)): v for k, v in sorted(runs.items())}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
