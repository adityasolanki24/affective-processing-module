"""Exploratory gait quality from synchronized recorded physics; no success claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def forward_velocity(velocity, quaternion):
    """Project measured world velocity onto the torso's forward axis; sway is not progress."""
    qw, qx, qy, qz = np.asarray(quaternion).T
    forward = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz),
                        2 * (qx * qz - qw * qy)], axis=-1)
    return np.sum(np.asarray(velocity) * forward, axis=-1)


def summarize(folder: Path) -> dict:
    meta = json.loads((folder / "metadata.json").read_text())
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    t = np.array([r["t"] for r in rows])
    dt = np.diff(t)
    feet = np.array([r["feet"] for r in rows])
    force = np.linalg.norm(np.array([r["contact_force"] for r in rows]), axis=2)
    contact = force > 5.0
    foot_speed = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) / dt[:, None]
    stance = contact[1:] & contact[:-1]
    clearance, strides = [], []
    for foot in range(2):
        z = feet[:, foot, 2]
        floor = float(np.median(z[contact[:, foot]])) if contact[:, foot].any() else None
        swing = z[~contact[:, foot]]
        clearance.append(float(np.percentile(swing - floor, 95)) if floor is not None and len(swing) else None)
        touchdowns = t[1:][contact[1:, foot] & ~contact[:-1, foot]]
        intervals = np.diff(touchdowns)
        intervals = intervals[(intervals > .2) & (intervals < 3.)]
        strides.append(float(np.median(intervals)) if len(intervals) else None)
    dq = np.array([r["dq"] for r in rows])
    accel = np.diff(dq, axis=0) / dt[:, None]
    velocity = np.array([r["linear_velocity"] for r in rows])
    speed = forward_velocity(velocity, [r["root_quaternion"] for r in rows])
    # Straight walking only, excluding the two-second start transient.
    straight = (t >= 4.) & (t < 8.)
    return {"metadata": meta, "stance_slip_mean_m_s": float(foot_speed[stance].mean()) if stance.any() else None,
            "contact_samples": contact.sum(axis=0).tolist(), "swing_clearance_p95_m": clearance,
            "stride_median_s": strides,
            "stride_asymmetry_s": abs(strides[0] - strides[1]) if all(x is not None for x in strides) else None,
            "tilt_p95_deg": float(np.percentile([r["tilt"] for r in rows], 95)),
            "minimum_root_height_m": min(r["z"] for r in rows),
            "joint_acceleration_rms_rad_s2": float(np.sqrt(np.mean(accel ** 2))),
            "straight_speed_m_s": float(speed[straight].mean()) if straight.any() else None,
            "world_x_displacement_m": float(rows[-1]["root_position"][0] - rows[0]["root_position"][0]),
            "fall": meta["fell"], "observed_s": float(t[-1]),
            "limitations": ["Pilot, not confirmatory or visual acceptance.",
                            "Stance requires >5 N on both ends of a 20 ms sample.",
                            "Clearance is relative to each foot-link's median stance height."]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = {p.name: summarize(p) for p in a.folders}
    a.out.write_text(json.dumps(result, indent=2))
    for name, row in result.items():
        print(name, {k: v for k, v in row.items() if k not in ("metadata", "limitations")})


if __name__ == "__main__":
    main()
