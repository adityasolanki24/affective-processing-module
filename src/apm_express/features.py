"""Motion features of an expressive approach episode, grouped as in MASTER_PREREGISTRATION.md §5.

Input: the arrays from `apm_express.episode.run_episode` (biped or kinematic), the target, and the start.
Every feature is computed from measured motion (pose, velocities, joint angles), never from the intent. The one
exception is `command`, the validated command actually issued, which is part of the observable behaviour.
`SPEED` lists the features that carry mean speed or progress. The "non-speed" analyses drop them.
"""
from __future__ import annotations

import math

import numpy as np

from apm_express.adapter import COMMAND_NAMES, TRANSLATING

GROUPS = {
    "speed": ("progress_speed", "arrival_time", "moving_fraction", "path_speed"),
    "temporal": ("pauses_per_m", "mean_pause_s", "switch_rate", "backward_s", "pause_regularity"),
    "posture": ("pitch_moving", "pitch_rest", "face_elev_rest", "face_yaw_rest_std", "face_yaw_rest_to_target"),
    "acceleration": ("rms_acc", "rms_jerk", "sparc"),
    "spatial": ("straightness", "max_lateral_dev", "final_distance", "final_bearing_abs", "turn_rate",
                "body_yaw_rest_std", "retreat_m"),
    "cost": ("mean_power", "mean_tau_sq"),
}
SPEED = frozenset(GROUPS["speed"])
ALL = tuple(f for g in GROUPS.values() for f in g)
TRANSLATING_IDX = np.array([COMMAND_NAMES.index(c) for c in sorted(TRANSLATING)])
REST_IDX = COMMAND_NAMES.index("rest")
BACKWARD_IDX = COMMAND_NAMES.index("backward")


def sparc(speed: np.ndarray, dt: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2015); more negative = less smooth. NaN if no movement."""
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 4 or np.max(np.abs(speed)) < 1e-6:
        return float("nan")
    nfft = int(2 ** (math.ceil(math.log2(len(speed))) + padlevel))
    f = np.arange(0, nfft) * (1.0 / dt) / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    keep = f <= fc
    f, mag = f[keep], mag[keep]
    above = np.nonzero(mag >= amp_th)[0]
    f, mag = f[above[0]:above[-1] + 1], mag[above[0]:above[-1] + 1]
    if len(f) < 2:
        return 0.0
    df = np.diff(f) / (f[-1] - f[0])
    dm = np.diff(mag)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


GAIT_STEPS = 27


def _gait_average(v: np.ndarray, n: int = GAIT_STEPS) -> np.ndarray:
    """Centred moving average with edge padding (same length)."""
    v = np.asarray(v, dtype=float)
    if len(v) < n:
        return v
    pad = np.pad(v, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def episode_features(arrays: dict, target, start=(0.0, 0.0), dt: float = 0.02, smooth_n: int = 25) -> dict:
    t = arrays["t"]
    # positions averaged over one gait period (27 steps = 0.54 s) so that step-to-step sway does not count as path
    x, y = _gait_average(arrays["x"]), _gait_average(arrays["y"])
    yaw = arrays["yaw"]
    cmd = arrays["command"].astype(int)
    tx, ty = float(target[0]), float(target[1])
    dist = np.hypot(tx - x, ty - y)
    d0 = float(math.hypot(tx - start[0], ty - start[1]))
    final = float(dist[-1])
    within = dist <= final + 0.05
    # arrival = first time after which the distance stays within final + 0.05 m
    later_out = np.nonzero(~within)[0]
    k_arr = int(later_out[-1] + 1) if len(later_out) else 0
    k_arr = min(max(k_arr, 1), len(t) - 1)
    arrival = float(t[k_arr])
    app = slice(0, k_arr + 1)
    post = slice(k_arr, len(t))

    c, s = np.cos(yaw), np.sin(yaw)
    vfwd = c * arrays["vx_world"] + s * arrays["vy_world"]
    speed = np.hypot(arrays["vx_world"], arrays["vy_world"])
    kern = np.ones(smooth_n) / smooth_n
    sm = np.convolve(vfwd, kern, mode="same")
    acc = np.gradient(sm, dt)
    jerk = np.gradient(np.convolve(acc, kern, mode="same"), dt)

    seg_xy = np.hypot(np.diff(x[app]), np.diff(y[app]))
    path_len = float(seg_xy.sum())
    progress = max(d0 - final, 1e-6)
    translating = np.isin(cmd, TRANSLATING_IDX)
    rest = cmd == REST_IDX

    pauses = [(a, b) for a, b in _segments(rest[app]) if (b - a) * dt >= 0.4 and a > 0]
    pause_lengths = np.array([(b - a) * dt for a, b in pauses])
    gaps = np.diff([a for a, _ in pauses]) * dt if len(pauses) > 2 else np.array([])
    regularity = float(np.std(gaps) / np.mean(gaps)) if len(gaps) >= 2 and np.mean(gaps) > 0 else 0.0

    sx, sy = start
    line = np.array([tx - sx, ty - sy]) / max(d0, 1e-9)
    lateral = np.abs((x[app] - sx) * line[1] - (y[app] - sy) * line[0])
    bearing = _wrap(np.arctan2(ty - y, tx - x) - yaw)
    yaw_rate = np.gradient(np.unwrap(yaw), dt)
    rel_progress = -np.diff(dist[app])

    rest_post = rest[post]
    face_yaw = arrays.get("face_yaw")
    face_yaw_arr = np.zeros(len(t)) if face_yaw is None else np.asarray(face_yaw)
    face_elev = arrays.get("face_elev")
    has_face = face_yaw is not None and bool(rest_post.any())
    pitch = arrays["pitch"]

    def rest_mean(a, sl=post, mask=None):
        if a is None:
            return 0.0
        v = a[sl]
        m = rest[sl] if mask is None else mask
        return float(np.nanmean(v[m])) if m.any() else float(np.nanmean(v))

    feats = {
        "progress_speed": progress / max(arrival, dt),
        "arrival_time": arrival,
        "moving_fraction": float(translating[app].mean()),
        "path_speed": path_len / max(arrival, dt),
        "pauses_per_m": len(pauses) / progress,
        "mean_pause_s": float(pause_lengths.mean()) if len(pause_lengths) else 0.0,
        "switch_rate": float(np.count_nonzero(np.diff(cmd)) / max(t[-1] - t[0], dt)),
        "backward_s": float(np.count_nonzero(cmd[app] == BACKWARD_IDX) * dt),
        "pause_regularity": regularity,
        "pitch_moving": float(pitch[translating].mean()) if translating.any() else float(pitch.mean()),
        "pitch_rest": float(pitch[rest].mean()) if rest.any() else float(pitch.mean()),
        "face_elev_rest": rest_mean(face_elev),
        "face_yaw_rest_std": float(np.nanstd(face_yaw_arr[post][rest_post])) if has_face else 0.0,
        "face_yaw_rest_to_target": float(np.nanmean(np.abs(_wrap(face_yaw_arr[post] - bearing[post]))[rest_post]))
        if has_face else 0.0,
        "rms_acc": float(np.sqrt(np.mean(acc[app] ** 2))),
        "rms_jerk": float(np.sqrt(np.mean(jerk[app] ** 2))),
        "sparc": sparc(speed[app], dt),
        "straightness": float(min(1.0, math.hypot(x[k_arr] - sx, y[k_arr] - sy) / max(path_len, 1e-6))),
        "max_lateral_dev": float(lateral.max()) if len(lateral) else 0.0,
        "final_distance": final,
        "final_bearing_abs": float(np.abs(bearing[-50:]).mean()),
        "turn_rate": float(np.mean(np.abs(yaw_rate))),
        "body_yaw_rest_std": float(np.std(np.unwrap(yaw[post])[rest_post])) if rest_post.any() else 0.0,
        "retreat_m": float(np.clip(-rel_progress, 0, None).sum()),
        "mean_power": float(np.mean(arrays["power"])),
        "mean_tau_sq": float(np.mean(arrays["tau_sq"])),
    }
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in feats.items()}


def stability(arrays: dict, completed: bool = True, fall_tilt: float = 45.0) -> dict:
    tilt = arrays["tilt"]
    return {"fell": bool((tilt > fall_tilt).any() or not completed), "max_tilt": float(np.nanmax(tilt)),
            "fallback_s": float(np.count_nonzero(arrays.get("fallback", np.zeros(1))) * 0.02)}
