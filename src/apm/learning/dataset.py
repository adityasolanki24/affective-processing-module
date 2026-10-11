"""Experience → supervised sequences for outcome prediction.

An experience is the AffectLink message stream of one episode. For each sample time (every `stride_s`) the record is

    observation history (features over the last `history` samples)  →  did <kind> occur within (t, t + horizon]?

Labels come only from OutcomeSignal messages (observed). OutcomeHypothesis messages are never labels. Samples are
"armed" only: none during an ongoing stability loss (onset → recovery or re-arm) and none after a fall, so the
model is never asked to predict an event that is already happening.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from apm.learning.features import WIDTH, FeatureTracker
from apm.protocol.messages import EmbodimentProfile, OutcomeSignal


@dataclass
class Episode:
    episode_id: str
    times: np.ndarray            # sample times (s)
    features: np.ndarray         # [T, WIDTH]
    events: dict[str, list[float]]  # observed outcome kind → times
    meta: dict
    extra: np.ndarray | None = None  # optional per-sample extra inputs (e.g. APM affect), [T, k]


def episode_from_stream(episode_id: str, profile: EmbodimentProfile, stream: Iterable[tuple[float, list]],
                        stride_s: float, meta: dict, on_sample=None) -> Episode:
    """`stream`: (t, messages) per control step. `on_sample(t, messages_so_far)` may return extra inputs."""
    tracker = FeatureTracker(profile)
    times, rows, extras = [], [], []
    events: dict[str, list[float]] = {}
    next_sample = None
    for t, messages in stream:
        for m in messages:
            tracker.ingest(m)
            if isinstance(m, OutcomeSignal):
                events.setdefault(m.kind, []).append(m.header.stamp_s)
        if t <= 0:
            continue
        if next_sample is None:
            next_sample = t
        if t + 1e-9 >= next_sample:
            times.append(t)
            rows.append(tracker.vector(t))
            if on_sample is not None:
                extras.append(on_sample(t))
            next_sample += stride_s
    return Episode(episode_id, np.asarray(times), np.asarray(rows).reshape(-1, WIDTH), events, meta,
                   np.asarray(extras) if extras else None)


def armed_mask(times: np.ndarray, events: dict[str, list[float]], rearm_s: float = 1.0) -> np.ndarray:
    """False during [stability_loss onset, recovery (or onset + rearm_s)] and after any fall."""
    armed = np.ones(len(times), bool)
    recoveries = sorted(events.get("recovery", []))
    for onset in events.get("stability_loss", []):
        end = next((r for r in recoveries if r >= onset), onset + rearm_s)
        armed &= ~((times >= onset) & (times <= max(end, onset + rearm_s)))
    for fall in events.get("fall", []):
        armed &= times < fall
    return armed


def samples(episodes: list[Episode], kind: str, horizon_s: float, history: int, use_extra: bool = False):
    """→ X [N, history, F], y [N], group index [N] (episode), sample times [N]."""
    xs, ys, gs, ts = [], [], [], []
    for gi, e in enumerate(episodes):
        if use_extra and e.extra is None:
            raise ValueError("episode has no extra inputs")
        feats = np.concatenate([e.features, e.extra], axis=1) if use_extra and e.extra is not None else e.features
        padded = np.concatenate([np.zeros((history - 1, feats.shape[1])), feats])
        armed = armed_mask(e.times, e.events)
        occurrences = np.asarray(e.events.get(kind, []))
        for i, t in enumerate(e.times):
            if not armed[i]:
                continue
            xs.append(padded[i:i + history])
            ys.append(float(np.any((occurrences > t) & (occurrences <= t + horizon_s))))
            gs.append(gi)
            ts.append(t)
    extra = episodes[0].extra
    width = episodes[0].features.shape[1] + (extra.shape[1] if use_extra and extra is not None else 0)
    return (np.asarray(xs, dtype=np.float32).reshape(-1, history, width), np.asarray(ys, dtype=np.float32),
            np.asarray(gs), np.asarray(ts))
