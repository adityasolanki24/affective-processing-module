"""Decision-level datasets from exploration lifetimes: features at the decision, the action actually taken, and
what actually happened over the next 2 s (collision, slip, goal progress). Decisions within 2 s of the episode end
are dropped (their outcome window is censored). Ground-truth columns are kept separately for analysis only and are
never model inputs."""
from __future__ import annotations

import numpy as np

from apm_research.predictor import HORIZON_DECISIONS


class DecisionRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._episode: list[dict] = []

    def decision(self, world, policy, action: int, t: float) -> None:
        x = policy.last_features
        truth = world.truth()
        people = [p for p in truth["people"].values()]
        rx, ry = truth["robot"][0], truth["robot"][1]
        near = min(people, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2, default=None)
        self._episode.append({"x": None if x is None else x.copy(), "a": action, "t": t,
                              "collisions": len(world.log.collisions), "slips": len(world.log.slips),
                              "progress": world.log.progress,
                              "near_disposition": near[3] if near else None})

    def end_episode(self, world) -> None:
        ep = self._episode
        for i in range(len(ep) - HORIZON_DECISIONS):
            now, later = ep[i], ep[i + HORIZON_DECISIONS]
            if now["x"] is None:
                continue
            self.rows.append({**now, "y_col": float(later["collisions"] > now["collisions"]),
                              "y_slip": float(later["slips"] > now["slips"]),
                              "y_prog": float(later["progress"] - now["progress"])})
        self._episode = []

    def arrays(self):
        X = np.stack([r["x"] for r in self.rows]).astype(np.float32)
        A = np.array([r["a"] for r in self.rows])
        return (X, A, np.array([r["y_col"] for r in self.rows], np.float32),
                np.array([r["y_slip"] for r in self.rows], np.float32),
                np.array([r["y_prog"] for r in self.rows], np.float32))
