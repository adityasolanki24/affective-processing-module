"""Episodic memory: a bounded record of observed outcomes with the predictions and state that preceded them.

Each episode pairs what was *predicted* (the strongest live hypothesis of the same kind, if any) with what was
*observed*, plus a compact snapshot of affect and body margins. This is the raw material for calibration and for
the learning stage; it is never itself treated as new evidence (replaying an episode adds no counts).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

MAX_EPISODES = 2048


@dataclass(frozen=True)
class Episode:
    stamp_s: float
    outcome_id: str
    kind: str
    magnitude: float
    confirmed: bool
    evidence_quality: float
    target: str | None
    identity: str | None
    identity_confidence: float
    predicted_probability: float | None
    action_context: str | None
    affect: tuple[float, float, float]
    margins: dict[str, float]


class EpisodicMemory:
    def __init__(self, capacity: int = MAX_EPISODES):
        self.episodes: deque[Episode] = deque(maxlen=capacity)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def recent(self, n: int = 10) -> list[Episode]:
        return list(self.episodes)[-n:]

    def to_dict(self) -> dict:
        return {"capacity": self.episodes.maxlen, "episodes": [asdict(e) for e in self.episodes]}

    @classmethod
    def from_dict(cls, data: dict) -> EpisodicMemory:
        memory = cls(int(data["capacity"]))
        for e in data["episodes"]:
            memory.add(Episode(**{**e, "affect": tuple(e["affect"])}))
        return memory
