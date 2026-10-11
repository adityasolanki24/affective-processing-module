"""Helpers shared by embodiment adapters (robot side of AffectLink)."""
from __future__ import annotations

from dataclasses import dataclass, field

from apm.protocol.messages import SCHEMA_VERSION, Header


@dataclass
class Emitter:
    """Stamps AffectLink headers for one source: per-stream strictly increasing sequence numbers within an epoch."""
    source: str
    epoch: int = 0
    seq: dict[str, int] = field(default_factory=dict)

    def header(self, stream: str, stamp_s: float) -> Header:
        self.seq[stream] = self.seq.get(stream, -1) + 1
        return Header(schema_version=SCHEMA_VERSION, source=self.source, stream=stream, epoch=self.epoch,
                      seq=self.seq[stream], stamp_s=float(stamp_s))

    def restart(self) -> None:
        """Begin a new epoch (send a Reset with the first header of the new epoch)."""
        self.epoch += 1
        self.seq.clear()
