"""Frozen programme-M (APM v1.0) seed partitions: the only source of M world seeds. New entropy, independent of F–L."""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0010_2026_1019
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "D": 5, "Y": 6}  # pilot, train, validation, locked test, demo,
# Y: the remediation's corrected confirmatory test (fresh; never used before its protocol was frozen)
EXPERIMENTS = {"M5": 5, "M6": 6, "M7": 7, "R": 8}  # M5 fitting, M6 experiments A-F, M7 demos, R remediation


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))
