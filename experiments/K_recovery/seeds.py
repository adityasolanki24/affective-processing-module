"""Frozen programme-K seed partitions: the only source of K world seeds. New entropy, independent of F–J."""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0007_2026_1018
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "D": 5}  # pilot, train, validation, locked test, demonstration
EXPERIMENTS = {"K1": 1, "K2": 2, "K3": 3, "K4": 4, "K5": 5, "K6": 6}  # K6 (humanoid) added 2026-10-18; others unchanged


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))
