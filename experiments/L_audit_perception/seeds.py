"""Frozen programme-L seed partitions: the only source of L world seeds. New entropy, independent of F–K.

Experiments are added (never renumbered) as each section of `PROTOCOL.md` is frozen."""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0008_2026_1018
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "D": 5}  # pilot, train, validation, locked test, demonstration
EXPERIMENTS = {"LB1": 1, "LB2": 2, "LB3": 3, "LB4": 4,  # B: NERVA, wheeled, humanoid recovery; NERVA frozen tasks
               "LC1": 5, "LC2": 6, "LC3": 7,  # C: perception quality, NERVA / wheeled / humanoid
               "LC4": 8, "LC5": 9,  # C: vision type models (T) and recovery comparison (V, X), NERVA / wheeled
               "LD": 10,  # D: stress scenarios (wheeled, IMU + vision)
               "LE": 11,  # E: expression-controller comparison (NERVA, IMU + vision): T tuning, X test
               "LF": 12, "LG": 13}  # F: runtime demos (partition D); G: continuous demonstration (partition D)


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))
