"""APM v1.0 M1: the end-to-end integration check (MuJoCo wheeled robot, abstract camera so it runs without GL)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs the sim extra (mujoco)")
def test_integration_end_to_end():
    sys.path.insert(0, str(REPO / "experiments" / "M_release" / "scripts"))
    import m1_integration
    assert m1_integration.main(["--camera", "abstract", "--n", "8"]) == 0
