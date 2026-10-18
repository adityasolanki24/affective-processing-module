"""Programme L Phase C: the perception code (pure NumPy; no simulator needed)."""
from __future__ import annotations

import inspect

import numpy as np

from apm_transfer.vision import MotionPerception, VisionCamera


def _scene(cam: VisionCamera, face: float, rng: np.random.Generator) -> np.ndarray:
    """A wall at 1.5 m with a column face at `face` m covering the central third of the image."""
    d = np.full((cam.height, cam.width), 1.5)
    d[:, cam.width // 3: 2 * cam.width // 3] = face
    d += rng.normal(0, 1, d.shape) * cam.depth_sigma * (1 + d)
    return d


def _run(frames: list[np.ndarray], cam: VisionCamera, d_nav: float = 0.6) -> tuple[list[float], MotionPerception]:
    p = MotionPerception(cam)
    for f in frames:
        p.frame(f, False, d_nav, 0.0)
    return p.end_window(d_nav), p


def test_static_scene_gives_no_events():
    cam, rng = VisionCamera(pitch_deg=0.0), np.random.default_rng(0)
    reaches, _ = _run([_scene(cam, 0.48, rng) for _ in range(40)], cam)
    assert reaches == []


def test_an_approaching_column_is_one_event_with_its_reach():
    cam, rng = VisionCamera(pitch_deg=0.0), np.random.default_rng(1)
    faces = [0.48] * 6 + [0.42, 0.34, 0.26, 0.34, 0.42] + [0.48] * 6  # face from 0.48 m to 0.26 m and back
    reaches, _ = _run([_scene(cam, f, rng) for f in faces], cam, d_nav=0.6)
    assert len(reaches) == 1
    assert abs(reaches[0] - (0.6 - 0.26)) < 0.03  # reach = navigation distance − closest face


def test_global_shift_is_egomotion_not_a_lunge():
    cam, rng = VisionCamera(pitch_deg=0.0), np.random.default_rng(2)
    frames = [_scene(cam, 0.48, rng) for _ in range(5)]
    shifted = [np.roll(_scene(cam, 0.48, rng), 6, axis=1) for _ in range(3)]  # the camera turned
    reaches, _ = _run(frames + shifted + [_scene(cam, 0.48, rng) for _ in range(5)], cam)
    assert reaches == []


def test_own_motion_frames_are_ignored():
    cam, rng = VisionCamera(pitch_deg=0.0), np.random.default_rng(3)
    p = MotionPerception(cam)
    for _ in range(5):
        p.frame(_scene(cam, 0.48, rng), False, 0.6)
    for f in (0.40, 0.30, 0.40):
        p.frame(_scene(cam, f, rng), True, 0.6)  # the robot itself moved: no evidence
    for _ in range(5):
        p.frame(_scene(cam, 0.48, rng), False, 0.6)
    assert p.end_window(0.6) == []


def test_perception_takes_only_images_and_self_knowledge():
    params = list(inspect.signature(MotionPerception.frame).parameters)
    assert params == ["self", "depth", "own_motion", "d_nav", "eye_offset"]
    assert list(inspect.signature(MotionPerception.__init__).parameters) == ["self", "cam"]
