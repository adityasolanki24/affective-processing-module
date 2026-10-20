"""Evaluation-only geometry; never supplied to APM's observations or controller."""
from __future__ import annotations

import numpy as np


def person_distance(robot_xy, person_xy, robot_front: float, person_radius: float) -> dict:
    """Centre distance and signed radial surface gap to the actual person collider.

    The front radius is an envelope approximation, not a mesh minimum-distance query.
    Negative values indicate overlap of those envelopes. Station coordinates must
    not be used here: the physical person's collider moves during an interaction.
    """
    centre = float(np.linalg.norm(np.asarray(robot_xy) - np.asarray(person_xy)))
    return {"root_to_person_centre_m": centre,
            "radial_surface_gap_m": centre - float(robot_front) - float(person_radius),
            "robot_front_m": float(robot_front), "person_radius_m": float(person_radius)}
