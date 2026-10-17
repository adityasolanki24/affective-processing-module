"""A kinematic stand-in for the biped controller: same interface and command set, no physics.

Used only for fast unit tests and for debugging adapters. It is never evidence of physical feasibility. Commands
are tracked with a first-order lag (time constant `tau`); tilt is zero unless a test injects it.
"""
from __future__ import annotations

import math

import numpy as np

from apm_express.adapter import COMMAND_NAMES, COMMANDS, MotionRequest


class KinematicBiped:
    dt = 0.02

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, tau: float = 0.3):
        self.x, self.y, self.yaw, self.tau = x, y, yaw, tau
        self.v = np.zeros(3)
        self.cmd: np.ndarray = np.zeros(3)
        self.head: np.ndarray = np.zeros(4)
        self.e_pitch = 0.0
        self.t = 0.0
        self.tilt = 0.0
        self.requests: list[MotionRequest] = []

    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw

    def physical(self) -> dict:
        return {"tilt_deg": self.tilt, "angular_speed": abs(float(self.v[2]))}

    def apply(self, request: MotionRequest) -> None:
        if request.command not in COMMAND_NAMES:
            raise ValueError(f"unknown command {request.command!r}")
        self.requests.append(request)
        self.cmd = np.array(COMMANDS[request.command])
        self.e_pitch, self.head = request.e_pitch, np.array(request.head)

    def push(self, dvx: float, dvy: float) -> None:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.v[0] += c * dvx + s * dvy
        self.v[1] += -s * dvx + c * dvy

    def advance(self) -> dict:
        a = self.dt / max(self.tau, self.dt)
        self.v += a * (self.cmd - self.v)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * self.v[0] - s * self.v[1]) * self.dt
        self.y += (s * self.v[0] + c * self.v[1]) * self.dt
        self.yaw = (self.yaw + self.v[2] * self.dt + math.pi) % (2 * math.pi) - math.pi
        self.t = round(self.t + self.dt, 10)
        return {"t": self.t, "x": self.x, "y": self.y, "z": 0.0, "yaw": self.yaw, "pitch": 0.02 * self.e_pitch,
                "tilt": self.tilt, "vx_world": c * self.v[0] - s * self.v[1], "vy_world": s * self.v[0] + c * self.v[1],
                "angular_speed": abs(float(self.v[2])), "head_joints": self.head.copy(), "power": 0.0, "tau_sq": 0.0,
                "finite": True}
