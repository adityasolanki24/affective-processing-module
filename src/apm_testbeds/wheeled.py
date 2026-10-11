"""A minimal differential-drive wheeled robot testbed (second embodiment for APM portability experiments).

Deliberately simple, not a navigation stack: a 2D 6 m × 6 m arena with static circular obstacles, moving pedestrians
(each with a persistent identity), low-friction floor patches and a battery. The robot drives to random waypoints
with a reactive controller (heading to goal, speed reduced and steering deflected near obstacles), which is
imperfect on purpose so that collisions, wheel slip, blocking and goal completion all occur.

Physics (unicycle, dt = 0.05 s): v, ω follow the controller with acceleration limits. Wheel traction margin
    m = 1 − |v·ω| / (μ g)            (lateral acceleration against the floor's friction limit)
with μ = 0.8 on normal floor and 0.08 on slippery patches. m < 0 is a slip: the robot loses speed and its heading
is perturbed. The battery drains with time and speed. The testbed reports ground truth only; the APM adapter
(`apm.adapters.wheeled`) turns it into AffectLink messages and owns the outcome detectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT = 0.05
ARENA = 6.0
ROBOT_R = 0.2
V_MAX, W_MAX = 0.8, 2.0
ACCEL, ALPHA = 1.0, 6.0
G = 9.81
MU_NORMAL, MU_SLIPPERY = 0.8, 0.08
SENSOR_RANGE = 3.0


@dataclass
class Obstacle:
    oid: str
    kind: str          # "obstacle" | "person"
    identity: str | None
    x: float
    y: float
    r: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Telemetry:
    t: float
    x: float
    y: float
    theta: float
    v: float
    omega: float
    traction_margin: float
    energy: float
    slipping: bool
    goal: tuple[float, float]
    goal_start_distance: float
    goal_distance: float
    reached: bool
    collisions: list[str]
    entities: list[dict] = field(default_factory=list)


class WheeledWorld:
    def __init__(self, seed: int, obstacles: int = 6, people: int = 3, patches: int = 3):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.patches = [(r.uniform(1, 5), r.uniform(1, 5), r.uniform(0.6, 1.2)) for _ in range(patches)]
        self.items: list[Obstacle] = []
        for i in range(obstacles):
            self.items.append(Obstacle(f"o{i}", "obstacle", None, r.uniform(0.5, 5.5), r.uniform(0.5, 5.5),
                                       r.uniform(0.15, 0.4)))
        for i in range(people):
            speed, heading = r.uniform(0.2, 0.9), r.uniform(0, 2 * math.pi)
            self.items.append(Obstacle(f"p{i}", "person", f"ped-{(seed * 7 + i) % 12}", r.uniform(0.5, 5.5),
                                       r.uniform(0.5, 5.5), 0.25, speed * math.cos(heading), speed * math.sin(heading)))
        self.x, self.y = self._free_point()
        self.theta, self.v, self.omega = r.uniform(-math.pi, math.pi), 0.0, 0.0
        self.energy = float(r.uniform(0.25, 1.0))
        self.t = 0.0
        self._new_goal()
        self._touching: set[str] = set()

    def _free_point(self) -> tuple[float, float]:
        for _ in range(200):
            x, y = self.rng.uniform(0.5, 5.5, 2)
            if all(math.hypot(x - o.x, y - o.y) > o.r + ROBOT_R + 0.3 for o in self.items):
                return float(x), float(y)
        return 3.0, 3.0

    def _new_goal(self) -> None:
        self.goal = self._free_point()
        self.goal_start = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)

    def mu(self, x: float, y: float) -> float:
        return MU_SLIPPERY if any(math.hypot(x - px, y - py) < pr for px, py, pr in self.patches) else MU_NORMAL

    def _control(self) -> tuple[float, float]:
        gx, gy = self.goal[0] - self.x, self.goal[1] - self.y
        heading = math.atan2(gy, gx)
        push_x = push_y = 0.0
        nearest = math.inf
        for o in self.items:
            dx, dy = self.x - o.x, self.y - o.y
            d = math.hypot(dx, dy) - o.r - ROBOT_R
            nearest = min(nearest, d)
            if d < 0.8:
                w = (0.8 - d) / 0.8
                norm = max(1e-6, d + o.r + ROBOT_R)
                push_x, push_y = push_x + w * dx / norm, push_y + w * dy / norm
        heading = math.atan2(math.sin(heading) + 1.5 * push_y, math.cos(heading) + 1.5 * push_x)
        error = math.atan2(math.sin(heading - self.theta), math.cos(heading - self.theta))
        w_cmd = max(-W_MAX, min(W_MAX, 3.0 * error))
        v_cmd = V_MAX * max(0.0, math.cos(error)) * min(1.0, max(0.15, nearest / 0.8)) * min(1.0, math.hypot(gx, gy))
        return v_cmd, w_cmd

    def step(self) -> Telemetry:
        v_cmd, w_cmd = self._control()
        self.v += max(-ACCEL * DT, min(ACCEL * DT, v_cmd - self.v))
        self.omega += max(-ALPHA * DT, min(ALPHA * DT, w_cmd - self.omega))
        mu = self.mu(self.x, self.y)
        margin = 1.0 - abs(self.v * self.omega) / (mu * G)
        slipping = margin < 0
        if slipping:  # traction lost: speed bleeds off and heading is perturbed
            self.v *= 0.6
            self.theta += float(self.rng.normal(0, 0.25))
        self.theta = math.atan2(math.sin(self.theta + self.omega * DT), math.cos(self.theta + self.omega * DT))
        nx, ny = self.x + self.v * math.cos(self.theta) * DT, self.y + self.v * math.sin(self.theta) * DT
        collisions = []
        for o in self.items:
            if o.kind == "person":
                o.x, o.y = o.x + o.vx * DT, o.y + o.vy * DT
                if not 0.3 < o.x < ARENA - 0.3:
                    o.vx = -o.vx
                if not 0.3 < o.y < ARENA - 0.3:
                    o.vy = -o.vy
            if math.hypot(nx - o.x, ny - o.y) < o.r + ROBOT_R:
                collisions.append(o.oid)
        new = [c for c in collisions if c not in self._touching]
        self._touching = set(collisions)
        if collisions:  # contact: the robot is stopped and pushed back
            self.v = -0.1
        else:
            self.x, self.y = min(ARENA - ROBOT_R, max(ROBOT_R, nx)), min(ARENA - ROBOT_R, max(ROBOT_R, ny))
        self.energy = max(0.0, self.energy - DT * (0.002 + 0.006 * abs(self.v)))
        self.t += DT
        distance = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        reached = distance < 0.3
        telemetry = Telemetry(self.t, self.x, self.y, self.theta, self.v, self.omega, margin, self.energy, slipping,
                              self.goal, self.goal_start, distance, reached, new, self._observe())
        if reached:
            self._new_goal()
        return telemetry

    def _observe(self) -> list[dict]:
        out = []
        for o in self.items:
            dx, dy = o.x - self.x, o.y - self.y
            d = math.hypot(dx, dy) - o.r - ROBOT_R
            if d > SENSOR_RANGE:
                continue
            bearing = math.atan2(math.sin(math.atan2(dy, dx) - self.theta), math.cos(math.atan2(dy, dx) - self.theta))
            rel_vx = o.vx - self.v * math.cos(self.theta)
            rel_vy = o.vy - self.v * math.sin(self.theta)
            closing = -(rel_vx * dx + rel_vy * dy) / max(1e-6, math.hypot(dx, dy))
            out.append({"track": o.oid, "kind": o.kind, "identity": o.identity, "distance": max(0.0, d),
                        "bearing": bearing, "closing": closing})
        return out
