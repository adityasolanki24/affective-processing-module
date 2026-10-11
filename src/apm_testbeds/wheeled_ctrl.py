"""Intervention-capable wheeled testbed (APM v0.2, experiment E0). The v0.1 `apm_testbeds.wheeled` is unchanged.

Same robot physics and constants as v0.1 (unicycle, dt 0.05 s, 6 m arena, traction margin 1 − |vω|/(μg), slip on
low-friction patches). What is new:

* **A bounded high-level action**, held for one decision period (0.5 s = 10 steps) and chosen by the system under
  test: GO (ordinary v0.1-style reactive navigation), SLOW (speed ≤ 0.3), YIELD (stop, may turn toward the goal),
  DETOUR (wider berth around people, speed ≤ 0.6). Navigation, dynamics and safety are one code path for every system.
* **A safety supervisor** that no policy can bypass and that never depends on the APM: speed cap, and an emergency
  brake (3 m/s²) whenever an entity is within 0.15 m clearance in the forward ±70° sector and closing.
* **Separate random streams** (layout, goal sequence, each pedestrian, slip, sensing) so exogenous randomness is
  common across policies; the goal sequence is pre-generated and independent of the robot.
* **Pedestrians with persistent identities and hidden dispositions** (ground truth, never observed): *careful*
  people slow down and sidestep when the robot is close and approaching; *careless* people keep going and cut across
  a moving robot's path when close. Everyone walks around a stopped robot. Both look identical to the robot's sensors.
* **Goal deadlines** (missed deadline = task failure, next goal) and **contact episodes** (one collision event per
  entity per contact episode, 1.0 s refractory; severe if robot speed ≥ 0.4 m/s at contact).

Policies receive only `Obs` objects (what the robot measures, stamped at the current step). Ground truth for
evaluation is available separately through `truth()`; harness code must never pass it to a policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT = 0.05
ARENA = 6.0
ROBOT_R = 0.2
PERSON_R = 0.25
V_MAX, W_MAX = 0.8, 2.0
ACCEL, ALPHA = 1.0, 6.0
BRAKE_DECEL = 3.0
G = 9.81
MU_NORMAL, MU_SLIPPERY = 0.8, 0.08
SENSOR_RANGE = 3.0
DECISION_STEPS = 10
SAFETY_CLEARANCE, SAFETY_SECTOR = 0.05, math.radians(60)
CONTACT_REFRACTORY_S, SEVERE_SPEED = 1.0, 0.2
GOAL_TOLERANCE = 0.3

GO, SLOW, YIELD, DETOUR = range(4)
ACTIONS = ("GO", "SLOW", "YIELD", "DETOUR")
N_ACTIONS = len(ACTIONS)
# action → (speed cap m/s, person clearance range m, person repulsion gain)
MODES = {GO: (V_MAX, 0.8, 1.5), SLOW: (0.3, 0.8, 1.5), YIELD: (0.0, 0.8, 1.5), DETOUR: (0.6, 1.5, 3.0)}


@dataclass(frozen=True)
class Identity:
    name: str
    disposition: str  # "careful" | "careless" (ground truth; never observed)


@dataclass(frozen=True)
class WorldConfig:
    episode_s: float = 60.0
    people: tuple[int, int] = (2, 4)            # present per episode (inclusive), drawn from the population
    obstacles: tuple[int, int] = (3, 6)
    patches: tuple[int, int] = (0, 3)
    ped_speed: tuple[float, float] = (0.2, 1.0)
    deadline_base_s: float = 4.0
    deadline_speed: float = 0.35
    identity_error: float = 0.0                  # P(a present person is reported with another person's identity)
    identity_ambiguous: float = 0.0              # P(reported identity confidence is 0.4 instead of 0.9)
    patch_affinity: tuple[str, ...] = ()         # identities that hang around low-friction patches (E3 S5)
    force_present: tuple[str, ...] = ()          # identities that are always present
    n_goals: int = 60


@dataclass(frozen=True)
class Seen:
    track: str
    kind: str                 # "person" | "obstacle"
    identity: str | None      # reported identity (may be wrong under identity_error)
    identity_confidence: float
    distance: float           # clearance, m
    bearing: float            # rad, robot frame
    closing: float            # radial closing speed, m/s (> 0 approaching)
    ttc_p: float              # robot-side TTC collision estimate (v0.1 formula; 0 when not closing)


@dataclass(frozen=True)
class Obs:
    t: float
    v: float
    omega: float
    traction_margin: float    # 1 − |vω|/(μg), clipped to [0, 1] as in v0.1
    energy: float
    goal_distance: float
    goal_bearing: float
    goal_remaining_s: float
    goal_index: int
    entities: tuple[Seen, ...]
    events: tuple[tuple, ...]  # ("collision", track, severe) | ("slip",) | ("goal_reached",) | ("goal_timeout",)
    #                           | ("intervention",)
    action: int                # action in force during this step


def child(seed: np.random.SeedSequence, i: int) -> np.random.SeedSequence:
    """Deterministic child stream (unlike SeedSequence.spawn, it never depends on how often it was called)."""
    return np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, i))


def ttc_probability(distance: float, closing: float) -> float:
    """The v0.1 adapter's estimator: p = σ((1.5 s − TTC)/0.4 s) for TTC < 3 s; 0 when not closing (retracted)."""
    if closing <= 0.05:
        return 0.0
    ttc = distance / closing
    return 1.0 / (1.0 + math.exp(-(1.5 - ttc) / 0.4)) if ttc < 3.0 else 0.0


@dataclass
class _Body:
    oid: str
    kind: str
    x: float
    y: float
    r: float
    identity: Identity | None = None
    speed: float = 0.0
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    rng: np.random.Generator | None = None
    reported: str | None = None
    reported_confidence: float = 0.0


@dataclass
class EpisodeLog:
    collisions: list = field(default_factory=list)   # (t, track, kind, identity, disposition, severe)
    slips: list = field(default_factory=list)
    goals_issued: int = 0
    goals_completed: int = 0
    goal_times: list = field(default_factory=list)
    timeouts: int = 0
    interventions: int = 0
    distance: float = 0.0
    progress: float = 0.0
    idle_periods: int = 0
    decision_periods: int = 0
    actions: list = field(default_factory=list)
    fallbacks: int = 0
    contact_steps: list = field(default_factory=list)  # (t, track) for every step in contact (ground truth)


class WheeledCtrlWorld:
    def __init__(self, seed: np.random.SeedSequence, population: list[Identity], config: WorldConfig | None = None):
        self.config = cfg = config or WorldConfig()
        layout_ss, goal_ss, people_ss, slip_ss, sense_ss = (child(seed, i) for i in range(5))
        layout = np.random.default_rng(layout_ss)
        self.slip_rng = np.random.default_rng(slip_ss)
        sense = np.random.default_rng(sense_ss)
        self.patches = [(float(layout.uniform(1, 5)), float(layout.uniform(1, 5)), float(layout.uniform(0.6, 1.2)))
                        for _ in range(int(layout.integers(cfg.patches[0], cfg.patches[1] + 1)))]
        self.bodies: list[_Body] = []
        for i in range(int(layout.integers(cfg.obstacles[0], cfg.obstacles[1] + 1))):
            self.bodies.append(_Body(f"o{i}", "obstacle", float(layout.uniform(0.5, 5.5)),
                                     float(layout.uniform(0.5, 5.5)), float(layout.uniform(0.15, 0.4))))
        forced = [p for p in population if p.name in cfg.force_present]
        others = [p for p in population if p.name not in cfg.force_present]
        n = max(len(forced), int(layout.integers(cfg.people[0], cfg.people[1] + 1)))
        chosen = forced + [others[i] for i in layout.permutation(len(others))[:max(0, n - len(forced))]]
        person_streams = [child(people_ss, i) for i in range(max(1, len(chosen)))]
        names = [p.name for p in population]
        for i, ident in enumerate(chosen):
            rng = np.random.default_rng(person_streams[i])
            x, y = self._free_point(layout)
            body = _Body(f"p{i}", "person", x, y, PERSON_R, ident, float(rng.uniform(*cfg.ped_speed)),
                         float(rng.uniform(-math.pi, math.pi)), rng=rng)
            u = sense.random()
            if u < cfg.identity_error and len(names) > 1:
                body.reported, body.reported_confidence = str(sense.choice([n for n in names if n != ident.name])), 0.9
            elif u < cfg.identity_error + cfg.identity_ambiguous:
                body.reported, body.reported_confidence = ident.name, 0.4
            else:
                body.reported, body.reported_confidence = ident.name, 0.9
            self.bodies.append(body)
        self.x, self.y = self._free_point(layout)
        self.theta = float(layout.uniform(-math.pi, math.pi))
        self.v = self.omega = 0.0
        self.energy = float(layout.uniform(0.5, 1.0))
        goal_rng = np.random.default_rng(goal_ss)
        self.goals = [self._free_point(goal_rng) for _ in range(cfg.n_goals)]
        self.t = 0.0
        self.step_index = 0
        self.action = GO
        self.log = EpisodeLog()
        self._last_contact: dict[str, float] = {}
        self._was_slipping = False
        self._goal_i = -1
        self._next_goal()

    # ------------------------------------------------------------------ setup helpers
    def _free_point(self, rng: np.random.Generator) -> tuple[float, float]:
        for _ in range(200):
            x, y = (float(v) for v in rng.uniform(0.5, 5.5, 2))
            if all(math.hypot(x - o.x, y - o.y) > o.r + ROBOT_R + 0.3 for o in self.bodies if o.kind == "obstacle"):
                return x, y
        return 3.0, 3.0

    def _next_goal(self) -> None:
        self._goal_i += 1
        self.goal = self.goals[self._goal_i % len(self.goals)]
        d = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        self.goal_deadline = self.t + self.config.deadline_base_s + d / self.config.deadline_speed
        self.goal_issued_t = self.t
        self.log.goals_issued += 1

    def mu(self, x: float, y: float) -> float:
        return MU_SLIPPERY if any(math.hypot(x - px, y - py) < pr for px, py, pr in self.patches) else MU_NORMAL

    @property
    def done(self) -> bool:
        return self.t >= self.config.episode_s - 1e-9

    # ------------------------------------------------------------------ control (shared by every system)
    def _navigate(self, action: int) -> tuple[float, float]:
        cap, person_range, person_gain = MODES[action]
        gx, gy = self.goal[0] - self.x, self.goal[1] - self.y
        heading = math.atan2(gy, gx)
        push_x = push_y = 0.0
        nearest = nearest_person = math.inf
        for o in self.bodies:
            dx, dy = self.x - o.x, self.y - o.y
            d = math.hypot(dx, dy) - o.r - ROBOT_R
            if o.kind == "person":
                nearest_person = min(nearest_person, d)
            else:
                nearest = min(nearest, d)
            rng, gain = (person_range, person_gain) if o.kind == "person" else (0.8, 1.5)
            if d < rng:
                w = (rng - d) / rng
                norm = max(1e-6, d + o.r + ROBOT_R)
                push_x, push_y = push_x + gain / 1.5 * w * dx / norm, push_y + gain / 1.5 * w * dy / norm
        heading = math.atan2(math.sin(heading) + 1.5 * push_y, math.cos(heading) + 1.5 * push_x)
        error = math.atan2(math.sin(heading - self.theta), math.cos(heading - self.theta))
        w_cmd = max(-W_MAX, min(W_MAX, 3.0 * error))
        v_cmd = (cap * max(0.0, math.cos(error)) * min(1.0, max(0.15, nearest / 0.8))
                 * min(1.0, max(0.3, nearest_person / 0.4)) * min(1.0, math.hypot(gx, gy)))
        return v_cmd, w_cmd

    def _closing(self, o: _Body) -> tuple[float, float, float]:
        """(clearance, bearing in robot frame, radial closing speed) of a body."""
        dx, dy = o.x - self.x, o.y - self.y
        dist = math.hypot(dx, dy)
        d = dist - o.r - ROBOT_R
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - self.theta), math.cos(math.atan2(dy, dx) - self.theta))
        rel_vx, rel_vy = o.vx - self.v * math.cos(self.theta), o.vy - self.v * math.sin(self.theta)
        closing = -(rel_vx * dx + rel_vy * dy) / max(1e-6, dist)
        return d, bearing, closing

    def safety(self, v_cmd: float) -> tuple[float, bool]:
        """Safety supervisor: identical for every system, independent of any policy or APM state."""
        v_cmd = min(v_cmd, V_MAX)
        for o in self.bodies:
            d, bearing, closing = self._closing(o)
            if d < SAFETY_CLEARANCE and abs(bearing) < SAFETY_SECTOR and closing > 0 and self.v > 0:
                return 0.0, True
        return v_cmd, False

    # ------------------------------------------------------------------ dynamics
    def _move_people(self) -> None:
        for o in self.bodies:
            if o.kind != "person":
                continue
            assert o.rng is not None and o.identity is not None
            o.heading += float(o.rng.normal(0.0, 0.35)) * math.sqrt(DT)
            hx, hy = math.cos(o.heading), math.sin(o.heading)
            speed = o.speed
            if o.identity.name in self.config.patch_affinity and self.patches:
                px, py, _ = min(self.patches, key=lambda p: math.hypot(p[0] - o.x, p[1] - o.y))
                if math.hypot(px - o.x, py - o.y) > 0.4:
                    to = math.atan2(py - o.y, px - o.x)
                    hx, hy = 0.5 * hx + 0.5 * math.cos(to), 0.5 * hy + 0.5 * math.sin(to)
            rx, ry = self.x - o.x, self.y - o.y
            gap = math.hypot(rx, ry) - o.r - ROBOT_R
            moving = self.v > 0.15
            if not moving and gap < 0.6:
                away = math.atan2(-ry, -rx)  # everyone walks around a stopped robot
                hx, hy = 0.4 * hx + 0.6 * math.cos(away), 0.4 * hy + 0.6 * math.sin(away)
            elif o.identity.disposition == "careless" and moving and gap < 1.5:
                ax, ay = self.x + 0.6 * math.cos(self.theta) - o.x, self.y + 0.6 * math.sin(self.theta) - o.y
                to = math.atan2(ay, ax)  # inattentive: cuts across the moving robot's path
                hx, hy = 0.5 * hx + 0.5 * math.cos(to), 0.5 * hy + 0.5 * math.sin(to)
            elif o.identity.disposition == "careful" and moving and gap < 1.0:
                approach = self.v * (math.cos(self.theta) * -rx + math.sin(self.theta) * -ry) / max(1e-6, gap + 0.45)
                if approach > 0.05 or hx * rx + hy * ry > 0:
                    away = math.atan2(-ry, -rx)  # sidestep away and slow down
                    hx, hy = 0.4 * hx + 0.6 * math.cos(away), 0.4 * hy + 0.6 * math.sin(away)
                    speed *= 0.3
            norm = max(1e-6, math.hypot(hx, hy))
            o.heading = math.atan2(hy, hx)
            o.vx, o.vy = speed * hx / norm, speed * hy / norm
            for b in self.bodies:  # people walk around static obstacles
                if b.kind == "obstacle":
                    dx, dy = o.x - b.x, o.y - b.y
                    d = math.hypot(dx, dy)
                    if d < b.r + o.r + 0.05:
                        o.vx, o.vy = o.vx + speed * dx / max(d, 1e-6), o.vy + speed * dy / max(d, 1e-6)
            o.x, o.y = o.x + o.vx * DT, o.y + o.vy * DT
            if not 0.3 < o.x < ARENA - 0.3:
                o.x = min(ARENA - 0.3, max(0.3, o.x))
                o.heading = math.pi - o.heading
            if not 0.3 < o.y < ARENA - 0.3:
                o.y = min(ARENA - 0.3, max(0.3, o.y))
                o.heading = -o.heading

    def step(self, action: int) -> Obs:
        if action not in MODES:
            raise ValueError(f"illegal action {action!r}")
        self.action = action
        events: list[tuple] = []
        v_cmd, w_cmd = self._navigate(action)
        v_cmd, intervened = self.safety(v_cmd)
        if intervened:
            events.append(("intervention",))
            self.log.interventions += 1
        decel = BRAKE_DECEL if intervened else ACCEL
        self.v += max(-decel * DT, min(ACCEL * DT, v_cmd - self.v))
        self.omega += max(-ALPHA * DT, min(ALPHA * DT, w_cmd - self.omega))
        mu = self.mu(self.x, self.y)
        margin = 1.0 - abs(self.v * self.omega) / (mu * G)
        slipping = margin < 0
        if slipping:
            self.v *= 0.6
            self.theta += float(self.slip_rng.normal(0, 0.25))
        if slipping and not self._was_slipping:
            events.append(("slip",))
            self.log.slips.append(self.t + DT)
        self._was_slipping = slipping
        self.theta = math.atan2(math.sin(self.theta + self.omega * DT), math.cos(self.theta + self.omega * DT))
        self._move_people()
        speed_before = abs(self.v)
        nx, ny = self.x + self.v * math.cos(self.theta) * DT, self.y + self.v * math.sin(self.theta) * DT
        nx, ny = min(ARENA - ROBOT_R, max(ROBOT_R, nx)), min(ARENA - ROBOT_R, max(ROBOT_R, ny))
        t_next = self.t + DT
        contact = False
        for o in self.bodies:
            dx, dy = nx - o.x, ny - o.y
            dist = math.hypot(dx, dy)
            if dist < o.r + ROBOT_R:
                contact = True
                self.log.contact_steps.append((t_next, o.oid))
                # resolve penetration: the robot is pushed out along the contact normal
                k = (o.r + ROBOT_R + 1e-3) / max(dist, 1e-6)
                nx, ny = o.x + dx * k, o.y + dy * k
                nx, ny = min(ARENA - ROBOT_R, max(ROBOT_R, nx)), min(ARENA - ROBOT_R, max(ROBOT_R, ny))
                last = self._last_contact.get(o.oid)
                if last is None or t_next - last > CONTACT_REFRACTORY_S:
                    severe = speed_before >= SEVERE_SPEED
                    events.append(("collision", o.oid, severe))
                    ident = o.identity
                    self.log.collisions.append((t_next, o.oid, o.kind, ident.name if ident else None,
                                                ident.disposition if ident else None, severe))
                self._last_contact[o.oid] = t_next
        if contact:
            self.v = 0.0
        moved = math.hypot(nx - self.x, ny - self.y)
        before = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        self.x, self.y = nx, ny
        after = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        self.log.distance += moved
        self.log.progress += before - after
        self.energy = max(0.0, self.energy - DT * (0.002 + 0.006 * abs(self.v)))
        self.t = t_next
        self.step_index += 1
        if after < GOAL_TOLERANCE:
            events.append(("goal_reached",))
            self.log.goals_completed += 1
            self.log.goal_times.append(self.t - self.goal_issued_t)
            self._next_goal()
        elif self.t >= self.goal_deadline:
            events.append(("goal_timeout",))
            self.log.timeouts += 1
            self._next_goal()
        return self._observe(min(1.0, max(0.0, margin)), tuple(events))

    def _observe(self, margin: float, events: tuple[tuple, ...]) -> Obs:
        seen = []
        for o in self.bodies:
            d, bearing, closing = self._closing(o)
            if d > SENSOR_RANGE:
                continue
            d = max(0.0, d)
            seen.append(Seen(o.oid, o.kind, o.reported, o.reported_confidence, d, bearing, closing,
                             ttc_probability(d, closing)))
        gx, gy = self.goal[0] - self.x, self.goal[1] - self.y
        gb = math.atan2(math.sin(math.atan2(gy, gx) - self.theta), math.cos(math.atan2(gy, gx) - self.theta))
        return Obs(self.t, self.v, self.omega, margin, self.energy, math.hypot(gx, gy), gb,
                   max(0.0, self.goal_deadline - self.t), self._goal_i, tuple(seen), events, self.action)

    def observe_now(self) -> Obs:
        """The current observation without events (for the first decision of an episode)."""
        margin = 1.0 - abs(self.v * self.omega) / (self.mu(self.x, self.y) * G)
        return self._observe(min(1.0, max(0.0, margin)), ())

    def truth(self) -> dict:
        """Ground truth for evaluation only (never passed to a policy)."""
        return {"t": self.t, "robot": (self.x, self.y, self.theta, self.v),
                "people": {o.oid: (o.x, o.y, o.identity.name if o.identity else None,
                                   o.identity.disposition if o.identity else None, o.reported)
                           for o in self.bodies if o.kind == "person"},
                "mu": self.mu(self.x, self.y)}
