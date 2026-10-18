# APM v1.0 release: milestone tracker

This is the only tracker for the final sprint. The final account is `REPORT.md`. Confirmatory tests are frozen in
`PROTOCOL.md` before their data exist.

**Branch:** `release/apm-v1.0`, from `research/apm-v0.8-audit` at 84aeef9. Nothing is merged without approval.

**Cloud cap:** US$500, approved 2026-10-18. Spending is logged in the cost section of `REPORT.md`.

**Pinned stack:**
- Isaac Sim 5.1.0 (pip wheel, Python 3.11, Ubuntu 22.04);
- Isaac Lab v2.3.2;
- GCP L4 (Ada, RT cores, 24 GB);
- NVIDIA driver 580 (Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580`).

Not chosen:
- Isaac Sim 6.x: Isaac Lab's stable branch still targets 5.1.
- Isaac Lab 3.0: early access.
- A100 and H100: they lack RT cores.

| # | milestone | gate (must pass before the next safety-critical step) | status |
|---|---|---|---|
| M1 | freeze and stabilise v0.8 | clean integration test (runtime → sensors → memory → decision → bounded command → safety veto → motion → persist/restart → replay → AffectLink) passes from a clean checkout; L's open defects fixed (task-scoring artefact, falls characterised) | done (433294a) |
| M2 | Isaac Sim infrastructure | pinned env restores on a fresh VM; headless render of a stock scene; physics/sensor smoke test | done |
| M3 | three embodiments in Isaac | each robot imported, controller validated by a battery (stand, walk, stop, turn, pushes); MuJoCo vs Isaac characterised | gates passed (see below) |
| M4 | perception and interaction | camera/IMU/odometry → perception → `Observation`, no simulator truth; moving agents with physical presence; ≥ 2 functional scenes | in progress |
| M5 | integrated APM | the same runtime drives all three robots in Isaac; body-specific expressive realisation | |
| M6 | final evaluation | `PROTOCOL.md` frozen; experiments A–F on fresh seeds; held-out scene | |
| M7 | videos and packaging | 4 videos from recorded traces; trace files, seeds, commands | |
| M8 | QA and release | checklist verified; cloud audit; docs; release report | |

## M3 gate evidence (Isaac Sim 5.1 / PhysX unless stated)

**Robots.** NERVA (Open Duck Mini v2), Booster T1, and NVIDIA Nova Carter. Nova Carter replaced the custom wheeled
robot by user decision on 2026-10-19; the custom robot remains only as a MuJoCo cross-physics reference.

| robot | controller | battery (10 seeds × 12 s per condition) | result |
|---|---|---|---|
| NERVA | frozen neutral ONNX (sha 610b1c59…), NERVA's control step and SafetyMonitor | rest, forward, backward, turns, stop/start, head motion, repeated pushes 0.05–0.20 m/s walking and at rest | 0 falls in all nominal conditions and with pushes ≤ 0.15 m/s; repeated 0.20 m/s pushes at rest: 5/10 falls (MuJoCo: the same 5/10) |
| T1 | Isaac Lab PPO v1 (`policies/t1_isaaclab_v1`, sha b14d86f3…) + frozen velocity map | speeds 0–0.5, turns ±0.5/±1, stop/start, repeated pushes 0.1–0.8 m/s walking/turning/standing | 0 falls nominal; pushes ≤ 0.3 m/s 0 falls; 0.5 m/s turning 3/10, 0.8 m/s 2–7/10; with the map, speed error < 0.008 m/s over 0.05–0.20 m/s |
| Nova Carter | the asset's wheel drives, (v, ω) interface | straight, turn, arc, stop, shove 0.6 m/s | speed error 0.002 m/s, turn error 0.023 rad/s, stop 0.10 m, supervisor trips on shove, no fall; signed drive test confirms forward motion |

**NERVA cross-physics** (same battery and stack; MuJoCo = the validated backlash model):

| condition | MuJoCo | Isaac |
|---|---|---|
| rest tilt (95th pct) | 10.9° | 10.5° |
| forward 0.074 m/s | 0.072 | 0.051 |
| backward −0.074 m/s | −0.085 | −0.068 |
| turn ±0.60 rad/s | +0.649 / −0.562 | +0.641 / −0.609 |
| repeated 0.20 m/s pushes at rest | 5/10 falls | 5/10 falls |

**Failed or superseded iterations** (kept for the record):
- The MuJoCo-trained T1 policy falls in PhysX within 0.6 s, even though the kinematics match to 0.07 mm. It was
  replaced by Isaac Lab training.
- T1 v2 (yaw-rate commands, tighter tracking, 0.8 m/s training pushes) failed its battery: a crouched gait below the
  0.45 m trunk threshold, and no tracking at 0.1 m/s.
- The NERVA import needed four fixes: merged defaults, MuJoCo inertials, joint frames rebuilt from MuJoCo, and gains
  in SI units. The backlash joints are removed.

**Frozen from the gates, before any experiment data:**
- the people's lunge surface speed is capped at 0.15 m/s (NERVA's survivable repeated push);
- the T1 velocity map.
