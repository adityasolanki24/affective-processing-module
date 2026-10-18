# Project state

*The single authoritative account of what APM is and does today. Updated 2026-10-18, after programme K (branch
`research/apm-v0.7-recovery`, not merged; report: `experiments/K_recovery/REPORT.md`).*

Everything below is **simulation only**. No hardware has been tested, and no human has evaluated the robot's
expression.

## Current architecture (one sentence)

A robot observes encounters through its own sensors. An **explicit, revisable memory** turns them into a belief about
each entity (a learned latent type, with change-aware forgetting). A **transparent mapping** turns beliefs into a
bounded **expressive intent**. An **embodiment adapter** realises that intent on a validated motion controller, and
an **independent safety monitor** can always override it. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

**Which implementation is current**

| layer | current choice | code | evidence |
|---|---|---|---|
| world/consequence prediction (wheeled) | small MLP ensemble + EMA risk filter (decision A) | `apm_select`, `apm_research` | G (G9 validity re-test) |
| experience memory | EM-learned entity types + change-aware belief, with one change step per meeting (`transition_each_meeting=True`, K deviation D1) | `apm_revise.belief`, `apm_transfer.nerva_memory` | I1, J1, J4; K3-X2 |
| belief → expression | **decision-theoretic mapping**: argmin of task cost (the robot's own learned time model + task values fixed from requirements + a safety risk cap) | `apm_transfer.expression_map`, `apm_transfer.infoseek` | **K3b: beats L0 in 4/4 frozen tasks** (replaces H's L0) |
| information gathering | non-myopic value of information; with a camera, an OBSERVE action from beyond reach | `apm_transfer.infoseek` | K3-X2, K5-X2: fastest recovery, ~3× fewer contacts. Needs a sensor that sees the entity: modelled, not on NERVA |
| embodiment adapters | NERVA biped (validated); kinematic wheeled base | `apm_express.adapter`, `apm_express.wheeled` | H0b, H1b, H7 |
| safety | `apm_express.safety.SafetyMonitor` + neutral fallback; wheeled supervisor | `apm_express.safety`, `apm_testbeds.wheeled_ctrl` | H1b, G |

The v0.1 runtime in `src/apm` (AffectLink protocol, PAD affect core) still exists and is tested. Its **PAD layer
was shown to be harmful** (programme G) and is not part of the current architecture. AffectLink itself stays as the
robot interface contract ([affectlink.md](affectlink.md)).

## Robots available

| robot | what it is | physics | controller |
|---|---|---|---|
| **NERVA** (Open Duck Mini v2 biped) | MuJoCo model from the NERVA repository | full rigid-body physics | frozen validated neutral ONNX policy: 7 commands (rest, ±x, ±y at 0.074 m/s, turns ±0.6 rad/s), 0.54 s gait clock; head offsets and torso pitch inside a measured envelope |
| wheeled testbed | 2D unicycle, 6 m arena, pedestrians with hidden dispositions | 2D kinematics with slip | reactive navigation + bounded high-level actions + safety brake |
| kinematic wheeled base | H7's second expressive embodiment | kinematic only | (v, ω) tracking with limits |
| **MuJoCo wheeled robot** (programme K) | 2.9 kg differential drive, casters, chassis IMU | full physics | wheel-speed (v, ω) tracking + shove/tilt supervisor |
| **Booster T1 humanoid** (programme K6) | MuJoCo model from mujoco_playground | full physics | walking policy trained here with pushes (2.1 h, one L4); gate G0 passed: 0 falls, 100% arrival |

Nothing has run on hardware.

## Milestones toward the objective

The objective: a robot that learns from experience and expresses learned internal states through its own action,
with the loop closed in both directions.

| milestone | status | evidence |
|---|---|---|
| Experience changes beliefs | **done (sim)** | J4 (AUROC 1.00, IMU only); K3-X2 |
| Beliefs change physical expression | **done (sim)** | K3b (0.29 m where the task needs closeness, 0 where it does not); K4 |
| Expression changes future outcomes | **done (sim)** | K4 controls (no expression: 3.6–13× more contacts); K3-X2 (information-seeking: 3× fewer contacts) |
| New evidence changes beliefs | **done (sim), both directions** | K3-X2: pushy→calm in 3–5 meetings, calm→pushy in 1 |
| Behaviour recovers after conditions change | **done (sim)** on NERVA, a wheeled robot and a humanoid: reliably with a camera, at the margin with NERVA's IMU alone | K3-X2, K5-X2, K6-X, K4 continuous video |
| Stability and task performance preserved | **done for the candidates** (0 falls, 100% arrival); repeated pushes remain a NERVA controller limit | K3-X, K3-X2 |
| Expression interpretable to humans | **untested** | H2 needs an ethics determination |

## What is demonstrated (simulation)

- A robot learns, from its own sensors and without labels, which entities disrupt it, and changes how it approaches.
- It revises that expectation when the entity changes, in both directions, and its behaviour recovers. On NERVA:
  5 meetings with a camera, 8 with the IMU alone.
- With a camera it gathers information safely, by watching from beyond reach before approaching.
- The whole loop runs continuously in one lifetime: `experiments/K_recovery/videos/K4_two_robots_continuous.mp4`.
- The same memory and decision code works on three bodies (biped, wheeled, humanoid), each with its own fitted
  models. H's hand-written mapping does not transfer: it saturates on the humanoid.

## Failed hypotheses (do not revisit without new evidence)

- PAD / emotion-like affect state improves behaviour: **harmful** under equal tuning (G).
- Recurrent or specialised neural regulators/mappings beat explicit memory plus a simple mapping: **no** (G, H, J4,
  J6).
- Memory is a universal default under distribution shift: **no**. It fails with ≥ 25% behavioural overlap between
  types and with heavy sensor noise (J2).
- H's fixed L0 closes the loop on NERVA: **no** (J5). Its caution styles lengthen exposure; replaced by the
  decision-theoretic mapping (K3b).
- A myopic value-of-information rule escapes the avoidance trap: **no**. It values every observation at zero once
  the robot is confident (K, found before data).
- Value of information helps with IMU-only sensing on NERVA: **no** (K3: identical to plain DT).

## Active limitations and blockers

1. **NERVA lacks a sensor that sees other agents' motion.** Its IMU alone recovers only at the margin. The camera
   that makes recovery reliable is modelled, not built.
2. **NERVA's motion vocabulary is narrow, and repeated sub-envelope pushes can topple it.** This is a motor-control
   milestone (NERVA M1), not an APM problem.
3. **Two-type worlds only.** With overlapping behaviours, the memory degrades (J2).
4. **No human evidence** on legibility, and no hardware.

## Latest results

Programme K, 2026-10-18 (`experiments/K_recovery/REPORT.md`): recovery demonstrated on NERVA and a wheeled robot, the
learned mapping confirmed, a continuous two-robot demonstration, and a memory defect found and fixed (D1).

## What happens next

See [ROADMAP.md](ROADMAP.md):
1. a real exteroceptive sensor on NERVA;
2. NERVA motor milestone M1;
3. a human legibility study, after ethics approval;
4. migrating the validated components into a runtime behind AffectLink.
