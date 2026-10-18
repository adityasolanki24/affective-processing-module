# Project state

*The single authoritative account of what APM is and does today. Updated 2026-10-18, after programme J; programme K
(recovery from negative experience) is in progress on branch `research/apm-v0.7-recovery`.*

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
| experience memory | EM-learned entity types + change-aware belief (`TypeBelief`) | `apm_revise.belief`, `apm_transfer.nerva_memory` | I1, J1; J4 on NERVA |
| belief → expression | H's fixed lookup **L0** (kept by J's decision E) | `apm_express.states.lookup_intent` | H4b, J5 |
| candidate replacement | decision-theoretic mapping (5-parameter learned time model + explicit task values) | `apm_transfer.expression_map` | J6, **exploratory only** |
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

Nothing has run on hardware.

## Milestones toward the objective

The objective: a robot that learns from experience and expresses learned internal states through its own action,
with the loop closed in both directions.

| milestone | status | evidence |
|---|---|---|
| Experience changes beliefs | **done (sim)** | J4 C1 (AUROC 1.00, NERVA IMU only); J5 Q1 |
| Beliefs change physical expression | **done (sim)**, small with L0 | H4b; J5 Q2 (0.04 m); J6 0.26 m (exploratory) |
| Expression changes future outcomes | **partial** | J5 Q3 primary failed; J6-X CL −0.009 (exploratory); J6 learned mapping −0.027 (exploratory) |
| New evidence changes beliefs | **one direction only** | calm → pushy in 2–4 meetings; pushy → calm essentially never (J4 C3, J5 Q4, J6) |
| Behaviour recovers after conditions change | **not demonstrated** | the avoidance trap (J5, J6) |
| Stability and task performance preserved | **yes, with a known gap** | 0 falls in confirmatory systems; repeated sub-envelope pushes caused 2 falls in an ablation (J5) |
| Expression interpretable to humans | **untested** | H2 study needs ethics approval; an informal friends' survey is running and is not study data |

## What is demonstrated (simulation)

- Fixed-type experience memory learns, without labels, from the robot's own sensors: near-miss clearance (wheeled)
  and IMU acceleration (NERVA, with an efference-copy gate and sensing pauses).
- On the wheeled testbed it revises within 2–3 episodes and halves avoidable collisions. The safety gain comes from
  holding discriminating beliefs, not from revision (J1).
- Identical robots with different histories move measurably differently in the same situation (H4b: history
  recoverable from motion in 98–100% of cases).
- Memory state survives save/restore exactly (J4 C4).

## Failed hypotheses (do not revisit without new evidence)

- PAD / emotion-like affect state improves behaviour: **harmful** under equal tuning (G).
- Recurrent or specialised neural regulators/mappings beat explicit memory plus a simple mapping: **no** (G, H, J4,
  J6).
- Memory is a universal default under distribution shift: **no**. It fails with ≥ 25% behavioural overlap between
  types and with heavy sensor noise (J2).
- H's fixed L0 closes the loop on NERVA: **no** (J5). Its caution styles lengthen exposure.

## Active limitations and blockers

1. **Recovery from negative experience** (programme K's target).
   - After bad experience the robot keeps its distance.
   - At that distance its only sensor (IMU contact evidence) barely distinguishes a changed entity from an unchanged
     one.
2. **NERVA's motion vocabulary is narrow.** Seven fixed walking commands at one speed, plus head and torso offsets.
   Tempo is only pause timing.
3. **Repeated sub-envelope perturbations can topple NERVA.** The validated envelope covers single pushes only. This is
   a motor-control limitation, not an APM problem.
4. **The learned expression mapping is exploratory.** Its task value was chosen after inspecting behaviour (J
   deviation D3).
5. **No human evidence** on legibility, and no hardware.

## Latest results

Programme J, 2026-10-18 (`experiments/J_validation_transfer/reports/FINAL_REPORT.md`).
- **Decision E:** keep explicit memory + L0; the learned decision-theoretic mapping is the candidate.
- Paired videos: `experiments/J_validation_transfer/videos/`.

## What happens next

Programme K, in this order ([ROADMAP.md](ROADMAP.md)):
1. Diagnose what a distanced NERVA can and cannot observe about a changed entity.
2. Implement the smallest safe information-gathering mechanism.
3. Confirm it, and the learned mapping with task values frozen in advance, on fresh seeds.
4. One continuous two-robot demonstration of experience → belief → expression → outcome → revision → recovery.
5. The same learning loop on a second, different embodiment.
