# Project state

*The single authoritative account of what APM is and does today. Updated 2026-10-18, after programme L (branch
`research/apm-v0.8-audit`, not merged; report: `experiments/L_audit_perception/REPORT.md`). Programme L audited and
corrected programme K; where they differ, L governs.*

Everything below is **simulation only**. No hardware has been tested, and no human has evaluated the robot's
expression.

## Current architecture (one sentence)

1. The robot senses encounters with its **IMU and a depth camera**.
2. An **explicit, revisable memory** turns them into a belief about each entity: a learned latent type, with
   change-aware forgetting.
3. A **decision-theoretic mapping** with an **enforced risk cap** turns beliefs into a bounded **expressive
   intent**.
4. An **embodiment adapter** realises the intent on a validated motion controller.
5. An **independent safety layer** can always override it.

All of it runs in one **integrated runtime** that speaks AffectLink. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

**Which implementation is current**

| layer | current choice | code | evidence |
|---|---|---|---|
| sensing | IMU (efference-gated, self-calibrated) + **rendered depth camera with motion perception** (no simulator state) | `apm_transfer.nerva_world`, `apm_transfer.vision` | L C.1: gate passed on NERVA and wheeled; failed on the humanoid (false events) |
| experience memory | EM-learned entity types + change-aware belief, one change step per meeting, exact save/restore | `apm_transfer.nerva_memory` | K3-X2; L A6, B, D (alternatives not better) |
| belief → expression | decision-theoretic mapping with task values fixed from requirements and an **enforced predicted-risk cap** (HOLD when nothing is safe) | `apm_transfer.infoseek` | K3b; L LB4; **L E: beats H's lookup, a retuned lookup and a tuned monotonic rule in 4/4 tasks** |
| information gathering | non-myopic value of information, OBSERVE from beyond reach (needs the camera) | `apm_transfer.infoseek` | L C5 survives (NERVA 0.38× contacts); no gain in the G continuous demo |
| runtime | `Runtime`: observation interface, persistence, deterministic replay, independent supervisor, body adapters, AffectLink v1 | `apm_transfer.runtime` | L F1–F6 on NERVA, wheeled and humanoid |
| embodiment adapters, safety | NERVA (validated), MuJoCo wheeled, Booster T1 humanoid | `apm_express`, `apm_transfer.*_world` | H, K, L A7 |

The v0.1 runtime in `src/apm` still exists. Its AffectLink protocol is the contract the new runtime speaks; its PAD
core is superseded (programme G).

## Robots available

| robot | physics | controller | notes |
|---|---|---|---|
| **NERVA** (Open Duck Mini v2 biped) | full | frozen validated neutral ONNX policy, 7 commands | rare falls under repeated sub-envelope pushes (L C.3: one seed, first meeting) |
| **MuJoCo wheeled robot** | full | (v, ω) tracking + shove/tilt supervisor | |
| **Booster T1 humanoid** | full | walking policy trained here | NumPy policy = brax (L A7); no falls ≤ 0.3 m/s repeated pushes; falls from 0.5 m/s |
| wheeled testbed, kinematic base | 2D / kinematic | | programmes F–H |

## Milestones toward the objective

| milestone | status | evidence |
|---|---|---|
| Experience changes beliefs | **done (sim)** | J4, K, L |
| Beliefs change physical expression | **done (sim)** | K3b, L LB4, L G1 (matched situation: 0.11 m farther, 0.9 fewer contacts) |
| Expression changes future outcomes | **done (sim)** | L G4 (no expression: no protection); L D (memory halves contacts per risky meeting) |
| New evidence changes beliefs, both directions | **done (sim)** | L B: camera 1–2 meetings each way |
| Behaviour recovers after change | **done with the abstract camera on 3 bodies; with real rendered vision on NERVA (7 meetings), not yet on the wheeled robot (11)** | L B, C.3, G2 |
| Safety constraint enforced | **done** (0 over-cap decisions; independent supervisor) | L A3, B, F5 |
| Integrated runtime across bodies | **done (sim)** | L F |
| Effective under stress | **partly**: contacts always reduced; the task-cost gain is significant in 6/15 scenarios | L D |
| Expression interpretable to humans | **untested** | H2 needs an ethics determination |

## Failed hypotheses (do not revisit without new evidence)

- PAD / emotion-like affect state improves behaviour: **harmful** (G).
- Recurrent or specialised neural mappings beat explicit memory plus a simple mapping: **no** (G, H, J4, J6).
- Memory is a universal default under distribution shift: **no** (J2). Its *cost* advantage under stress is small
  (L D).
- H's fixed lookup closes the loop on NERVA: **no** (J5, K3b, L LB4).
- A recency-weighted or context-keyed memory beats the Bayesian type memory: **no** (L D).
- Value of information always helps: **no**. It helps with a camera in single encounters (L C5), but not in the
  continuous NERVA demonstration (L G5).
- The abstract camera's wheeled recovery holds with real perception: **no** (L C.3).
- A calibrated simple rule (retuned lookup or monotonic rule) matches the decision-theoretic mapping: **no** (L E).

## Active limitations

1. **Real vision.** Lunges cannot be seen at the hand-over distance. Real vision recovers slowly on the wheeled
   robot, and the humanoid's sway produces false events.
2. **Enforced cap without a distance sensor.** An IMU-only robot holds beyond reach and recovers slowly, or never.
3. **NERVA motor limits** under repeated pushes (NERVA M1).
4. **No human evidence, no hardware.**

## Latest results

Programme L (`experiments/L_audit_perception/REPORT.md`):
- the audit fixes;
- 6 of 7 K claims survive;
- a genuine vision pipeline;
- stress tests;
- the integrated runtime;
- a continuous two-robot demonstration with rendered vision.

## What happens next

See [ROADMAP.md](ROADMAP.md):
1. vision robust at close range and to egomotion;
2. a real camera on NERVA;
3. NERVA M1;
4. human legibility (ethics);
5. the GRU comparison deferred from L D.
