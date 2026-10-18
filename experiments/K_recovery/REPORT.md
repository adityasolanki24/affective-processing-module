# Programme K report: recovering from negative experience (2026-10-18)

**Governed by:**
- `PROTOCOL.md`, frozen at 13b7ef2 before any test data;
- `DEVIATIONS.md` D1 (memory defect: corrective test) and D2 (rerun after a spot preemption);
- `prereg/K6_HUMANOID.md`.

Everything is simulation only.

> **Errata from the programme L audit (2026-10-18; `experiments/L_audit_perception/`).** The tables below are left
> as published. Read them with these corrections:
> 1. **"Handoff" / "task success"** is the stopping-distance proxy `Task.p_success` (now `p_reach`), not a completed
>    handoff. NERVA has no arm. P5 measured navigation arrival, not task completion.
> 2. **Recovery counts.** A lone favourable final meeting was counted as confirmed. Corrected, the counts drop
>    (e.g. NERVA DT_CAM_VOI 50 → 47 of 50); no median or milestone verdict changes.
> 3. **The risk cap was not enforced.** DT, DT_VOI, DT_CAM and DT_UNC approached beyond the predicted cap in 5–29% of
>    decisions. DT_CAM_VOI never did. Results for those systems are re-tested with the cap enforced in L.
> 4. **The "camera" is an abstract sensor** that reports generated lunge events. It is not image processing. Its
>    ±40° bearing test treated a 40° field of view as a half-angle.

## What APM can do now that it could not before

1. **Revise a negative expectation and recover its behaviour, in both directions, within one lifetime, inside the
   safety envelope.**
   - **NERVA:** after a disruptive entity turns calm, the robot walks back into handoff range in 5 meetings with a
     camera, or 8 with its IMU alone.
   - **Wheeled robot:** the same, in 5 meetings.
   - **Calm turning disruptive:** caution returns within 1–2 meetings.
   - No falls in any candidate system.
2. **Gather information safely when it is worth it.**
   - With a camera, an uncertain robot watches the entity from 0.9 m, beyond its reach, before deciding to approach.
   - This cuts contacts from a still-disruptive entity about 3× (NERVA 0.32 vs 0.98 per meeting) and makes pushes
     about 3× weaker.
3. **Express experience through task-appropriate movement**, with a mapping confirmed on fresh data under frozen task
   values.
   - It beats H's fixed mapping in 4 of 4 tasks (cost, contacts, task success).
   - It shows experience-dependent stopping distance only where the task needs closeness.
4. **Run the whole loop continuously**: experience → belief → expression → physical outcome → new evidence →
   revision → recovered behaviour. Two identical NERVA robots in single, never-reset simulations; video and trace.
5. **Run the same learning and decision code, unchanged, on three physically different bodies**: the NERVA biped,
   a MuJoCo wheeled robot and a Booster T1 humanoid on a walking controller trained here. Each has its own fitted
   sensor coding, time model and type model.

## K1: why NERVA could not recover (diagnosis)

Evidence a calm meeting carries (KL(calm‖pushy) per 4 s window, `results/K1_information.json`):

| | IMU at 0.2 m | at 0.4 m | at 0.6 m (where a cautious robot stops) | camera extension, any distance |
|---|---|---|---|---|
| nats | 0.93 | 0.55 | **0.14** | 0.74 |

**Observable versus not:**
- **Observable now:** pushes that move the IMU. At 0.6 m these are weak, so a confident belief needs about 20 calm
  meetings to halve.
- **Simulator-only:** every lunge and its reach.
- **Needs a sensor:** seeing the lunges with a camera; modelled here, not on NERVA today.
- **Needs unsafe contact:** standing within reach.

**A second cause, found during K (D1):** the change-aware memory, in use since programme J, skipped its change step
between consecutive meetings.
- Decisive evidence then drove beliefs to exactly 0 or 1, from which nothing could revise them.
- Fixed (`transition_each_meeting=True`) and re-tested on fresh seeds.
- J's "never revises pushy → calm" was partly this defect.

## K3: recovery on NERVA (1,600 lifetimes per test, 50 per system and history)

Kaplan–Meier medians in B1 meetings after the switch. Primary candidates, frozen in advance: DT_VOI (IMU only) and
DT_CAM_VOI (camera).

| system | K3-X frozen (defective memory): pushy→calm / behaviour / calm→pushy | K3-X2 corrective (fixed memory) | contacts per pushy meeting (X2) | milestone X / X2 |
|---|---|---|---|---|
| L0 (H's mapping) | 6 / 7 / 3, **1 fall**, repeated pushes 26% | 4 / 4 / 2, repeated pushes 28% | 1.79 | — / — |
| DT (J's mapping, frozen task values) | 8 / 11 / 2 | 5 / **8** [7, 8] / 1 | 0.98 | ✗ / ✓ (at the margin; 45/50 recover) |
| DT_VOI (IMU) | 8 / 11 / 2 | 5 / 8 / 1 | 0.97 | ✗ / ✓ (identical to DT: with IMU alone, information is never worth its price) |
| DT_UNC (entropy-triggered) | 6 / 8 / 2 | 4 / 6 / 1 | — | not a candidate |
| DT_CAM | 4 / 5 / 2 | 3 / 5 / 1 | 0.97 | ✓ / ✓ |
| **DT_CAM_VOI** | 5 / 7 / 2 | **3 / 5 / 1** | **0.32** | **✓ / ✓** (all 50 recover) |
| ORACLE (reference) | — / 1 / — | — / 1 / — | — | reference |

**Safety and task:** every candidate system had 0 falls, 0 trips and 100% arrival.

Figures: `figures/K3_recovery_nerva.png`. Results: `results/K3_X.json`, `results/K3_X2.json`.

## K5: the same on a wheeled robot (MuJoCo physics, 1,200 lifetimes per test)

| system | K5-X (frozen) | K5-X2 (fixed) | contacts per pushy meeting (X2) |
|---|---|---|---|
| DT (IMU) | 5 / **9** / 4 ✗ | 4 / 8 / **5** ✗ (calm→pushy too slow) | 0.88 |
| DT_CAM_VOI | 4 / 6 / 2 ✓ | **3 / 5 / 1 ✓** | **0.19** |

The wheeled body senses a distant push better than NERVA (its IMU is quieter), but alone it is still too slow in one
direction. Figures: `figures/K5_recovery_wheeled.png`.

## K3b: frozen-task confirmation of the learned mapping (NERVA, 640 lifetimes)

| task (values fixed in advance) | DT − L0 task cost | contacts | task success L0 → DT | expressive difference DT (risky − safe) |
|---|---|---|---|---|
| handoff (≤ 0.3 m) | −10.8 s [−11.1, −10.4] | −0.11 | 0.52 → 0.77 | 0.29 m |
| inspect (≤ 0.4 m) | −8.4 s | −0.20 | 0.86 → 0.84 | 0.19 m |
| separation (≥ 0.5 m) | −22.0 s | −0.27 | 0.15 → 1.00 | 0 |
| waypoint | −11.2 s | −0.27 | 1.00 → 1.00 | 0 |

**DT replaces L0** (4/4 tasks). Differentiation appears only when the task needs closeness, which is the appropriate
behaviour, not a maximal one.

## K4: continuous two-robot demonstration (NERVA, camera extension, fixed memory)

**Setup.** Two identical robots, each in a single never-reset simulation. 36 encounters each, for example: walk from
home, stop, sense, walk home.
- **Stage 1 (1–10):** A meets a disruptive entity, B a reliable one.
- **Stages 2–3 (11–16):** both meet the disruptive entity.
- **Stages 4–6 (17–36):** it becomes reliable.

**Trace** (`results/K4/K4_demo_trace.json`, `videos/K4_two_robots_continuous.mp4`, `figures/K4_demo.png`):
- A is pushed 17 times over its first 4 approaches, then watches from 0.9 m.
- In stage 2, A takes no contacts while B, approaching, takes 9 and then learns caution.
- After the change, A's belief falls 0.87 → 0.14 in 5 encounters. A returns to 0.40 m, then 0.30 m, and resumes
  handoffs; B follows a meeting later.

**Controls** (10 seeds × 2 robots per condition; demonstration-level, not preregistered;
`results/K4/K4_controls_fixed.jsonl`):

| condition | stage-2 stop A / B | stage-2 contacts A / B | recovers after the change |
|---|---|---|---|
| full | 0.85 / 0.76 m | 1.7 / 6.1 | 10/10 by encounter ≈ 22 |
| no expression | 0.40 / 0.40 | 21.8 / 21.9 | no behaviour change at all |
| no memory update | 0.40 / 0.40 | 20.6 / 21.7 | nothing learned |
| shuffled histories | 0.66 / 0.83 | 10.4 / 2.2 | behaviour follows the memory given, not the body |
| no active information | 0.59 / 0.51 | 8.5 / 12.7 | yes, with about 5× more contacts |

## K6: humanoid (Booster T1, MuJoCo; `prereg/K6_HUMANOID.md`)

**Controller.**
- A walking policy trained here with mujoco_playground's published PPO recipe, with random pushes.
- 202 M steps, 2.1 h on one L4 GPU (`humanoid/trained/`).
- Deployed in plain MuJoCo with NumPy inference.

**Gate G0** (50 encounters): **passes**.
- 0 falls, 100% arrival, 100% of stops within 0.1 m of the request.
- About 2 contacts per pushy encounter, absorbed.

**K6-X** (600 lifetimes, fixed memory, task handoff; `results/K6_X.json`, `figures/K6_recovery_humanoid.png`):

| system | pushy→calm / behaviour / calm→pushy | contacts per pushy meeting | milestone |
|---|---|---|---|
| DT (IMU) | 8 / **12** / 2 | 0.83 | ✗ (its own gait sway masks the evidence) |
| DT_CAM | 3 / 5 / 1 | 0.82 | ✓ |
| **DT_CAM_VOI** | **3 / 6 / 1** | **0.25** | **✓** |
| L0 | degenerate: always stops at 0.6 m, handoff success 2% | 0.85 | — |

- 0 falls and 0 trips in every system.
- L0 fails on this body because the robot's self-calibrated push detector fires in > 90% of windows from gait sway
  alone. L0 reads that detection rate, so its caution saturates.
- The decision-theoretic mapping reads the type belief, so it is unaffected. This is direct evidence that the
  hand-written mapping does not transfer across bodies, while the learned one does.

## Expression capability audit (section 8 of the brief)

**NERVA's validated controller offers:**
- 7 discrete commands at one speed (0.074 m/s), turns at ±0.6 rad/s, a fixed 0.54 s gait clock;
- head offsets only at rest (±0.4 rad);
- torso pitch at 9–30% of reference (E1′).

**Consequences:**
- Tempo can only be pauses.
- Turning dynamics and speed cannot be modulated.
- Repeated sub-envelope pushes can topple it.

**Required:** a motor-control milestone (NERVA M1): continuous (vx, vy, ω) commands plus a repeated-push gate. It is
not an APM problem.

**Evidence that M1 is achievable:** the humanoid controller trained here with random pushes in about 1 h on one GPU.

## Milestones

| milestone | status | evidence |
|---|---|---|
| Experience changes beliefs | **done (sim)** | K3-X2, K5-X2, K6-X; history AUROC 1.00 (J4) |
| Beliefs change physical expression | **done (sim)** | K3b differentiation 0.29 m (handoff); K4 |
| Expression changes future outcomes | **done (sim)** | K4 controls: no expression → 3.6–13× more stage-2 contacts; DT_CAM_VOI 3× fewer contacts than DT |
| New evidence changes beliefs | **done (sim), both directions** | K3-X2: pushy→calm 3–5, calm→pushy 1 |
| Behaviour recovers after conditions change | **done (sim)** with camera on all three bodies; at the margin with NERVA's IMU only | K3-X2, K5-X2, K6-X, K4 |
| Stability and task performance preserved | **done for candidates** (0 falls, 100% arrival); L0 fell once; repeated-push risk remains a controller limitation | K3-X |
| Expression interpretable to humans | **untested** | H2 needs ethics approval |

## What remains unvalidated

- Any hardware.
- The camera extension: modelled, not built or measured.
- Real people.
- Human legibility.
- Recovery when entities are less separable than two types (J2 showed this is where the memory breaks).
- NERVA under repeated pushes (M1).

## Compute

About US$11 estimated (upper bound US$14) of the US$40 cap (`COMPUTE_LOG.md`). Final cloud audit: nothing left running.
