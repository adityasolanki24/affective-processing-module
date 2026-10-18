# Programme L report: audit corrections, real perception, next milestone (2026-10-18)

**Governed by:**
- `PROTOCOL.md`, append-only. Each section was frozen by commit before its data:
  - §B at ffdc691; §C at a0e5c6d; §D at fe8ae67; §E at 6f18eb0; §G at 2c30c54;
  - §F, an engineering verification, was written after the demos (LD3).
- `DEVIATIONS.md` (LD1–LD6).

Programme K's results are preserved unchanged. Its report carries an errata block that points here.

Everything is **simulation only**. No hardware was used, nobody was recruited, and no ratings were produced.

## Answers

1. **Did the corrected recovery claims survive?**
   - Mostly: 6 of K's 7 claims survive with every audit correction applied together on fresh seeds.
   - The 7th (C7) fails. Caution returns within 1–2 meetings with the camera on every body, but IMU-only on the
     wheeled robot needs 5. K's own K5 data already showed this, so K over-generalised that claim.
   - Applied to K's saved data, the corrections remove some recovered lifetimes but change no median or milestone
     verdict.
2. **Is there a genuine vision pipeline, and does recovery work with noisy observations?**
   - **Yes, there is a genuine pipeline.** Images are rendered from a depth eye mounted on the robot. A
     foreground-motion detector reads only pixels, the robot's own-motion flags and its navigation distance.
   - It passes the pre-registered perception gate on NERVA and the wheeled robot: recall 0.78/0.82, 0.07/0.06
     false events per window, reach error about 3 cm.
   - It fails the gate on the humanoid: 0.31 false events per window, from standing sway.
   - **Recovery with real vision is partly achieved:**
     - NERVA: IMU + vision meets P1–P3 (pushy → calm revision in 5 meetings, recovery in 7, calm → pushy in 2;
       48 of 50 recover). Every NERVA system fails P4 on the same first-meeting fall.
     - Wheeled robot: real vision recovers too slowly (11 meetings, 27 of 50), where the abstract camera
       succeeds. **K's wheeled camera result depended on the idealised sensor.**
3. **Is the system effective under changed entity behaviour?**
   - **No, by the pre-registered task-cost criterion:** memory beats no memory on task cost in 6 of 15 stress
     scenarios.
   - **Yes on safety:** memory cuts contacts per risky meeting by 0.4–1.2 per meeting (often about half) in every
     scenario that has risky meetings. That includes drift, erratic switching, occlusion, a poor camera, tracking
     errors, speed changes and context dependence.
   - The cost advantage is small (0.2–1.4 s per meeting) and is often within noise.
4. **Is the safety constraint enforced?**
   - **Yes.** With no option inside the predicted-risk cap, the robot now executes an explicit
     `NO_SAFE_ACTION` → HOLD.
   - A second leak was found and fixed: the entropy-triggered policy checked the cap for one window and then
     executed three.
   - Every corrected system made 0 over-cap approaches in 8,000+ decisions per system. On the same seeds, K's
     behaviour made 5.2%.
   - An independent supervisor in the runtime re-checks every decision; it is unit-tested to veto all over-cap
     approaches.
   - Predicted risk is not physical safety. Falls still happen when the controller's limits are exceeded.
5. **Is the expressive controller better than a calibrated simple baseline?**
   - **Yes.** The decision-theoretic controller beats H's lookup, a retuned lookup and a tuned monotonic rule in
     4 of 4 tasks, on cost and on contacts. Both baselines had equal tuning budgets: 12 configurations per task.
   - Against the best simple baseline (the monotonic rule) it saves 1.4–2.3 s per encounter and 0.10–0.17
     contacts.
   - Adding value of information is equivalent in two tasks; there, the simpler controller (no VOI) is
     recommended.
   - VOI's large wins in the waypoint and separation tasks are an artefact of K's task definitions (below).
6. **Is there an integrated runtime, and does it work across bodies?**
   - **Yes:** `src/apm_transfer/runtime.py`, an executable demo, and checks F1–F6 passing on NERVA, the wheeled
     robot and the T1 humanoid.
   - What the checks cover: restart persistence (bit-identical), exact replay, valid AffectLink v1 messages, cap
     compliance and bounded intents.
7. **What is missing for human-readable expression?** See *What is missing* below.

## Pass/fail per phase

| phase | result |
|---|---|
| A audit fixes | **pass**: all 7 findings reproduced and fixed, with tests. A3 found a second cap leak. |
| B corrective test | **pass** for C1–C6 (survive); **C7 fails** |
| C vision | perception gate: **pass** on NERVA and wheeled, **fail** on the humanoid (G2). Recovery with real vision: **not met** (Q1 and Q2 false on both bodies; NERVA misses only on a shared fall) |
| D stress | **fail** on the pre-registered cost criterion (6/15); contacts reduced in every risky scenario. Memory choice: M1 stays. GRU trigger fired, GRU not run (LD2) |
| E controllers | **pass**: E3 (and E4) beat every calibrated simple baseline in 4/4 tasks; E5 not justified |
| F runtime | **pass** (F1–F6 on 3 bodies) |
| G continuous demo | G1, G2, G4 **pass**; G3 fails mechanically on a < 1 mm difference; G5 **fails** (information seeking did not speed recovery) |

## Corrected claims (programme K)

Verdicts are §B.4's mechanical rules on LB1–LB4: fresh seeds, every correction, recovery target calibrated on V.

| K claim | verdict | corrected numbers (medians: pushy→calm / recovery / calm→pushy, B1 meetings) |
|---|---|---|
| C1 NERVA camera + VOI meets the recovery milestone | **survives** | 2 / 4 / 2; 50/50 recover; 0.31 contacts per pushy meeting |
| C2 NERVA IMU only meets it (at the margin) | **survives**, at the margin | 5 / 8 / 2; 41/50 recover |
| C3 wheeled: camera meets it, IMU only does not | **survives** | camera 4 / 6 / 2 (49/50); IMU only: 3/50 recover |
| C4 humanoid: same | **survives** | camera 3 / 6 / 1 (50/50); IMU only: 1/50 recover |
| C5 information gathering cuts contacts | **survives** | NERVA ratio 0.38 [0.35, 0.42] |
| C6 decision-theoretic mapping beats H's lookup | **survives** | 4/4 tasks with measured task success (LB4); handoff completed 78% vs 0% |
| C7 caution returns within 1–2 meetings | **fails** | camera candidates 1–2 on all bodies; wheeled IMU only: 5 |

**Why IMU only fails on wheels and the humanoid but passes on NERVA.**
- With the cap enforced, a confident IMU-only robot HOLDs beyond reach, where IMU silence carries no information.
  It can then come back only through slow forgetting.
- On the wheeled robot and the humanoid it holds in 51–60% of B1 meetings. On NERVA, whose IMU belief rarely
  becomes that confident, it holds in only 11%.
- K's IMU-only recoveries partly relied on approaching beyond the cap. The paired legacy reference DT_K made 5.2%
  over-cap decisions and had 0.97 contacts per pushy meeting, against 0.80 when corrected.

**Task naming (A4).**
- "Handoff" in K was a stopping-distance proxy.
- L measures three things separately: navigation arrival, reaching the interaction distance, and *holding* it for a
  full sensing window. A HOLD means the task was not attempted.
- No handoff or manipulation is claimed: NERVA has no arm.

**Reanalysis of K's saved data** (A1, A2; `results/A_reanalysis_K.json`):
- Recovered counts drop, e.g. NERVA DT_CAM_VOI 50 → 47 and humanoid DT 20 → 14.
- No median and no milestone verdict changes.
- The task-derived target gives the same verdicts as K's test-partition target.

## Phase A: audit findings reproduced and fixed

| finding | fix | evidence |
|---|---|---|
| A1: a lone favourable final meeting counted as a confirmed event | confirmed onset needs 2 consecutive favourable observations (`recovery_metrics.py`); `--legacy-last-meeting` reproduces K | tests; reanalysis |
| A2: recovery target computed on the test partition | V-calibrated targets frozen before X, plus a task-derived criterion | `configs/LB_recovery_targets.json`, `LC_recovery_targets.json` |
| A3: risk cap not enforced (K approached anyway) | explicit `NO_SAFE_ACTION` → HOLD; per-encounter check also in the entropy-triggered policy; legacy flag | adversarial tests; K's DT / DT_CAM / DT_UNC exceeded the cap in 5–29% of decisions, DT_CAM_VOI never |
| A4: "task success" was a proxy | `p_reach` (model); measured interaction hold; no handoff claim | row fields `reached_interaction`, `hold_completed` |
| A5: the "camera" is not vision; FOV used as a half-angle | `Camera` documented as an abstract event sensor; 40° full angle; `Camera.legacy()` | tests; all worlds use `within_fov` |
| A6: memory under stress | λ applied exactly once per encounter (global clock, including empty encounters); exact save/restore; both directions; optional log-space accumulation (equal to 1e-12; survives 400 decisive observations); legacy clock reproduces D1 | `tests/test_l_audit.py` |
| A7: humanoid controller | NumPy policy = brax inference (max \|Δaction\| 7e-7 over 2,000 rollout states); weights = training pickle array by array; model hash matches; envelope measured (below) | `results/A7_*.json` |

**Humanoid envelope** (simulation, 20 seeds per condition, supervisor on; not a certification):
- **No falls** at 0–0.5 m/s, in turns up to 1 rad/s, in stop/start, from perturbed initial joints, with 2× the
  training observation noise, and under repeated pushes ≤ 0.3 m/s (3× K's entity cap) while walking, turning or
  standing.
- **Falls begin** under repeated 0.5 m/s pushes (turning 4/20, standing 2/20) and dominate at ≥ 0.8 m/s.
- **Single pushes at the training interval:** 0/20 falls at 0.5, 2/20 at 0.8 and 4/20 at 1.0 m/s.

## Phase C: rendered perception

Pipeline: `src/apm_transfer/vision.py`.
- **World side (render only; physics unchanged).** The entity is a person-scale column. It physically slides toward
  the robot through each lunge, and the move peaks at the step where the impulse model applies any push.
- **The eye.** A 64 × 48 depth camera with a 40° field of view on the robot (NERVA head, wheeled mast, T1 head),
  pitched 10° down. It renders at 10 Hz during sensing windows, with noise and dropouts.
- **Perception (`MotionPerception`).** It sees only depth frames, the efference copy, the navigation distance and
  its own camera specification. Method:
  - a median background;
  - a noise-scaled threshold;
  - spatial coherence;
  - egomotion rejection;
  - foreground counted only at the target surface;
  - a one-frame gap tolerance;
  - reach = navigation distance − eye offset − closest foreground.
- Unit tests check its interface.

**C.1 metrics** (400 encounters per body, each paired with the abstract camera on identical physics):

| body | recall (visible lunges) | false events / window | reach MAE | KL(calm‖pushy) at 0.3 / 0.4 / 0.6 / 0.9 m: vision vs abstract | gate |
|---|---|---|---|---|---|
| NERVA | 0.78 | 0.07 | 3.0 cm | 0.45 / 0.49 / 0.94 / 0.66 vs 0.46 / 0.72 / 0.71 / 0.63 | **pass** |
| wheeled | 0.82 | 0.06 | 3.0 cm | 0.00 / 0.55 / 0.94 / 0.79 vs 0.83 / 0.68 / 0.87 / 0.61 | **pass** |
| humanoid | 0.84 | **0.31** | 3.1 cm | 0.03 / 0.34 / 0.48 / 0.89 vs 0.69 / 0.67 / 0.62 / 0.91 | **fail (G2)** |

**What real vision cannot do that the abstract camera did.**
- **Near the entity, a camera sees nothing useful.** At a 0.3 m hand-over stop on the wheeled robot or the humanoid,
  a lunge cannot visibly move before it touches the robot, so the information is ≈ 0. The abstract camera reported
  the lunge's full reach regardless.
- **Lunges that reach the robot look as long as the gap, no longer.**
- **Close lunges merge into one event.**

**C.3 recovery** (fresh seeds; V-calibrated targets; vision type models fitted on LC-T):

| body | system | pushy→calm / recovery / calm→pushy | recovered | contacts / pushy meeting | milestone |
|---|---|---|---|---|---|
| NERVA | IMU only | 5 / 8 / 2 | 38/50 | 0.83 | P4 ✗ (falls) |
| NERVA | IMU + abstract camera | 2 / 4 / 1 | 49/50 | 0.37 | P4 ✗ (falls) |
| NERVA | **IMU + rendered vision** | **5 / 7 / 2** | **48/50** | **0.44** | P1–P3 ✓, P4 ✗ (falls) |
| NERVA | rendered vision only | 4 / 8 / 7 | 38/50 | 0.76 | P3 ✗ |
| wheeled | IMU only | – / – / 4 | 4/50 | 0.37 | ✗ |
| wheeled | IMU + abstract camera | 3 / 5 / 2 | 50/50 | 0.22 | P1–P3 ✓ |
| wheeled | **IMU + rendered vision** | **6 / 11 / 6** | **27/50** | **0.46** | P2, P3 ✗ |
| wheeled | rendered vision only | 1 / 1 / never | 50/50 | 1.58 | P3 ✗ (never becomes cautious) |

- **The NERVA falls** are one seed at the first meeting with the pushy entity, the same for every system, before
  anything can have been learned (LD6).
- **On NERVA, real vision is close to the abstract camera:**
  - recovery 7 vs 4 meetings; paired difference +3.0 [2.3, 3.7];
  - contacts +0.08 [0.04, 0.12] per pushy meeting.
  - It is far better than the IMU alone (0.44 vs 0.83 contacts).
- **On the wheeled robot it is not.** Recovery is slower, at 11 meetings, and caution returns in 6.

## Phase D: stress scenarios

Setup: the wheeled robot, IMU + vision; 30 seeds × 2 histories per scenario; `results/D_stress.json`,
`figures/D_stress.png`.

| scenario | M1 − M0 cost (s) | M1 − M0 contacts per risky meeting | D1 |
|---|---|---|---|
| nominal | −0.39 [−0.91, 0.14] | −0.86 | ✗ |
| overlapping types | −1.42 [−1.96, −0.88] | −0.40 | ✓ |
| slow drift | −1.00 [−1.46, −0.55] | −0.84 | ✓ |
| erratic | −0.18 [−0.79, 0.38] | −0.77 | ✗ |
| positions | −0.22 [−0.88, 0.44] | −0.98 | ✗ |
| occlusion | −0.95 [−1.53, −0.40] | −1.08 | ✓ |
| 6 identities | +0.30 [−0.16, 0.77] | −0.60 | ✗ |
| tracking errors | −0.23 [−0.74, 0.31] | −0.74 | ✗ |
| fast lunges | −0.60 [−1.21, 0.03] | −1.00 | ✗ |
| slow lunges | −0.94 [−1.49, −0.39] | −1.00 | ✓ |
| poor camera | −0.61 [−1.16, −0.06] | −0.87 | ✓ |
| context-dependent | −1.18 [−1.68, −0.70] | −0.78 | ✓ |
| goal: inspect | +0.14 [−0.38, 0.68] | −1.19 | ✗ |
| goal: separation | +0.06 | −0.01 | ✗ (nothing to gain) |
| goal: waypoint | +0.05 | −0.02 | ✗ (nothing to gain) |

**Memories.**
- The leaky recency-weighted memory (M2) is cheaper than M1 in 2 scenarios (inspect, slow lunges) and costlier in
  none: short of the 3 needed.
- The context-keyed memory (M3) is never better, not even in the context scenario, and is worse with 4 positions.
- **M1 stays.**
- The GRU trigger fired; the GRU was not run (LD2).

## Phase E: expression controllers

Setup:
- NERVA with IMU + rendered vision, the M1 memory and every correction;
- 30 seeds × 4 histories per task;
- E1 and E2 tuned on LE-T, 12 configurations each per task;
- results in `results/E_controllers.json`.

Cost (s per encounter) / contacts per encounter:

| task | E0 H's lookup | E1 retuned lookup | E2 monotonic rule | **E3 decision-theoretic** | E4 E3 + VOI | recommended |
|---|---|---|---|---|---|---|
| handoff | 31.4 / 0.46 | 27.0 / 0.23 | 14.3 / 0.35 | **12.2 / 0.23** | 12.0 / 0.21 | **E3** (E4 within 1 s) |
| inspect | 22.1 / 0.46 | 20.1 / 0.64 | 13.3 / 0.32 | **11.0 / 0.15** | 10.6 / 0.12 | **E3** |
| separation | 29.7 / 0.46 | 14.4 / 0.23 | 7.2 / 0.18 | 5.7 / 0.07 | 0.25 / 0.00 | E4 (mechanical; artefact) |
| waypoint | 18.7 / 0.46 | 14.4 / 0.23 | 7.2 / 0.18 | 5.8 / 0.08 | 0.26 / 0.00 | E4 (mechanical; artefact) |

**E3 against the best calibrated baseline (E2), paired 95% CIs:**

| task | cost | contacts |
|---|---|---|
| handoff | −2.14 [−2.47, −1.81] | −0.12 [−0.15, −0.09] |
| inspect | −2.31 [−2.62, −2.01] | −0.17 [−0.20, −0.14] |
| separation | −1.48 [−1.75, −1.22] | −0.11 [−0.13, −0.09] |
| waypoint | −1.41 [−1.67, −1.16] | −0.10 [−0.12, −0.08] |

**Measured handoff hold completion:** E0 0%, E1 0%, E2 73%, E3 77%, E4 76%. Retuning H's lookup does not make it
complete the task.

**The E4 artefact.**
- For waypoint and separation, E4 always chooses OBSERVE. Its stand-off point (0.9 m) is the robot's *start*
  position, so the robot never moves.
- K's frozen definitions of those two tasks are satisfied by not moving: a waypoint "beside the entity" with
  proximity irrelevant, and a separation ≥ 0.5 m.
- So E4's near-zero cost there measures a flaw in the task specification, not better expression. The answer to the
  question rests on E3, which wins every task without it.

**Falls (NERVA's controller limit):** E3 and E4 had 2 each in handoff; E1 had 1 and E2 had 2 in inspect.

## Phase F: integrated runtime

`src/apm_transfer/runtime.py`; demo `scripts/f_runtime_demo.py`; results `results/F_runtime_*.json`.

| check | NERVA (IMU + vision) | wheeled (IMU + vision) | humanoid (IMU + abstract camera) |
|---|---|---|---|
| F1 same code runs | ✓ | ✓ | ✓ |
| F2 restart (save → delete → load) gives identical decisions | ✓ | ✓ | ✓ |
| F3 exact replay from the log | ✓ 24/24 | ✓ 24/24 | ✓ 24/24 |
| F4 AffectLink v1 messages valid and round-trip | ✓ 72/72 | ✓ 72/72 | ✓ 72/72 |
| F5 no approach above the cap; supervisor vetoes | ✓ (max 0.298 vs 0.30; 0 vetoes) | ✓ (0.294; 0) | ✓ |
| F6 bounded intents | ✓ | ✓ | ✓ |

**Parts:**
- a robot-independent `Observation`;
- persistent Bayesian type memory, with atomic save;
- deterministic per-decision random streams;
- VOI intent bounded to the grid, OBSERVE or HOLD;
- an independent `SafetySupervisor` with its own risk model;
- body adapters that use each body's own expression adapter and safety monitor;
- a JSONL log and replay;
- AffectLink v1: `EntityObservation` in; `MemoryExpectation`, `ActionTendency` and `ExpressionIntent` out.

## Phase G: continuous demonstration on the runtime

Two identical NERVA robots, IMU + rendered vision, the corrected runtime, one never-reset lifetime each.

| condition | stage 2 stop, A − B | stage 2 contacts, A − B | robots recovering after the change (median encounters) |
|---|---|---|---|
| FULL | **+0.11 m [0.05, 0.16]** | **−0.90 [−1.43, −0.25]** | A 10/10 (8.5), B 10/10 (9) |
| no memory | 0.000 m | +0.43 [−0.22, 1.07] | trivial: it never becomes cautious |
| no expression | 0.000 m | −0.05 [−0.58, 0.47] | 0/10: never reaches the hand-over distance |
| no information seeking | +0.09 m [0.03, 0.14] | −0.72 [−1.20, −0.10] | A 10/10 (7.5), B 10/10 (8.5) |

Checks:
- G1 (matched-circumstance difference): ✓.
- G2 (revision and recovery): ✓.
- G3: fails *mechanically*. The memory-free stop difference is < 1 mm but its CI excludes 0; behaviourally the
  robots are identical.
- G4: ✓.
- **G5 fails.** Information seeking did not speed recovery here. Without it, recovery is as complete and slightly
  faster.

**The end-to-end demo and the paired video:** `videos/G_two_robots_vision_runtime.mp4`. It is the pre-declared
lifetime LG-D 0, re-rendered once after a renderer defect (LD4; the trace is identical).
- The top half is robot A, the bottom half robot B. Each half shows:
  - the third-person view, with the entity column where the world puts it;
  - the decision trace (belief, decision, stopping distance, contacts, camera codes);
  - an inset of the robot's own depth image.
- What happens: A learns that the entity is disruptive in its 4th encounter and watches from 0.9 m. B learns it
  only in the shared stage 2 (encounter 15). After the change, B recovers by encounter 23 and A by 31, because A's
  negative history is longer.
- Plot: `figures/G_demo.png`.

## Recommended architecture (one)

**What a robot runs:**
1. its own validated motion controller and safety monitor;
2. an **IMU + rendered/real depth camera** sensing adapter that codes encounters into an `Observation`;
3. the **Bayesian change-aware type memory** (one change step per meeting, λ 0.02, exact persistence);
4. the **decision-theoretic expression mapping** with task values fixed from requirements and an **enforced
   predicted-risk cap** (HOLD when nothing is safe);
5. an **independent supervisor** re-checking every decision;
6. AffectLink v1 at the boundary.

**Sensing.** IMU + vision, not vision alone: the camera informs from a distance, the IMU at contact range. On
bodies where the camera cannot see lunges at the hand-over distance, the IMU is the only evidence there.

**Value of information.** Keep it as an option, not a default. It helped strongly in K and in LB1 (C5 survives),
but added nothing in the continuous NERVA demonstration (G5).

**Simplicity.** The decision-theoretic controller *without* VOI (E3) is the recommended default. It is simpler
and is never worse than a calibrated simple rule (§E). VOI is an option, used when the camera can observe from beyond
reach and the task needs closeness.

## Behaviour comparison

- **Contacts per pushy meeting on NERVA (LB1/LC4):**
  - H's lookup: 1.73;
  - IMU only: 0.80;
  - IMU + rendered vision: 0.44;
  - IMU + abstract camera: 0.31–0.37;
  - oracle: 0.
- **Recovery after the entity turns calm (NERVA):**
  - abstract camera: 4 meetings;
  - real vision: 7;
  - IMU: 8.
- **Wheeled robot:**
  - abstract camera: 5–6 meetings;
  - real vision: 11;
  - IMU only: never within the lifetime.
- **Holds** (the task not attempted because nothing is within the cap) occur in 51–60% of B1 meetings for IMU-only
  wheeled and humanoid robots, against 0% with the camera.

## Limitations

- **Simulation only.** The rendered camera is idealised in its own ways:
  - a single entity of known shape;
  - a clean scene;
  - depth, not RGB;
  - no motion blur or rolling shutter;
  - a navigation distance with only 3 cm error.
- **Lunges are scripted impulses.** The entity's motion is animated to match them, not simulated as a body.
- **NERVA falls under rare repeated sub-envelope pushes** at its first meeting with a disruptive entity. This is a
  controller limit (NERVA M1).
- **The humanoid's perception fails on false events** caused by its sway.
- **Wheeled real-vision recovery is slow.** On that body the abstract camera's performance was not reproduced.
- **Phase D's cost criterion is insensitive.** Task costs are dominated by approach time; memory's benefit shows in
  contacts, which the criterion only constrains.
- **The GRU alternative was not tested** (LD2).
- **The C.3 P4 clause referenced a system the design did not run** (LD6).
- **Holding beyond reach makes IMU-only robots recover slowly.** It is the price of an enforced cap without a
  distance sensor.

## What is missing for human-readable expression

- **No human has seen any of this behaviour.** H2 needs an ethics determination.
- **The expressive channels are small and task-shaped:** stopping distance, tempo, hesitation, sensing time. They
  are legible to a machine (H showed that history can be recovered from motion) but not designed for people.
- **There is no social signal** for "I am uncertain, I am watching you" beyond distance. Gaze, head and posture
  channels exist on NERVA but the decision layer does not use them.
- **There is no calibration of what people infer from these motions.**
- **No model of the observer.**

## Cloud cost and cleanup

Cap: **US$60** (approved 2026-10-18). Costs are estimated from VM hours, since live billing is not readable.

| resource | hours | estimate |
|---|---|---|
| apm-l-1 n2d-highcpu-128 spot (us-central1), preempted | ~0.55 | ~US$0.6 |
| apm-l-2 n2d-highcpu-128 spot (us-east4), self-deleted when idle | ~1.9 | ~US$1.9 |
| apm-l-3 n2d-highcpu-8 spot (render test) | ~0.2 | < US$0.1 |
| apm-l-4 n2d-highcpu-128 on-demand (us-central1) | 2.1 | ~US$8.3 |
| apm-l-5 n2d-highcpu-96 on-demand (us-west1), self-deleted when idle | ~1.1 | ~US$3.2 |
| apm-l-6 n2d-highcpu-128 on-demand (us-east4) | 0.5 | ~US$2.1 |
| Cloud NAT (3 regions, on and off) | – | < US$0.5 |
| **total** | | **≈ US$16.5, upper bound ≈ US$20** of the US$60 cap |

**Final audit (2026-10-18 09:46 UTC):** no VMs, disks, routers or addresses.
- All NATs are deleted (us-central1, us-east4, us-west1).
- Job inputs and outputs are kept in the project bucket under `apm-j/apm-l-*`.
- Raw lifetimes are git-ignored under `runs/L/` and hashed in each analysis output.

## Branch, commits, CI, PR

See the last section, written at the end.
