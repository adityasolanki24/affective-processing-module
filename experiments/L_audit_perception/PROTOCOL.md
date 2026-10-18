# Programme L protocol: audit corrections, real perception, next milestone

**Governance.**
- This file is **append-only**. Each section is frozen by the commit that adds it, before any data it governs
  exist (partition V or X). A frozen section is never edited. Any later change goes in `DEVIATIONS.md` with a
  reason.
- Seeds come only from `seeds.py` (new entropy, independent of programmes F–K). Partition P is for debugging only.
- Confirmatory runs use `--addendum ../L_audit_perception/PROTOCOL.md` on a clean committed tree.
- Programme K's results are preserved unchanged. Phase A (audit fixes, `results/A*`) is code correction and
  reanalysis, not a test. Everything is simulation only.

---

## B. Corrective recovery test (frozen 2026-10-18, before any L V or X data)

**Question.** Which of programme K's conclusions survive when all the audit corrections are applied together on
fresh seeds?

**Corrections in force in every system** (no legacy flags, except in the two reference systems below):
- A3: the risk cap is enforced; with no option within it, the robot executes HOLD (stops at 0.9 m, beyond reach;
  the task is not attempted; no IMU evidence there);
- A5: the camera's 40° field of view is a full angle (±20°);
- the fixed memory clock (`--transition-fix`);
- A1: confirmed event onset (2 consecutive favourable observations, both observed).

**Pilot basis.** One local pilot (LB3, partition P, 12 lifetimes) checked that HOLD, OBSERVE and the new fields
run. It showed that IMU-only DT holds at 0.88 m once confident in a pushy type. No value, criterion or system was
chosen from it.

### B.1 Experiments

| exp | body | systems | histories | V (safe only) | X |
|---|---|---|---|---|---|
| LB1 | NERVA | L0, DT_STATIC, DT, DT_VOI, DT_UNC, DT_CAM, DT_CAM_VOI, ORACLE, **DT_K**, **DT_CAM_VOI_K** | risky_B, safe, change_B, late_B | 0–29 | 0–49 |
| LB2 | wheeled | L0, DT, DT_VOI, DT_CAM, DT_CAM_VOI, ORACLE | same | 0–29 | 0–49 |
| LB3 | humanoid (T1) | L0, DT, DT_VOI, DT_CAM, DT_CAM_VOI, ORACLE | same | 0–29 | 0–49 |
| LB4 | NERVA | L0, DT | risky_B, safe; tasks waypoint, inspect, handoff, separation | – | 0–39 |

- All bodies, configs, type models, time models, edges, task values and VOI settings are K's, unchanged.
- **DT_K and DT_CAM_VOI_K** are DT and DT_CAM_VOI with K's legacy cap and legacy camera. They run on the same
  fresh seeds, so the effect of the corrections is a paired comparison. They are references, not candidates.

### B.2 Recovery target, calibrated independently

- Per body and system, the target is the mean B1 stopping distance at encounters ≥ 10 in the safe history, on
  partition **V**.
- `l_b_targets.py` writes `configs/LB_recovery_targets.json`. That file is committed **before** any X analysis
  and never changed afterwards.

### B.3 Criteria (LB1–LB3; `l_b_analyze.py`)

As in K §4, with corrections:

| | criterion |
|---|---|
| P1 | change_B: KM median pushy → calm revision ≤ 6 B1 meetings |
| P2-task | change_B: KM median ≤ 8 B1 meetings until the B1 stopping distance is ≤ 0.35 m (handoff 0.30 + 0.05) for 2 consecutive meetings |
| P2-V | change_B: KM median ≤ 8 B1 meetings until it is within 0.05 m of the V-calibrated target for 2 consecutive meetings |
| P3 | late_B: KM median calm → pushy revision ≤ 4 B1 meetings |
| P4 | 0 falls; trips ≤ 1% of encounters; repeated-push encounters not above DT's (paired upper bound ≤ +0.02, change_B and late_B); **0 executed approaches with predicted risk above the cap** (recomputed from the logged belief) |
| P5 | navigation arrival (at the commanded stop, a hold included) ≥ 0.99 |

- **The recovery milestone is met** by a candidate only if P1, P2-task, P2-V, P3, P4 and P5 all pass.
- **Candidates:**
  - IMU only: DT_VOI on all bodies;
  - camera extension: DT_CAM_VOI on all bodies.
- DT and DT_CAM are judged the same way, for reference.
- **Task outcomes are reported separately, not as criteria:**
  - interaction reached: stopping distance ≤ 0.35 m;
  - interaction hold completed: within 0.35 m throughout the first sensing window;
  - hold (no safe action) rate.

**LB4** (K §5, with measured success):
- Per encounter, task cost = approach time + 4 s × (windows − 1) + 20 s × strong perturbations
  + 12.6 s × (1 − success).
- Success is measured:
  - inspect and handoff: hold completed;
  - separation: stopping distance ≥ 0.5 m;
  - waypoint: arrival.
- DT is promoted for a task if:
  - cost difference: 95% upper bound < 0;
  - contacts: upper bound ≤ +0.05;
  - 0 falls.
- DT replaces L0 if it is promoted in ≥ 3 of 4 tasks.

### B.4 Verdicts on programme K's claims (rules fixed now)

| claim (K) | survives | weakens | fails |
|---|---|---|---|
| C1 NERVA camera + VOI meets the milestone | DT_CAM_VOI milestone met | P1, P3, P4, P5 met and exactly one P2 variant met | otherwise |
| C2 NERVA IMU only meets it (at the margin) | DT_VOI milestone met | as C1 | otherwise |
| C3 wheeled: camera + VOI meets, IMU only does not | DT_CAM_VOI met and DT_VOI not met | DT_CAM_VOI weakens (as C1) | DT_CAM_VOI not met, or DT_VOI met |
| C4 humanoid: same as C3 | as C3 | as C3 | as C3 |
| C5 safe information gathering cuts contacts per pushy meeting (NERVA ~3×) | DT_CAM_VOI / DT ratio ≤ 0.5, bootstrap upper bound < 1 | upper bound < 1, ratio > 0.5 | upper bound ≥ 1 |
| C6 the decision-theoretic mapping beats H's lookup (4 of 4 tasks) | promoted in 4 of 4 | 3 of 4 | ≤ 2 |
| C7 caution returns within 1–2 meetings (calm → pushy) | candidates' P3 median ≤ 2 | ≤ 4 | > 4 |

- **Reported next to C5, never as a verdict.** The pilot showed that a holding robot avoids contact by abandoning
  the task. So the realised task cost per B1 meeting (LB4's cost with measured success; a hold costs its time plus
  the 12.6 s retry) is reported for DT and DT_CAM_VOI, with paired bootstrap CIs.
- C5 is judged on LB1 (NERVA) and reported for LB2 and LB3. Its ratio is pooled across histories, over lifetimes,
  with bootstrap CIs.
- C1–C4 are judged per body.
- The verdicts are mechanical. Explanations, such as "IMU silence at the hold distance is not evidence", are
  reported separately from the verdicts.

### B.5 Compute

- Cloud spot VMs (`jcloud.py`), under programme L's US$60 cap. A preempted job is rerun with identical seeds, and
  the rerun is logged.
- Raw lifetimes are git-ignored under `runs/L/` and hashed in the analysis output.

---

## C. Rendered-image perception (frozen 2026-10-18, before any LC T, V or X data)

**Question.** Is there a genuine vision pipeline, and does recovery still work when the robot learns from noisy
visual observations instead of K's abstract event sensor?

**Pipeline** (`src/apm_transfer/vision.py` at this commit).

- **World side.**
  - The entity is drawn as a person-scale column, render-only, so physics is unchanged.
  - It physically slides toward the robot through each scheduled lunge, peaking at the impulse step.
  - A depth eye on the robot renders at 10 Hz during the sensing windows:
    - placement: NERVA's head, the wheeled robot's mast, the T1's head;
    - 64 × 48 px, 40° horizontal field of view, pitched 10° down;
    - noise σ (1 + depth) with σ = 1 cm, and 2% dropouts.
- **Perception side (`MotionPerception`).**
  - Inputs:
    - depth frames;
    - own-motion flags (efference copy);
    - the navigation estimate of the target distance (true + N(0, 3 cm) per encounter);
    - the robot's knowledge of its own camera: intrinsics, pitch, noise specification, mounting offset.
  - Method:
    - median background;
    - a noise-scaled threshold;
    - spatial coherence;
    - an egomotion rejection;
    - foreground counted only near the target surface;
    - events tolerate a one-frame gap;
    - reach = navigation distance − eye offset − closest foreground.
  - Codes are K's 8 categories.
  - It never reads simulator state. A unit test checks its interface.

**Pilot basis.** The perception was developed on smoke encounters and partition P only.
- A wheeled P pilot (LC2-P, 40 encounters) showed recall 1.0 for lunges that displace the column ≥ 5 cm and 0 below.
  That fixed the *definition* of a visible lunge.
- The gate thresholds below were set before that pilot and are unchanged.

### C.1 Perception quality (LC1 NERVA, LC2 wheeled, LC3 humanoid; X 0–399; `c_perception.py`)

- **Encounters.** Calm and pushy alternate, at stopping distances 0.3, 0.4, 0.6 and 0.9 m (stand-off), with 3
  windows each. Every encounter is simulated twice with the same seed: rendered vision, then the abstract camera.
- **Metrics:**
  - recall of visible lunges (in view, displacement ≥ 5 cm);
  - recall over all moving lunges;
  - false events per window;
  - reach bias and MAE;
  - KL(calm ‖ pushy) of the window code per stopping distance, for both sensors.
- **Perception gate per body:**
  - G1: recall ≥ 0.7;
  - G2: false events ≤ 0.25 per window;
  - G3: vision KL at 0.6 m ≥ 0.2 nats per window (the IMU gives 0.14 there, K1).
- "A genuine, working vision pipeline" on a body means G1–G3 pass. The humanoid is judged the same way.

### C.2 Vision type models (LC4 NERVA, LC5 wheeled; partition T 0–99)

- **Exploration:** RANDINTENT_VIS × 4 histories.
- **Fit:** `k_fit.py`, K's procedure (K by BIC), writing `configs/LC_{nerva,wheeled}_vision_types.json`.
- The fitted models are committed before any LC V or X run.

### C.3 Recovery comparison (LC4 NERVA, LC5 wheeled; V 0–29 safe only; X 0–49 × 4 histories)

| system | sensing | type model |
|---|---|---|
| DT_VOI | IMU only | K's |
| DT_CAM_VOI | IMU + abstract camera (corrected FOV) | K's |
| DT_VIS_VOI | rendered vision only (IMU not observed) | vision |
| DT_IMUVIS_VOI | IMU + rendered vision | vision |
| ORACLE | true type (reference) | K's |

- **Corrections:** all of §B (enforced cap, fixed memory, confirmed onset). V-calibrated targets are written by
  `l_b_targets.py` to `configs/LC_recovery_targets.json` and committed before any X analysis.
- **Criteria:** exactly §B.3, P1–P5 with both P2 variants (`c_recovery_analyze.py`).
- **Questions:**
  - Q1: does DT_IMUVIS_VOI meet the recovery milestone? (per body)
  - Q2: does DT_VIS_VOI meet it? (per body)
- **Reported, not decisive:** paired differences against DT_CAM_VOI in recovery meetings (censored at 14) and in
  contacts per pushy meeting.
- **Interpretation rules, fixed now:**
  - If Q1 passes where DT_VOI fails, the camera's benefit in K survives with real perception.
  - If Q1 fails where DT_CAM_VOI passes, K's camera result depended on the abstract sensor.

---

## D. Behavioural learning under stress (frozen 2026-10-18, before any LD data)

**Question.** Is the system effective when the entity's behaviour, the world or the sensing changes? Is another
memory better?

**Body and policy.**
- The wheeled robot with IMU + rendered vision.
- The decision-theoretic mapping with VOI, with every §B correction.
- Its type model is §C.2's wheeled vision model, committed before LD runs.

**Runner and analysis:** `d_stress.py` and `d_analyze.py` at this commit.

**Scenarios (15):**

| scenario | change |
|---|---|
| nominal | K's change at encounter 16 |
| overlap | forceful/gentle types, not seen in fitting |
| drift | gradual change over encounters 16–32 |
| erratic | flips with p 0.3 per meeting |
| positions | 4 entity places |
| occlusion | 40% of encounters partly hidden |
| identities | 6 identities, B2 opposite |
| tracking | 15% identity confusions |
| speed fast / slow | lunge speed ×1.6 / ×0.5 |
| camera_poor | 32×24, σ 3 cm, 10% dropout, 5 Hz |
| context | behaviour depends on the zone approached |
| goal inspect / separation / waypoint | other task goals |

- Histories: change (risky → safe) and late (safe → risky), for B identities.
- **Seeds:** experiment LD, partition X 0–29. Every system sees identical entity draws.

**Memories:**
- M0: none (prior);
- M1: Bayesian change-aware type memory (λ 0.02);
- M2: recency-weighted leaky evidence (α 0.2, set a priori: half-life about 3 meetings);
- M3: M1 keyed by (identity, zone or position);
- ORACLE: the true current type (reference).

**A small predictive model (GRU)** is run only if it is justified, i.e. if M1 and M2 both fail D1 in ≥ 2
scenarios. J4 found that the GRU adds nothing over the type model.

**Criteria.**
- **D1, per scenario:** M1 is effective if, against M0:
  - task cost per B meeting: paired upper bound < 0;
  - contacts per risky meeting: upper bound ≤ +0.05.
- **Answer to "effective under changed entity behaviour?":**
  - yes: D1 holds in ≥ 12 of 15 scenarios;
  - partly: 8–11;
  - no: < 8.
  - The failing scenarios are named.
- **Memory choice:**
  - M2 replaces M1 if its cost is lower (upper < 0) in ≥ 3 scenarios and higher (lower > 0) in none.
  - M3 is recommended as a context extension if it is lower than M1 in *context* and higher in at most one
    scenario.
  - Otherwise M1 stays: the simpler or incumbent option wins ties.
- **Reported:**
  - anticipation accuracy (the belief before the meeting matches the risk);
  - excess avoidance;
  - the gap to ORACLE.

---

## E. Fair comparison of expression controllers (frozen 2026-10-18, before any LE data)

**Question.** Is the decision-theoretic expressive controller better than a *calibrated* simple baseline?

**Setting.**
- NERVA with IMU + rendered vision.
- Memory: M1, the fixed clock, λ 0.02.
- Type model: the §C.2 NERVA vision model.
- K's four tasks, with their frozen values.
- All §B corrections apply (the cap binds E3 and E4, which have a risk model).
- Controllers: `src/apm_transfer/controllers.py`.

| id | controller | free parameters | tuning |
|---|---|---|---|
| E0 | H's lookup | none | none |
| E1 | retuned lookup (offset c, gain a) | 2 | 12-point grid, includes E0 |
| E2 | monotonic proximity rule (lo, width) | 2 | 12-point grid |
| E3 | decision-theoretic mapping (greedy) | none (task values frozen) | none |
| E4 | E3 + value of information | none | none |
| E5 | learned mapping | – | run only if justified (below) |

**Tuning (equal budgets).**
- Partition T, indices 0–7; histories risky_B and safe.
- Per task: all 12 configurations of E1 and of E2.
- `e_controllers.py tune` picks the lowest mean cost, with ties to the simpler (earlier) entry.
- It writes `configs/LE_tuned.json`, committed before any X run.

**Test.** Partition X, indices 0–29: E0, E1\*, E2\*, E3, E4 × 4 tasks × 4 histories.

**Criteria per task** (`e_controllers.py analyse`):
- E3 or E4 "beats the calibrated simple baselines" if, against **each** of E0, E1\* and E2\*:
  - paired task cost: upper bound < 0;
  - contacts per encounter: upper bound ≤ +0.05.
- **Answer:**
  - yes: one of them beats the baselines in ≥ 3 of 4 tasks;
  - partly: 1–2;
  - no: 0.
- **Recommended controller per task:**
  - the simplest controller (E0 < E1 < E2 < E3 < E4) whose cost is at most 1 s worse than the lowest-cost
    controller (upper bound of the paired difference ≤ 1 s);
  - the simpler option wins when they are equivalent.
- **E5 (learned)** is justified only if the answer is "no". It is not run in L otherwise.

---

## F. Integrated runtime (engineering verification; written 2026-10-18 *after* the first demo runs)

This section is not a statistical test. Its checks are binary properties of the code, with no thresholds that
could be tuned. It was written after the first demo runs, and nothing was changed in response to them.

**What runs.**
- `src/apm_transfer/runtime.py`, with `f_runtime_demo.py`.
- One 24-encounter lifetime per body, with B1 changing from pushy to calm at encounter 12. Seeds: LF, partition D.
- Bodies and sensing:
  - NERVA and wheeled: IMU + rendered vision, with their vision models;
  - humanoid: IMU + abstract camera, because there is no humanoid vision model.

**Checks** (all must pass on ≥ 2 bodies, including NERVA):
- F1: the same runtime code runs on the body;
- F2: decisions after a save → delete → load restart are identical to an uninterrupted runtime's;
- F3: a replay of the log reproduces every decision;
- F4: every AffectLink v1 message validates and survives an encode/decode round trip;
- F5: no executed approach exceeds the predicted-risk cap. The independent supervisor's vetoes are counted, and
  its veto of every over-cap approach is unit-tested;
- F6: every intent is within its bounds.
