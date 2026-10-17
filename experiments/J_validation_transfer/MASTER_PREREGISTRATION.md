# APM v0.6 master preregistration: validation of programme I, NERVA transfer and closed-loop expressive development (programme J)

## Status

**Frozen** at the commit that adds this file (2026-10-17). What existed at that point:
- Programme J's code.
- One smoke run of the J runner: J1-P, 4 lifetimes per family, for debugging.
- The decision read-out bank from J1-T. This is training data; it contains no adaptation outcomes.

No J1-V, J1-X, J2 or NERVA data existed.

**How this document may change:**
- Changes are recorded only as dated entries in `DEVIATIONS.md`, or as addenda in `prereg/` committed before the run
  they govern.
- An addendum may fix what this file leaves open (thresholds, tuned values, sample sizes, NERVA implementation
  details) or make a criterion stricter.
- It may never relax a criterion, change a test seed or primary metric, or add a comparison to a primary family once
  test data exist.

**Earlier programmes stay as they are.** Their preregistrations and results are never edited. Programme I's PR #3 is
not merged before J1 reports.

## 0. Objective and stance

The objective is robots that learn expectations from their own experience, revise them, develop persistent individual
tendencies, express those tendencies physically, and keep learning from the consequences.

**Stance:**
- Mechanisms must earn their place against simpler alternatives.
- Nothing assumes PAD, a recurrent network or a special emotional representation.
- Negative results are reported.
- No claim about human interpretation or feelings. Human studies are **not** authorised in programme J.

## 1. Phases and gates

| phase | experiment | gate to the next phase |
|---|---|---|
| 1 | **J1**: programme I's I1 on new seeds, calibration frozen beforehand; stricter adaptation measures; validity control | none; J2 runs regardless, and J1 decides whether PR #3's claims stand |
| 1 | **J2**: stress tests of the near-miss memory (distribution shifts, policies, sensing); action- and context-conditioned evidence | the memory is **promoted as a default** only if J2's survival criteria (§3.3) hold; J3 proceeds either way, since it tests the *principle* on another body |
| 2 | **J3**: NERVA evidence discovery from robot-observable signals | J4 and J5 need ≥ 1 graded NERVA signal with markedly more information than fall/binary evidence, and EM types discoverable without labels (§4.1) |
| 2 | **J4**: paired NERVA lifetimes; revision, retention, generalisation, false attribution, restart persistence | J5 needs the type-belief memory (or a simpler one) to pass J4's criteria on NERVA |
| 3 | **J5**: closed loop, experience → belief → expression → changed interaction → evidence → revised belief | J6 runs only if J5 demonstrates the loop |
| 3 | **J6**: does the expression mapping need to be learned? | — |
| 4 | Architecture decision A–E (§6) | — |

## 2. J1: programme I validation (wheeled)

**Why.** Programme I's primary adaptation metric derived its threshold from the *same confirmatory run's* S3
lifetimes, using true labels (`I_autonomous_revision/scripts/i_metrics.py:calibration`). The robot never saw those
labels, but the evaluation threshold was not independent of the test data. A threshold-crossing metric can also
reward forgetting toward the average.

**Systems.** Programme I's frozen I1 systems (`I_autonomous_revision/configs/I_systems_X_I1.json`): A0, G2b,
R_sparse, R_static, R_dense, Oracle. Plus the validity control **R_decay**: R_dense whose evidence updates stop at the
switch time (1,050 s). Its beliefs can only relax toward the prior afterwards.

**Calibration (J1-V, never test data).**
- S3, indices 0–199, per system:
  - μ_careless, μ_careful (end-of-episode p, episodes 10–29) and the midpoint threshold θ;
  - the observed careful-action share near careless or careful identities;
  - the decision read-out (`apm_transfer.readout`) for careless and careful identities (600 sampled beliefs each).
- R_decay uses R_dense's calibration (they are the same memory before the switch).
- Bank: 1,000 decision vectors from A0 on J1-T S3 0–39 (`configs/J1_readout_bank.json`).
- The calibration file is frozen in the J1 addendum before J1-X exists.

**Test (J1-X).** Indices 0–199 of S1–S10, all 7 systems: 14,000 lifetimes.

**Measures (`scripts/j1_metrics.py`, `j1_analyze.py`):**

| measure | definition |
|---|---|
| frozen-calibrated adaptation | programme I's definition with the frozen θ |
| decisive adaptation | beyond the midpoint of θ and the post-switch μ, for ≥ 2 consecutive episodes |
| decision read-out adaptation | the rule's careful-action share in a fixed bank of real situations, computed from end-of-episode beliefs, crossing its frozen midpoint |
| observed decision adaptation | the robot's actual careful-action share near the switched identity. Reported, but confounded: it also reflects the other person's behaviour |
| belief error trajectory | normalised distance from the disposition in force |
| predictive measures | collision log loss of the memory's p for switched identities after the switch; dense predictive log-likelihood (type memories) |
| original measure | programme I's test-calibrated adaptation, for comparison only |

**Criteria.** I1 *survives* iff R_dense meets all of:
1. C1: frozen-calibrated adaptation (median ≤ 6 episodes, 95% CI upper bound ≤ 10) in S1, S2 and S10.
2. C1b: decisive adaptation, with the same bounds, in S1, S2 and S10.
3. Programme I's C2–C5:
   - C2 discrimination: S3 AUROC ≥ 0.80, CI lower bound ≥ 0.70;
   - C3 safety and C4 completion: non-inferiority vs A0 (+10% / −2 pp), Holm-adjusted over the two tests;
   - C5 normalised false blame: within ±0.25, with the 95% CI within ±0.5, using the frozen gap.

**Validity.** R_decay must *fail* C1b in each of S1, S2 and S10. If it passes any, the measure is declared unable to
separate revision from forgetting, and J1 is inconclusive.

**Reported, not criteria:** decision read-out and observed decision adaptation; predictive measures.

## 3. J2: stress tests (wheeled)

### 3.1 Conditions

Applied by `apm_transfer.stress`. The frozen testbed's code is not edited: world settings, a world subclass for new
pedestrian behaviour, input transforms, physics constants and policy variants.

| group | conditions |
|---|---|
| overlapping behaviour | `overlap_25`, `overlap_40`: each approach is made with the *other* disposition's behaviour with probability 0.25 / 0.40 |
| novel movement | `erratic`: 40% of identities, never seen in training, switch behaviour every 1–3 s at random |
| gradual change | `drift`: in S1/S2 the switched identities change *gradually*; the per-approach probability of the post-switch behaviour rises linearly over episodes 10–20 |
| identity and sensing | `identity_uncertain` (G6); `sensor_noise_0.10` and `sensor_noise_0.20` (distance, bearing and closing noise in everything the robot observes, memory evidence included) |
| geometry and crowding | `crowd_dense`, `obstacles_dense`, `fast_pedestrians` (G6) |
| robot speed | `speed_limit_0.6` (G6); `robot_fast`: GO cap 1.0 m/s, which raises near-misses caused by the robot |
| robot policy (belief evaluation) | the memory runs while the robot is driven by B0 (always GO), a cautious rule (λc × 2), or R1 (G's static-penalty rule). This tests near-misses the robot caused itself |

### 3.2 Evidence models (all with programme I's belief dynamics and λ)

| model | emissions | fitted on |
|---|---|---|
| EU-I | programme I's frozen model | I's T |
| EU-T | unconditional, refitted | J2-T (below) |
| **EA** | action-conditioned E[k, band, c], band = robot speed during observable frames (≤ 0.35 / > 0.35 m/s) | J2-T |
| **EC** | context-conditioned E[k, ctx, c], ctx = (frontal approach, crowded) | J2-T |

- **J2-T:** nominal S3 lifetimes, with the robot driven by a mixture of A0, B0, cautious and R1 (100 lifetimes each), so
  that conditioning is learnable. No labels; K by BIC.
- **Conditional information analysis** (evaluation only, truth used): I(Z; C), and I(Z; C | band) and
  I(Z; C | ctx), with plug-in estimates and a bootstrap. The question: is clearance informative about the person after
  controlling for the robot's action and the situation?

### 3.3 Design, measures and survival criteria

**Design:**
- Closed loop under A0's rule: systems A0, EU-I, EU-T, EA, EC, in every condition of §3.1 except the policy group.
- Families S1, S2, S3, S5, S10; 200 lifetimes each (J2-X).
- Calibration per condition on J2-V S3 (frozen before J2-X).
- Policy group: the same memories observe under B0, cautious and R1, in families S1, S2, S3 and S5. Their beliefs are
  evaluated; A0 and R1 closed-loop metrics are reported.

**Measures** (per condition × system):
- frozen-calibrated and decisive adaptation;
- S3 AUROC;
- careful-misclassification rate: the share of truly careful identities whose final p is on the careless side of θ;
- normalised false blame (S5);
- avoidable collisions and completion vs A0 in the same condition;
- dense predictive log-likelihood;
- expected calibration error of p_start vs collision.

**Survival.** The near-miss memory (EU-I, or the best of EU-T/EA/EC selected on J2-V) is **promoted as a default**
only if, in **every** condition:
1. S3 AUROC ≥ 0.70;
2. avoidable collisions non-inferior to A0 (+10%);
3. completion non-inferior (−2 pp);
4. careful-misclassification ≤ 0.20.

And, in at least 80% of conditions, decisive adaptation median ≤ 6 in S1 and S2.

Otherwise it is reported as a *conditional* improvement, with the failing conditions listed. Every condition is
reported, whatever the outcome.

## 4. NERVA (J3–J6)

**Constraints:**
- NERVA's controller, contract and safety are used read-only through `apm_express`.
- Pushes and contact impulses stay inside the validated envelope: ≤ 0.10 m/s, the stop-transition tolerance from H1b.
- **No privileged simulator truth is used as a live observation.** Evidence comes only from what the robot measures:
  - IMU gyro and accelerometer, with NERVA's training noise model;
  - joint positions and velocities;
  - foot contacts;
  - actuator effort;
  - its own commanded motion and task progress by odometry, if implemented from joint data.

  Simulator body velocity, base pose and entity state are evaluation-only.

### 4.1 J3: evidence discovery

**World.** A NERVA encounter world (`apm_transfer.nerva_world`). Entities of hidden types physically interact with the
robot (contact impulses and their timing) with probabilities and magnitudes that depend on:
- the entity's type;
- the robot's approach: stopping distance, speed, dwell.

These are simulation dynamics, not scripted emotional outcomes. The exact entity models are fixed in the J3 addendum
from a pilot.

**Candidate evidence:**
- peak gyro magnitude;
- peak horizontal accelerometer deviation;
- stabilisation time (until the gyro stays below a threshold);
- foot-contact disruption count;
- actuator-effort peak;
- binary "perturbation detected";
- falls.

**Measures:**
- information about the entity type per encounter (KL, AUROC; truth used for evaluation only);
- calibration;
- EM type discovery with BIC, without labels;
- fall-only vs binary vs graded evidence.

**Gate:** a graded signal carries ≥ 5× the information of binary detection, or more than falls (expected to be zero
inside the envelope), and EM finds K ≥ 2 on T.

### 4.2 J4: revisable experience on NERVA

**Design:** paired robot lifetimes, identical initial physical and learning states, with different measured
experiences. Then identical probes:
- experience-dependent beliefs;
- revision after a change;
- retention;
- generalisation to related entities (kind pools) and situations;
- false attribution (zone confound, as in H);
- recovery from mistaken beliefs;
- persistence across a restart (state saved and reloaded).

**Memories:**
- H's explicit memory (A0 / M_id);
- the type-belief memory with NERVA-fitted emissions (same `RevisionMemory` interface);
- a cumulative statistical baseline;
- a recurrent baseline **only if** both explicit memories fail a J4 criterion.

**Criteria:** templates as in J1/H4b, fixed in the J4 addendum after the J3 pilot. These include history-effect
non-inferiority to H's A0, adaptation ≤ 6 encounters-blocks, and false attribution within bounds.

### 4.3 J5: closed-loop expressive development

**Loop:** belief → ExpressionIntent (H's fixed lookup L0) → `BipedExpressionAdapter` → NERVA approach. The approach
geometry and timing change the entity interaction, which produces measured evidence, which updates the belief.

**Situations:**
- cautious approaches produce larger separation;
- direct approaches change geometry;
- hesitation changes exposure duration;
- familiar contexts become unfamiliar;
- a disruptive entity becomes reliable;
- repeated safe experience.

**Ablations:**
- no memory: neutral intent;
- frozen expression: memory updates, expression stays neutral;
- randomised histories;
- no feedback: beliefs not updated after an initial phase.

**Questions, each with its own criterion fixed in the J5 addendum:**
1. Outcomes change beliefs.
2. Beliefs change expressive actions.
3. Expressive actions change subsequent outcomes, measured against the frozen-expression ablation.
4. New evidence revises beliefs.
5. The robot recovers from excessive caution.
6. The whole loop stays safe (0 falls, safety-monitor trips reported) and task-competent (arrival and completion
   non-inferior to neutral).

A history effect alone is insufficient: questions 3–5 must hold.

### 4.4 J6: does expression need to be learned?

**Systems:**
- explicit memory + fixed L0 mapping;
- explicit memory + small learned mapping;
- a learned residual around L0;
- a standard recurrent mapping;
- a specialised persistent-state model only if all simpler ones fail.

**Supervision:**
- Available: task utility (time, arrival), physical cost (perturbation evidence, effort, safety trips) and evidence
  consistency.
- Not available: human judgements. Simulator truth is never used as the target of "expression".

**Criteria:** whether a learned mapping improves utility, safety or evidence dependence beyond L0 at equal or lower
complexity. Fixed in the J6 addendum.

## 5. Statistics (all experiments)

- Unit = lifetime.
- Paired designs with common random numbers.
- G's paired cluster bootstrap (10,000 resamples) and margins (+10% collisions, −2 pp completion).
- Kaplan–Meier medians for adaptation (bootstrap 2,000).
- Holm within primary families.
- Secondary analyses are labelled exploratory.

## 6. Architecture decision (after J1–J6)

| option | architecture |
|---|---|
| A | predictive regulation + dynamic memory + fixed expression mapping |
| B | predictive regulation + dynamic memory + learned expression mapping |
| C | modular learned regulatory state + explicit memory + expression adapter |
| D | standard recurrent architecture with memory |
| E | insufficient evidence; keep the simpler baseline |

**Criteria:** safety, learning speed, adaptability, expressive history dependence (evidence-consistent, stable,
revisable and useful, not maximal), robustness, complexity and compute.

**Rules:**
- The simplest option meeting the criteria is chosen.
- No neural architecture is presented as novel merely because its hidden states produce different movements.

## 7. Compute and provenance

- Local runs for tests and pilots. Capped Google Cloud spot VMs for calibration, test and NERVA runs: the user
  approved US$40 for programme J on 2026-10-17. Spend is logged in `reports/COMPUTE_LOG.md`.
- Confirmatory runs are submitted from committed code, with `apm_transfer.provenance.confirmatory_lock` passing.
- Raw data stays under `runs/` (git-ignored); hashes go in `manifests/`.
- Models and checkpoints are committed or hashed.
