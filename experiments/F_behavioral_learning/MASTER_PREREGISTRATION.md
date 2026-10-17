# APM v0.2 master preregistration: critical experimental programme (E0–E7)

Status: **frozen** at the commit that adds this file. Written on 2026-10-11 after the repository audit
(`AUDIT.md`) and before any v0.2 data, pilot or held-out result existed. Changes after this commit are made only as
dated entries in `DEVIATIONS.md` or as per-experiment addenda (`prereg/E*_ADDENDUM.md`) that are committed before
the confirmatory run they govern. An addendum may only (a) fix sample sizes from pilot variance using the rules
below, (b) fix implementation details left open here, or (c) make a criterion *stricter*. It may never relax a
threshold, change a test seed, or change a primary metric.

## 0. Purpose and stance

The programme tries to falsify the need for APM's affective layer and then tests whether a trainable regulator beats
simpler alternatives. A negative result is an acceptable and useful outcome. Terms: "functional affect", "latent
state", "predictor", "regulator"; no claims about emotion, consciousness or personality.

Prior findings (C, D, B, E) are preserved and are not re-tested as if new. Old evaluation data are not reused as
fresh test data (see §3).

## 1. Systems under test

### 1.1 Testbed (implemented in E0; wheeled only for behaviour)

`apm_testbeds.wheeled_ctrl`, a new module. The v0.1 `apm_testbeds.wheeled` is left unchanged (Milestone D depends on
it). Same unicycle physics and constants as v0.1 (dt 0.05 s, 6 m arena, v ≤ 0.8 m/s, ω ≤ 2 rad/s, accel limits,
traction margin `1 − |vω|/(μg)`, μ = 0.8 / 0.08 on low-friction patches, slip bleeds speed and perturbs heading).
Additions:

- **Independent random streams** (layout, goal sequence, each pedestrian, slip noise, sensing) from
  `numpy.random.SeedSequence`, so exogenous randomness is common across policies (common random numbers). Divergence
  that remains is policy-caused (pedestrians react to the robot) and is part of the measured effect.
- **Pedestrians with persistent identities and hidden dispositions** drawn from a lifetime population:
  *careful* (slows and sidesteps when the robot is close and closing) or *careless* (keeps going and drifts toward
  the robot when close). Both look identical to sensors (class `person`, same speed range). Dispositions may change
  on a schedule (E3). Dispositions are ground truth for evaluation only.
- **Goal sequence with deadlines.** Waypoints are pre-generated per episode; each goal has a deadline
  `3 s + d / 0.4 m/s` (d = distance at issue). A missed deadline is a task failure and the next goal is issued.
- **Safety supervisor (identical for every system, never dependent on any policy or the APM):** speed cap; an
  emergency brake (3 m/s²) whenever any entity is within 0.15 m clearance in the forward ±70° sector and closing.
  Every activation is logged as an intervention.
- **Contact episodes:** a collision event per entity is reported once per contact episode (1.0 s refractory), fixing
  D4. Severe collision: robot speed ≥ 0.4 m/s at contact.
- **Bounded high-level action interface**, chosen every 0.5 s (10 physics steps) by the system under test:

  | action | meaning |
  |---|---|
  | GO | ordinary reactive navigation (v0.1 controller), speed ≤ 0.8 |
  | SLOW | same route, speed ≤ 0.3 |
  | YIELD | stop and wait (may rotate in place toward the goal) |
  | DETOUR | wider berth around people (person clearance 1.5 m instead of 0.8 m), speed ≤ 0.6 |

  "Resume normal progress" is GO. Navigation, dynamics and the safety supervisor are the same code for every system;
  only the action differs.
- **Observations available to every system** (and nothing else): own speed, turn rate, traction margin, energy;
  goal distance, bearing and remaining deadline; tracked entities within 3 m (track id, class, reported identity and
  identity confidence, distance, bearing, radial closing speed); event signals (collision with target and severity,
  slip onset, goal reached, goal timeout, safety intervention); the robot-side TTC collision estimator (same formula
  as v0.1, retracted when closing stops, fixing D3). No system receives dispositions, friction maps, future states
  or rewards that the others do not.

### 1.2 Decision systems

| id | system |
|---|---|
| B0 | ordinary controller: always GO (plus the shared safety supervisor) |
| B1 | strong non-affective baseline: an action-conditioned consequence predictor (collision within 2 s, slip within 2 s, goal progress within 2 s, for each action) trained on exploration data, explicit per-identity/type outcome memory (the APM `EntityMemory`, outcome-grounded), and the transparent rule `argmax_a  progress(a) − λc·P_col(a) − λs·P_slip(a)` |
| B2 | existing APM: B1's predictor, memory and inputs, plus the APM v0.1 core (hand-set appraisal, persistent PAD, advisory tendencies) running at 20 Hz on the adapter stream. Affect modulates the rule: `λc' = λc·(1 + k_pad·(max(0, A) + max(0, −V)))`, and tendency bonuses `+k_t·avoid_max` to DETOUR and `+k_t·disengage_max` to YIELD (max over tracked targets) |
| B2 ablations | B2 with: no PAD (k_pad = 0); no tonic appraisal (tonic input u ≡ 0, phasic outcome impulses kept); no entity-memory predictions (APM memory frames off); no persistent affect (affect state reset at every decision); shuffled affect (negative control: PAD/tendency readouts replayed from a different lifetime of the same partition) |
| B3 | APM-L learned regulator (E4): action-conditioned consequence predictor + explicit memory features + learned recurrent latent (fast and slow components) → bounded preference output added to predicted-consequence utility; safety supervisor unchanged |
| B4 | capacity-matched recurrent policy (E4): GRU policy, same inputs, actions, training data and environment-step budget, parameter count within ±10% of B3 |
| Oracle | evaluation-only: B1's rule with memory replaced by true disposition rates (upper reference in E3) |

**Tuning fairness.** B1's λc, λs are chosen on the validation partition by grid search maximising the tuning
utility J (§4.3). B2 receives the same λ grid crossed with k_pad ∈ {0.5, 1, 2} and k_t ∈ {0.1, 0.25, 0.5}
(strictly positive: the affect channel must be used). B2 therefore has more tuning freedom than B1, which biases the
comparison *in favour of* B2. A post-hoc "would k = 0 have been chosen" analysis is exploratory.

## 2. Hypotheses, primary metrics and criteria

General statistics (all experiments):
- **Unit of analysis = lifetime** (or episode for single-episode designs), never control steps. Paired designs:
  every system runs on the same lifetime seeds. CIs are 95% percentile intervals from a paired cluster bootstrap
  over lifetimes (10,000 resamples). With several neural seeds, the bootstrap is two-level (lifetimes, then
  seeds). Ratio metrics use ratio of totals.
- Relative difference `r = (metric_X − metric_Y) / metric_Y`.
- A non-significant difference is **not** equivalence. "Practically equivalent" is claimed only when the whole CI
  lies inside the stated practical margin; otherwise the result is "inconclusive".
- Multiplicity: Holm correction within each experiment's primary family (stated per experiment); secondary analyses
  are reported unadjusted and labelled.
- Pilots use only the pilot partition. Pilot results are used for debugging, variance estimates and sizing, are
  reported as pilots, and never enter confirmatory statistics.

### E0 — experimental integrity (gate)

H0: the testbed supports fair causal comparisons. **Pass requires all** automated tests: (1) different actions from
the same state change subsequent motion and outcomes; (2) all systems share the same legal actions and the same
safety supervisor code path; (3) an APM exception or absence leaves navigation and safety working (fallback to GO);
(4) a policy that always yields completes 0 goals (standing still cannot succeed); (5) no observation contains
information stamped after the decision time and no hidden state is exposed; (6) common random numbers: identical
action sequences reproduce identical episodes, and the goal sequence is independent of the policy; (7) duplicate and
missed event checks against ground-truth contact logs (0 duplicates, 0 misses); (8) the predictor training labels
use only outcomes after the decision; (9) partitions are disjoint and exclude the old seeds. Also reported (no
criterion): simulator throughput, TTC estimator calibration. **If E0 fails, no behavioural comparison runs.**

### E1 — does affective machinery add predictive information?

H1: appraisal and/or PAD add predictive information beyond identical high-frequency data and explicit memory.

Targets: wheeled `collision` within 1.0 s (primary), `stability_loss` (slip) within 0.5 s (secondary); NERVA
`stability_loss` within 0.5 s (primary, only if fresh NERVA data are available, see §3).
Sampling: 10 Hz samples, 2 s history, armed samples only (as Milestone C/D).

| arm | inputs (all at 10 Hz) |
|---|---|
| E1-LR | logistic regression, canonical features + risk summary (current sample) |
| E1-G | GRU, canonical masked features (as C/D) |
| E1-R | GRU + high-frequency risk summary: per 0.1 s window, from the full-rate stream the APM ingests: max and mean live hypothesis probability, min entity distance, max closing speed, count of each outcome kind |
| E1-U | E1-R + tonic appraisal vector u (5) |
| E1-P | E1-R + PAD (3) |
| E1-UP | E1-R + u + PAD (8) |
| E1-M | E1-R + memory expectations for the nearest entity (p_adverse, confidence, familiarity) (wheeled only) |
| E1-N | E1-R + 8 standard-normal noise channels (input-width control) |
| E1-C | E1-R with hidden size raised so that parameters match E1-UP within ±5% (capacity control) |

The APM providing u/PAD is fed exactly the stream that the risk summary is computed from (same rate, same messages).
All GRU arms: hidden 64 (except E1-C), same schedule and early stopping as C, **5 initialisation seeds**.

**Criterion (per embodiment).** Primary family (Holm, 3 tests): E1-U, E1-P, E1-UP each vs E1-R. An arm "adds
meaningful information" iff its relative Brier difference vs E1-R is ≤ −5% and the (Holm-adjusted) CI upper bound
is < 0, **and** it also beats E1-N and E1-C on the same rule's point-estimate condition (≤ −5%). "No meaningful
information" iff the CI lies within (−5%, +5%). Otherwise inconclusive. Secondary: E1-M vs E1-R (memory value),
E1-R vs E1-G (value of high-rate summaries), log loss, ECE, AUROC, recall at 5% FPR, false alarms in benign
encounters (careful identities), held-out shift partition, inference latency.

Decision: if PAD (E1-P) is not meaningful, PAD is no longer treated as a necessary predictive representation.

### E2 — does the existing APM improve behaviour?

H2: B2 improves safe, efficient behaviour beyond B1.

Environment: lifetimes of 10 × 60 s episodes in randomised worlds (pedestrian speed U(0.2, 1.0) m/s, 2–4 present
per episode from a population of 6 identities, 40% careless; 3–6 obstacles; 0–3 low-friction patches; goal
deadlines), memory persisting across a lifetime's episodes.

Primary metrics (per lifetime): goal completion rate (completed / issued), collision events per minute (all
collisions; person collisions secondary), plus severe collisions per minute, time to goal, distance per minute,
idle fraction (decision periods with mean speed < 0.05 m/s while no entity is within 1.0 m and no intervention),
interventions per minute.

**Criterion (B2 beats B1)**, primary family of 2 (Holm):
- (A) collision rate `r ≤ −10%` with CI upper bound < 0, and completion difference CI lower bound ≥ −3 pp; or
- (B) completion difference ≥ +5 pp with CI lower bound > 0, and collision `r` CI upper bound ≤ +5%;
- and in either case **no degenerate freezing**: B2 idle fraction ≤ B1 idle fraction + 5 pp.

Secondary: B1 vs B0 (same rule; if B1 does not beat B0 the testbed may not reward prediction at all, which is
reported), each ablation vs B2, severe collisions, interventions. Decision: if B2 does not beat B1, the hand-written
affective layer has not justified its complexity; it is kept as a baseline only.

### E3 — experience, changing relationships and false attribution

H3: outcome-grounded memory supports useful, persistent, revisable individual differences.

Lifetimes of 30 × 60 s episodes, population of 6 identities, scenario families: S1 hazardous→benign switch for two
identities at episode 15; S2 benign→hazardous; S3 stable mixed histories; S4 identity errors (20% of encounters
report a wrong identity with normal confidence, 10% ambiguous with confidence 0.4); S5 accidental co-occurrence (one
careful identity frequents low-friction patches, so slips co-occur with it); S6 censoring (a careless identity that
the policy learns to avoid, with and without ε = 0.1 randomized safe action variation near people); S7 same-class
newcomers (a new careful identity appears after a careless-heavy history). Systems: B0, B1, B2, B1-ε, Oracle.
A naive temporal-association memory (credits any adverse outcome within 1 s to every entity within 1.5 m) is an
evaluation reference only.

Metrics: per-identity ranking AUROC of memory p_adverse vs true disposition (per episode); adaptation delay (episodes
after a switch until the switched identity's p_adverse crosses the population median); predictive regret (Brier of
per-encounter collision prediction vs Oracle); false attribution (change in p_adverse of the S5 identity relative to
matched controls); overgeneralisation (S7 newcomer's first-encounter p_adverse minus its true rate); excess avoidance
(non-GO decisions near careful identities, minus Oracle); retention across restart (save/kill/restore mid-lifetime:
memory identical, decisions identical on a replayed observation stream); history dependence (two robots with
different histories given identical observation streams: decision divergence) and its usefulness (collisions with
careless identities and completion, vs B0 and Oracle).

**Criteria (all on the confirmatory partition, B1 memory; B2 reported alongside):** (i) S3 end-of-lifetime ranking
AUROC ≥ 0.80 (CI lower bound ≥ 0.70); (ii) S1/S2 median adaptation delay ≤ 6 episodes, with S1 under B1 and under
B1-ε reported separately (censoring); (iii) S5 false attribution: CI of the p_adverse change within ±0.03;
(iv) usefulness: B1 person-collision rate with careless identities ≤ 0.8 × B0 (CI upper < 1.0) with completion
non-inferior within 3 pp. Evidence for "adaptive dispositions" requires (i)–(iv); otherwise the finding is
"encounter statistics only" or "misleading statistics" (named per failed criterion).

### E4 — learned versus handwritten regulation

H4: a trainable persistent regulator (B3) improves long-horizon behaviour beyond explicit memory (B1), handwritten
APM (B2) and an ordinary capacity-matched recurrent policy (B4).

Training: on-policy actor-critic (PPO) on the training partition in the E2 environment; equal environment-step
budgets for B3 and B4 (fixed in the addendum from pilot learning curves, capped by the compute limit); 5 training
seeds each; checkpoint selection by validation J only. B3 ablations: without explicit memory features; without
persistent latent (reset per episode); fast-only and slow-only latent; identity-shuffled memory and history-reset
negative controls; B4 at 0.5× and 2× capacity (unmatched budgets).

**Criterion:** B3 counts as a credible candidate iff it beats **both** B4 and B1 under the E2 rule (A or B, with
anti-freezing; Holm over the two comparisons), with the two-level bootstrap over test lifetimes and training seeds.
If B3 beats B1 but not B4, that is reported as "no evidence for a specialised regulation structure". No claim of a
"novel emotional neural network" is made in any case.

### E5 — lifelong adaptation (gated)

Full version only if E4 yields a credible candidate. Otherwise the infrastructure (sequential phases: pedestrian
pattern, friction and task-distribution changes; frozen evaluation suite after each candidate update; shadow learning
with/without replay; rollback; corrupted-experience injection) is implemented and a reduced run on the B1 predictor is
reported. Success requires a held-out improvement ≥ 5% in J relative to the frozen policy with CI lower bound > 0,
no safety regression (collision `r` CI upper ≤ +5%), and at least one promoted update; "no forgetting because every
update was rejected" is reported as a failure of lifelong learning, not a success.

### E6 — cross-embodiment (gated)

Software portability, zero-shot predictive generalisation and beneficial transfer are separate claims. Full
transfer study only if E4 passes. Information must be matched: NERVA lacks entities, collisions and a behavioural
interface (B4 in AUDIT), so no cross-robot social/behavioural transfer is claimed in any case. If fresh NERVA data
are available, a reduced predictive study on the shared `stability_loss` semantics compares zero-shot, scratch,
fine-tuning, frozen shared encoder + robot head, and logistic regression at N ∈ {5, 20, 80} episodes, 3 seeds
(criterion as Milestone D TR2: ≥ 10% Brier improvement over scratch at N = 5 and 20 in ≥ 2 of 3 seeds). NERVA stays
observe-only; nothing feeds back into its motor policy.

### E7 — robustness and safety independence (runs from the start)

Faults: missing/stale observations, sensor outage (2 s), conflicting predictions, wrong identities, corrupted memory
file, out-of-order and duplicated messages, 10× event rate, miscalibrated estimator (p × 3, clipped), sudden world
change, APM crash and restart, model failure, 0.5 s delayed inference, uncertain outcome evidence.
**Criteria:** (1) no fault raises an exception into the robot loop; (2) on APM/model absence the decision falls back
to GO within one decision period; (3) the safety supervisor's activations are identical with the APM present, absent
or crashed for the same state (unit test); (4) under each fault, the collision rate of B1/B2 is not worse than B0's
by more than 5% relative (CI upper bound) — a fault-tolerant advisory layer should degrade to the ordinary controller,
not below it. Failures are reported per fault.

## 3. Data partitions, seeds and provenance

- World seeds come only from `experiments/F_behavioral_learning/seeds.py`:
  `SeedSequence(entropy=0xA9_0002_2026_1011, spawn_key=(experiment, partition, index))` with partitions
  **P** pilot, **T** train, **V** validation, **X** confirmatory test, **S** shift test. Partitions are disjoint by
  construction; v0.1 integer seeds feed `default_rng(int)` in the old testbed only and are never used.
- Confirmatory runs (partitions X and S) are executed only by runners that check that the experiment's addendum is
  committed and the working tree is clean, and that record the commit hash, source hashes (`src/`, the runner) and
  output hashes in a manifest.
- Neural initialisation seeds 0–4. Bootstrap seed 0.
- NERVA fresh data (if the policy file is provided): master seed 2027, simulator seeds 100,000 + trial index; split
  by trial into T/V/X (60/20/20, seeded permutation 0). The Milestone C dataset (seed 2026, simulator seeds 0–599) may
  be used only for pilots and is labelled as such.
- Large raw data live under `runs/` (git-ignored) with committed manifests (SHA-256 per file).

## 4. Procedures fixed now

4.1 **Sample sizes** come from pilots: N lifetimes = the smallest N giving ≥ 80% power (paired bootstrap-normal
approximation from pilot SDs) to detect the criterion effect (E2: r = −10% on collisions; and ±3 pp on completion),
with minimum 100 and maximum 400 lifetimes (E2), 60/240 (E3 per scenario family), 120/400 test lifetimes (E4). If
the cap is hit, the experiment runs at the cap and reports the achieved power.

4.2 **Predictor training (B1/B2):** exploration lifetimes on partition T with uniformly random actions at each
decision (randomised safe action variation; the safety supervisor stays on), model = MLP(64, 64) with action one-hot
input and three heads, early stopping on partition V, 5 seeds; model selection only on V.

4.3 **Tuning utility** J per lifetime = goals completed − 1.0 × collisions − 2.0 × severe collisions − 0.25 × slips.
J is used only for tuning and E4/E5 checkpoint selection; criteria are judged on the separate metrics above.

4.4 **Stop conditions:** E0 failure; a detected leakage or a confirmatory runner refusing to run; a compute cap
being exceeded; a critical validity issue found after a confirmatory run (reported, never silently re-run).
Re-running a confirmatory test with changed thresholds, rewards or seeds is forbidden.

4.5 **Compute limits:** this container (4 vCPU, no GPU) with a programme budget of 60 CPU-hours; no paid cloud
without the user's approval of a specific ceiling. Long jobs checkpoint and record machine, package versions, start/
end times and resources.

## 5. Final report

Answers Q1–Q7 of the brief and ends with exactly one of GO / PIVOT / REDESIGN / STOP, justified by the gate results.
