# APM v0.3 master preregistration: architecture discovery and model selection (programme G, G0–G8)

Status: **frozen** at the commit that adds this file (2026-10-12), written after programme F's confirmatory results
(`experiments/F_behavioral_learning/results/CONFIRMATORY_SUMMARY.md`) and before any programme-G pilot, tuning or
test data. Later changes only as dated entries in `DEVIATIONS.md` or as per-experiment addenda
(`prereg/G*_ADDENDUM.md`) committed before the confirmatory run they govern. An addendum may (a) fix sample sizes
from pilot variance with the rules below, (b) fix implementation details left open here, or (c) make a criterion
stricter. It may never relax a margin, change a test seed, change a primary metric or add a comparison to a primary
family after the test data exist.

## 0. Purpose and stance

Find the **smallest** learning and regulatory architecture that makes the robot behave better through experience.
Every mechanism (PAD, appraisal, persistent state, entity memory, neural regulators, continual learning) is a
hypothesis that must earn its complexity against strong, equally tuned simpler alternatives. A negative result is a
valid outcome. No claims about emotion, feelings or personality; "affect" names a control state.

Programme F findings are preserved and never re-tested as if new: E2 (B2 beat B1, −25% collisions; B2 had 9× the
tuning budget: 162 vs 18 configurations), E3 (ranking yes, revision no; avoidance censors evidence), E4 (learned B3
beat B1, not B4; B4 still improving at 1.2M decisions), E7 (fault criteria passed against a weak B0 reference).
F's partition X is never used again for selection or testing.

## 1. Testbed and information contract

- Testbed: `apm_testbeds.wheeled_ctrl` (unchanged), harness `apm_research.harness` (unchanged), decision period
  0.5 s, actions GO/SLOW/YIELD/DETOUR, the identical safety supervisor. G6 adds procedural variations through a
  separate world subclass; the nominal world stays bit-identical.
- Every system receives the same `Obs` stream (20 Hz) and nothing else; ground truth is read only by evaluation
  code. Systems built on the transparent rule use programme F's frozen consequence predictor
  (`runs/E2/predictor.json`, SHA-256 `0735d854…be314`) and the same explicit memory unless memory is the factor under
  test. Learned systems get the same features (`apm_research.features`) plus, where stated, the same memory features.
- **Tuning utility** (as F §4.3): J = goals − 1.0·collisions − 2.0·severe − 0.25·slips per lifetime. J is used for
  tuning, early stopping and checkpoint selection only; criteria use the separate metrics in §2.

## 2. Metrics and statistics (all experiments)

- **Unit = robot lifetime** (never a control step). Paired designs: all systems run on the same lifetime seeds
  (common random numbers). 95% CIs: percentile, paired cluster bootstrap over lifetimes, 10,000 resamples, seed 0;
  with several training seeds, two-level (lifetimes shared by both arms, then seeds within arm). Ratio metrics are
  ratios of totals. Relative difference r = A/B − 1.
- **Behavioural metrics** (every closed-loop experiment): collisions per minute, per 100 m travelled and person
  collisions per relevant encounter (person clearance < 1.0 m) — so that safety bought purely by reduced exposure is
  visible; severe collisions per minute; goal completion; idle fraction; hesitation (SLOW/YIELD with nobody within
  1.5 m and no obstacle within 0.5 m); action switching and A→B→A oscillation per decision; interventions; decision
  latency; parameter count; training cost (lifetime evaluations, environment steps, CPU-seconds).
- **Superiority rule** ("beats", F's E2 rule): (A) collision r ≤ −10% with one-sided p < 0.025 (after Holm) and
  completion-difference CI lower ≥ −3 pp; or (B) completion difference ≥ +5 pp with p < 0.025 (after Holm) and
  collision-r CI upper ≤ +5%; and in both cases idle-fraction difference ≤ +5 pp.
- **Non-inferiority** ("as good as"): one-sided 95% upper bound of collision r < **+10%** AND one-sided 95% lower bound
  of the completion difference > **−2 pp**; p_NI = the larger of the two one-sided bootstrap p-values.
  **Practical equivalence** additionally requires the mirror bounds (collision r lower > −10%, completion upper
  < +2 pp). A non-significant difference is never called equivalence; otherwise "inconclusive".
- **Multiplicity:** Holm within each experiment's primary family (stated per experiment). Secondary analyses are
  unadjusted and labelled exploratory.
- **Censoring:** adaptation times that are not reached by the end of the lifetime are right-censored; summarised by
  Kaplan–Meier median and restricted mean time (bootstrap over lifetimes), never by imputing the maximum.
- **Unwanted policies** are reported for every system: perpetual yielding (idle/hesitation), over-caution
  (completion loss), avoidance-driven censoring (fraction of encounters with no exposure), action collapse (entropy
  of action use).

## 3. Partitions and seeds

- `experiments/G_architecture_selection/seeds.py`: `SeedSequence(0xA9_0003_2026_1012, spawn_key=(experiment,
  partition, index))`, experiments G0–G8; partitions **P** pilot, **T** train, **V** tuning/validation, **X**
  confirmatory test, **S** shift test. A new programme entropy, so no G world equals any F world (tested).
- Neural initialisation seeds 0–4 (more only if an addendum says so before training). Bootstrap seed 0. Tuning
  samplers seeded per family (`apm_select.tuning`).
- X and S are used once per experiment, only by runners that check that the experiment's addendum is committed and no
  tracked file is modified, and that record commit, source hashes and output hashes.

## 4. Fairness and budgets

- Same observations, actions, safety supervisor, predictor and (unless under test) memory for every system.
- **Equal tuning budgets:** within an experiment every tuned family gets the same number of configurations
  (G1: K Latin-hypercube draws from a preregistered space) on the same tuning lifetimes, with the same selection
  rule (max mean J). A learned family gets the same number of lifetime evaluations for training + selection.
- Neural comparisons: equal environment-interaction budgets, equal hyper-parameter trials per architecture, equal
  training/validation seeds and lifetimes, approximately matched parameters (±10%) unless stated; parameter-matched
  and compute-matched results are both reported when they differ. All data and compute used to obtain a model are
  counted (including pretraining data and teacher rollouts).
- Winners are never selected on X or S.

## 5. Experiments

### G0 — fair comparison framework (gate)

Question: were B2's gains due to persistent regulation or to unequal optimisation freedom? Deliverables: audit of
F's E2 comparison; instrumented harness (exposure, encounters, switching, hesitation, latency); equal-budget tuning;
integrity tests (identity with F's B1/B2, no behavioural effect of instrumentation, causality of every regulator,
no hidden state in observations, G/F seed independence, equal budgets, degenerate policies cannot win J, replay
controls reproduce live behaviour, condition worlds as specified). **Gate:** all tests pass; otherwise no model
selection.

### G1 — Is PAD necessary, or is simple persistent risk regulation sufficient?

All systems use `argmax_a progress − λc·m(t)·P_col − λs·P_slip` (+ tendency bonuses for the B2 family) with the
frozen predictor and memory; only m(t) differs (`apm_select.regulators`):

| id | system | tuned parameters |
|---|---|---|
| R0 | B1 with F's tuned λ (λc 1, λs 2), untuned here (historical reference) | — |
| R1 | B1, static m ≡ 1, **equal budget** | λc, λs |
| R2 | EMA of the TTC threat signal with collision impulse | λc, λs, τ, k, c |
| R3 | asymmetric (hysteresis) filter | λc, λs, τ_up, τ_down, k, c |
| R4 | Bayesian (Gamma–Poisson) hazard per minute of close exposure, forgetting, UCB | λc, λs, τ_f, z, k |
| R5 | compact learned scalar regulator (8 inputs, bounded multiplier), evolution strategies | w (8), λc, λs |
| R6 | B2 (APM v0.1 PAD + tendencies), **equal budget** | λc, λs, k_pad, k_t |
| R7 | B2 with PAD replaced by the instantaneous appraisal threat u_threat | λc, λs, k, k_t |
| R8 | B2 with PAD replaced by a non-affective low-pass filter of u_threat + collision impulse | λc, λs, τ, k, c, k_t |
| controls | R6 and R_best with the modulation trace of another lifetime (shuffled) or their own trace shifted by one episode | — |
| B0 | always GO (reference only) | — |

Tuning: K = 48 configurations per family (R5: 48 × N lifetime evaluations), N tuning lifetimes of 10 episodes on G1-V
(fixed in the addendum; default N = 30). R_best = the family among R2–R5 with the highest validation mean J
(simple families only; the choice is written to `configs/G1_selected.json` before X exists).

**Primary family** (partition X, nominal, Holm over five p-values):
- **P1** R6 beats R1 (superiority rule): tests whether B2's advantage survives equal tuning.
- **P2** R_best is non-inferior to R6 (+10% collision, −2 pp completion).
- **P3** R_best beats R1: tests whether persistent regulation helps beyond a fairly tuned static penalty.

Decision: if P2 holds, prefer R_best (PAD is not needed for this benefit). If P2 fails and R6 beats R_best
(secondary), investigate which PAD dynamics matter (R7/R8 vs R6, secondary). If P1 fails, F's E2 gain is not
reproduced under equal tuning ("unequal optimisation explains it"). Removing PAD from B2 making things worse is not
evidence that PAD is necessary.

Secondary: all pairwise comparisons with R1 and R6; negative controls; exposure-normalised rates; hesitation,
switching, oscillation, latency; the safety–efficiency frontier (validation tuning tables and test descriptives);
robustness on partition S under conditions nominal, estimator miscalibration ×3, dense (4–6 people of 8) and sparse
(1–2) with the selected configurations unchanged (no retuning), each compared with R1 under the same condition and
with its own nominal performance.

Sample size: N_X from the pilot (§6), bounds 200–400; N_S = 100 per condition.

### G2 — Which memory can learn and revise experience?

Systems: M0 cumulative `EntityMemory`; M1 exponential forgetting; M2 sliding window; M3 hierarchical Bayesian
(individual within type, with uncertainty); M4 change-point aware (Bayesian online change-point, Adams & MacKay 2007);
M5 hidden-state filter (two-state HMM on disposition with switching probability); M6 small recurrent per-entity
memory (learned, matched information). Each memory feeds the same B1 rule (G1-selected regulator held fixed).

**Selective feedback.** Each encounter is classified from observations only: *exposed* (person within 0.5 m while
the robot moved > 0.15 m/s), *collision*, *unexposed* (never closer than 0.5 m, or only while the robot was stopped),
*avoided* (unexposed and the robot chose SLOW/YIELD/DETOUR when the person was within 1.5 m), *unreliable* (identity
confidence < 0.5 or a sensor gap). Only exposed encounters and collisions are evidence; no counterfactual labels.
Evidence models are a factor (all encounters vs exposed only) for M0 and the selected memory. Information gathering
is a separate factor: none; bounded uncertainty-guided probing (approach at SLOW only when the identity's posterior
uncertainty is high and nobody else is near; probe budget preregistered); a probe-count-matched random baseline.
No dangerous random interactions.

Scenarios (as F's E3, new seeds, plus): risky→safe, safe→risky, frequent vs rare identities, identity switching
errors, newcomers of known type, previously avoided identities, sensor gaps, confounded co-occurrence without
contact. Metrics: adaptation time (Kaplan–Meier, censored), per-encounter Brier and calibration, predictive regret
vs Oracle, false blame, unnecessary avoidance, collisions (closed loop), memory size.

**Primary family** (Holm): for the selected memory M* vs M0: (i) S1 and S2 KM median adaptation ≤ 6 episodes with
the upper CI ≤ 10; (ii) collision rate non-inferior to M0 (+10%, −2 pp); (iii) false blame within ±0.03. M* = the
smallest (fewest state variables per identity) memory that meets (i)–(iii) on validation; the test confirms it.
Whether M6 beats the best statistical memory is secondary.

### G3 — What persistent internal state is needed?

Systems with matched information and actions: none; exponential smoothing; linear dynamical system; single GRU state;
explicit fast + slow recurrent states; PAD (R6's state); a learned latent regulator state. Controlled-event episodes:
sudden danger, recovery after danger, retention across long gaps, changed circumstances, separation of transient
state and long-term memory. Tests: state reset, history shuffle, identity shuffle; *intervention tests* (clamp or
replace the state and measure the behavioural effect) and *probe tests* (does the state predict future collisions
beyond current features, E1-style Brier with lifetimes as clusters). **Primary:** does any persistent state beat
"none" (superiority rule) in closed loop, and does a separate slow state beat the best single-timescale state
(superiority)? Holm over these two.

### G4 — Architecture tournament (only mechanisms supported by G1–G3)

A0 strong non-neural (predictor + best G2 memory + best G1 regulator) — the bar; A1 APM v0.1 with the best memory
(labelled as a modified variant); A2 ordinary GRU actor-critic; A3 GRU + best memory features; A4 learned residual
regulator (bounded utility/risk adjustments on top of A0); A5 explicit fast/slow regulator (only if G3 supports
distinct timescales); A6 compact model-based policy (short-horizon search over high-level actions with the learned
consequence model and uncertainty, only if justified). Shortlist ≤ 4 neural candidates chosen on training/validation
evidence only. **Primary:** each shortlisted neural candidate vs A0 (superiority rule, two-level bootstrap, 5
seeds), Holm over the shortlist. If none beats A0, A0's family is selected. Prefer the simpler model whenever CIs do
not establish a practical benefit.

### G5 — Is the learning algorithm the bottleneck?

PPO from scratch; behaviour cloning of a validated safe teacher (A0); BC → PPO; DAgger with safety restrictions;
model-based action selection; residual policy learning around A0. Comparable pretraining conditions for every
architecture compared. All data and compute counted. Metrics: learning-curve area, final held-out performance,
sample efficiency, seed variance, collapse, action distribution, cost. Includes a diagnosis of F's B3/B4 action use
(GO/YIELD collapse vs optimal response), done on F's training records and new validation runs only.
**Primary:** for the best G4 architecture, best training method vs PPO from scratch (superiority rule on X at equal
or lower total interaction count) and best-trained ordinary GRU vs A0 (non-inferiority).

### G6 — Out-of-distribution robustness

Held-out procedural variations (never used in training or tuning): arena size, obstacle density, pedestrian
behaviours and movement patterns, speed limits, estimator calibration (×0.5, ×3), sensor noise, observation delay,
identity uncertainty, friction/control response, deadline shift, unfamiliar hazard mixtures. No per-condition
retuning. Interpolation, shift and post-adaptation results reported separately; faulted systems compared with strong
baselines (A0, R1), their own nominal performance and absolute safety limits — not only B0. **Criterion:** a
candidate is eligible for selection only if it is non-inferior to A0 in ≥ 80% of conditions and never exceeds 2× A0's
collision rate in any condition.

### G7 — Lifelong learning and cross-embodiment (gated)

Only candidates surviving G4–G6. Lifelong: frozen weights + dynamic memory; periodic updates; shadow updates with
replay and a promotion gate; one alternative continual-learning method if justified; genuinely changing environments
with frozen old/new evaluation suites; corrupted experience, false outcomes, rollback and memory recovery. Updates
rejected by the gate are not counted as learning. Cross-robot: only signals with comparable semantics (NERVA has no
people or identities; no social transfer claimed); scratch, frozen shared encoder + heads, fine-tuning, zero-shot;
negative transfer and data efficiency. NERVA stays observe-only.

### G8 — Minimisation and final decision

Ablate PAD, appraisal features, entity memory, episodic memory, slow state, fast state, neural prediction, learned
utility corrections, explicit uncertainty and continual updates from the strongest credible architecture. Report
performance, parameters, latency, RAM, training cost; final Pareto frontier (behaviour vs complexity, compute,
robustness); final choice of exactly one of A–E (§8).

## 6. Sample sizes

From pilots on partition P only: N = the smallest number of lifetimes giving ≥ 80% power for the experiment's
primary comparisons (superiority at r = −10% / +5 pp; non-inferiority at the margins with a true difference of 0),
from pilot SEs; bounds per experiment in the addendum (G1: 200–400). If a cap binds, the achieved power is reported
and any non-significant result is "inconclusive", never "equivalent".

## 7. Compute and stop rules

Pilots run locally where feasible. Paid cloud only within an explicit dollar approval from the user, phase by phase
(`reports/COMPUTE_PLAN.md`): private VMs, no committed cloud identifiers, hard maximum run duration, automatic
deletion, resumable jobs, private storage, teardown audit, and costs recorded when billing is visible. Stop
conditions: a failed gate, a detected leakage, a confirmatory runner refusing to run, a budget limit reached, or a
validity problem found after a confirmatory run (reported, never silently re-run).

## 8. Final decision (reports/ARCHITECTURE_DECISION.md)

Answers the ten questions of the brief with a ranked table of surviving architectures and selects exactly one of:
A simple predictive regulator; B standard learned recurrent architecture; C specialised learned affective regulator
(only if it significantly and practically beats conventional alternatives); D modular architecture (only if each
module independently justifies itself); E no architecture selected. The simplest option consistent with the
evidence is chosen.
