# APM v0.4 master preregistration: developmental expressive intelligence (programme H, H0–H8)

Status: **frozen** at the commit that adds this file (2026-10-12). It was written after the repository and NERVA
audit (`reports/H0_AUDIT.md`) and before any programme-H pilot, tuning or test data.

Later changes are allowed only in two forms:
- dated entries in `DEVIATIONS.md`;
- per-experiment addenda (`prereg/H*_ADDENDUM.md`), each committed before the confirmatory run it governs.

What an addendum may do:
- fix sample sizes from pilot variance, using the rules below;
- fix implementation details, feature definitions and tasks left open here;
- make a criterion stricter.

What an addendum may never do: relax a margin, change a locked test seed, change a primary metric, or add a
comparison to a primary family after the test data exist.

Programme G's preregistration and results are frozen and are not edited here.

## 0. Purpose and stance

**Question.** Can a robot develop persistent, experience-dependent internal states and express them through
physically embodied movement? What is the smallest mechanism that does this?

Every candidate mechanism has to earn its complexity against strong simple alternatives:
- persistent state;
- PAD;
- recurrent latents;
- fast/slow networks;
- learned expression mappings;
- learned motion latents.

The decisive comparison is against the strong simple baseline:
- the frozen v0.3 architecture A0 (learned consequence predictor, explicit memory and calibrated risk regulation),
- plus a fixed, transparent, safety-bounded expressive-style lookup.

A negative result is a valid outcome.

**No claims about emotions, feelings, consciousness or personality.** Style descriptors ("cautious", "hesitant",
etc.) are experimental labels for motion programmes, not states of the robot.

### Three separate claims (never merged)

- **Claim A, expressive motion.** The robot produces observably different, reliably controllable movement
  qualities while staying safe and task-competent. Measured in simulation (H0, H1).
- **Claim B, interpretable expression.** Independent human observers identify intended qualities better than
  chance (H2).
  - This requires real participants and the necessary ethics approval.
  - Without both, claim B is reported as **untested**.
  - LLM-generated or model-generated ratings never substitute for human observers. Automated observers are
    exploratory tools only.
- **Claim C, developmental expression.** Actual experienced outcomes change the robot's internal state and, through
  it, its later expressive behaviour. The change must be:
  - stable;
  - context-appropriate;
  - consistent with the evidence;
  - revisable;
  - caused by the history rather than by present input (H3, H4, H6).

A does not establish B. B does not establish C. C does not establish subjective emotion.

### Frozen findings carried in (not re-tested as new)

- **v0.3 decision A** (`../G_architecture_selection/reports/ARCHITECTURE_DECISION.md`):
  - PAD is harmful for risk regulation under fair tuning;
  - an EMA risk state beats a static penalty;
  - explicit memory is needed;
  - neural regulators are not better than A0;
  - revision under selective feedback is hard;
  - cross-robot learned transfer and useful deployment-time learning are not shown.
- **NERVA:**
  - the neutral turn-translation candidate passes the long neutral motor gate;
  - E1′ keeps that gate at e = 0, with a small (1–3.4°) monotonic pitch channel while moving and an ignored
    crouch channel;
  - S1–S6 and E1 failed their expressive criteria;
  - runtime head offsets cost up to 84% of the upstream policy's walking speed;
  - no human evaluation has ever been done.

## 1. Architecture under study and its boundaries

```
experience (observed outcomes) → memory / internal state → ExpressionIntent
    → EmbodimentExpressionAdapter → ValidatedMotionController → physics
                       ↑ independent SafetyMonitor + neutral fallback (cannot be overridden by the intent)
```

- **`ExpressionIntent`** is an embodiment-independent, bounded vector of movement qualities, plus an optional
  attention target. It never contains joint targets.
- **The adapter** maps an intent onto the embodiment's *validated* channels only. It clips every channel to its
  admitted range, rate-limits changes, enforces minimum command dwell times, and can always return neutral motion.
- **`ValidatedMotionController`** for the biped is a frozen NERVA ONNX policy behind NERVA's neutral motor
  contract:
  - the validated neutral candidate (SHA-256 `610b1c59…`), or E1′ (`f67d5cac…`, gate-validated at e = 0);
  - discrete trained commands, fixed 0.54 s gait clock;
  - no retraining in programme H unless a separate addendum and budget authorize it.
- **The `SafetyMonitor`** reads only the robot's physical state (tilt, angular rate, joint-limit margins, actuator
  force saturation). It never reads intents, memory or internal state. When it trips, the controller falls back to
  neutral rest with zero expressive offsets until the robot is calm.
- **Unrestricted neural joint commands are never used.** Learned components may only output an
  `ExpressionIntent`.
- **Everything new lives in a removable package (`src/apm_express/`).** APM v0.3 code (`src/apm`,
  `src/apm_research`, `src/apm_select`, `src/apm_testbeds`) is not modified unless a verified defect requires it,
  recorded in DEVIATIONS. NERVA's repository is used read-only; its controller and contracts are not changed.
- **Simulation only.** No physical hardware in programme H.

## 2. Experiments, gates and order

| phase | experiment | question | gate (fixed precisely in the addendum) |
|---|---|---|---|
| 1 | **H0** expressive-action interface | Which motion channels can be changed safely on the validated controller? Does NERVA's stable behaviour reproduce here? | **Gate.** (a) NERVA's neutral gate reproduces on this machine for the validated candidate; (b) E1′'s pitch channel reproduces; (c) ≥ 2 expressive channels other than mean speed are admitted with zero nominal falls; (d) monitor and fallback work. Fail → solve the interface before anything learns. |
| 2 | **H1** movement repertoire (X0–X3) | Can NERVA produce distinct, reliably controllable movement qualities? | **Gate.** A repertoire whose styles are distinguishable from non-speed features across held-out tasks, seeds and perturbations, with no loss of stability or task completion. |
| 2 | **H2** human perception | Can people perceive the distinctions? | Protocol, stimuli and power analysis only. **Pending** until ethics status is determined and the user authorizes data collection. No ratings fabricated. |
| 3 | **H3** internal-state representations S0–S7 | Which smallest state gives coherent, persistent, revisable expressive variation? | Decision rule, §5. |
| 3 | **H4** experience shapes expression | Do different actual histories produce reliable, coherent, history-caused differences in the same situation, beyond fixed-rule and memory baselines? | **Gate.** No developmental claim without a reproducible history-dependent effect beyond the strongest fixed-rule and current-risk baselines, surviving the controls of §5. |
| 4 | **H5** learned expression mapping L0–L5 | Does a learned mapping beat the fixed lookup? | Only after H1 and H4 pass. Simplest-adequate rule, §5. |
| 5 | **H6** dispositions | Do stable, revisable tendencies emerge over long lifetimes? | Repeatability and revisability criteria (addendum). |
| 5 | **H7** cross-embodiment | Does one abstract intent map to interpretable behaviour on a biped and a wheeled robot? | Claims restricted to what is measured; no "learned transfer" because an interface accepts both. |
| 5 | **H8** integrated evaluation | Does the integrated system beat A0 + memory + fixed lookup? | Only gate-passing mechanisms are combined. |

If a critical gate (H0, H1, H4) fails, downstream experiments that assume it are **not run**. The blocker is
reported, and the decision follows §6.

## 3. Learning objectives (kept separate)

- **World learning:** predict observable consequences, such as outcome of an encounter, perturbation magnitude
  and recovery. Training targets are measured outcomes.
- **Regulatory learning:** improve safe long-term decisions. This is v0.3's domain, and A0 is frozen.
- **Expressive learning:** acquire controllable movement qualities that communicate discernible differences while
  keeping the task competent.
  - Its supervision must be stated explicitly: reference styles, human perception data, or a self-supervised
    motion objective.
  - Ungrounded labels ("fear", "joy") are never used as if they were measured outcomes.
  - No reward called "emotion quality" is invented.
  - Better world prediction alone never justifies an expressive representation.
  - Where human perception data would be required and are unavailable, the experiment says so and restricts its
    claim.

## 4. Partitions, seeds and locks

- **Seeds** come from `experiments/H_expressive_intelligence/seeds.py`:
  `SeedSequence(0xA9_0004_2026_1012, spawn_key=(experiment, partition, index))`.
  - Experiments H0–H8.
  - Partitions: **P** pilot, **T** train, **V** tuning/validation, **X** locked test, **S** shift/OOD test.
  - The new entropy keeps every H world distinct from F and G worlds.
  - NERVA simulator seeds are derived from these (`sim_seed`).
- **Reproductions of NERVA protocols** use NERVA's own fixed seeds; they are reproductions, not new tests.
- **X and S are used once per experiment**, and only by runners that do all of the following:
  - check that the experiment's addendum is committed;
  - check that no tracked file is modified;
  - record commit, source hashes and output hashes (same lock as programme G).
- Neural initialisation seeds 0–4 unless an addendum says otherwise before training. Bootstrap seed 0.

## 5. Metrics, statistics and decision rules (common)

### Units and statistics

- **Unit of analysis:**
  - the trial (one controlled motion episode) for motion experiments (H0, H1);
  - the robot lifetime for developmental experiments (H3–H8).
- **Paired designs** with common random numbers throughout.
- **95% CIs:** percentile paired cluster bootstrap over units, 10,000 resamples, seed 0. Holm correction within
  each experiment's primary family. Secondary analyses are unadjusted and labelled exploratory.

### Motion metrics (every closed-loop motion experiment)

| group | metrics |
|---|---|
| stability | falls (tilt > 45° or non-finite state), max tilt, monitor trips, fallback time |
| task | completion within the time limit, time to goal, final distance |
| temporal structure | pause count and duration, stop-go rate, command-switch rate, moving fraction |
| posture | mean and range of torso pitch, head joint angles (where the channel is admitted) |
| acceleration | RMS base acceleration, jerk, spectral arc length of the speed profile (SPARC) |
| spatial | path length / straight-line ratio, lateral deviation, approach distance, orientation to target, retreat distance |
| cost | mechanical power, Σ τ² |

### Distinctness ("reliably controllable", H1)

- Cross-validated classification of the intended style from motion features.
- Folds are held out by task condition and seed, never by time step within one trial.
- Reported per feature group, and again **with all mean-speed and progress features removed**.
- Matched-speed pairs are reported separately.
- A style counts only if it is distinguishable by more than its average speed.

### Developmental metrics (H3–H6)

- **History effect:** the difference in intent and motion features between robots with different histories in
  identical evaluation situations, paired by situation seed.
- **Causal controls:** every history effect is compared with the effect under each of these:
  - identical present observations (built in);
  - shuffled outcome histories;
  - randomised identity labels;
  - memory removed;
  - state reset before evaluation;
  - current-risk-only baseline;
  - fixed style;
  - human-designed history-to-style lookup.
- **Evidence-consistency:** rank correlation between expressed quality and the outcome statistics actually
  experienced. Arbitrary divergence does not count.
- **Persistence:** half-life of an effect after the experience stops.
- **Revisability:** time for an effect to disappear or reverse after contradictory evidence. Censored times are
  summarised with Kaplan–Meier.
- **Dispositions (H6):** repeatability (intraclass correlation across matched contexts and time) and revisability.

### Decision rules

- **H3 state selection.** The smallest representation whose primary expressive criteria are non-inferior to
  the best one. Non-inferiority margins are fixed in the H3 addendum. Order of simplicity:
  S0 < S1 < S2 < S3 < S4 ≈ S5 < S6 < S7, with PAD (S4) counted as a 3-D specialised state.
  - PAD is selected only if it beats the simpler and the learned alternatives on expressive criteria.
  - A fast/slow network is preferred over a GRU only if it beats the GRU.
- **H5 mapping selection.** The same rule. A simple trainable mapping is preferred if it is as good as the
  recurrent ones; a GRU is preferred over a specialised network that it equals.

## 6. Final architecture decision (A–E), fixed in advance

Applied in `reports/EXPRESSIVE_ARCHITECTURE_DECISION.md`:

- **E: no validated expressive architecture yet.** Choose this if:
  - H0 or H1 fails; or
  - developmental results are inconclusive in a way that leaves no defensible recommendation beyond "not
    established".
- **A: fixed expressive styles.** Choose this if H0 and H1 pass but the H4 gate fails, so developmental
  expression is unproven.
- **B: experience-conditioned expressive mapping.** Choose this if:
  - the H4 gate passes with explicit memory feeding a small mapping (fixed L0 or small learned L1); and
  - no recurrent latent beats it on the H4/H8 primary criteria.
  The report states whether the mapping is fixed or learned.
- **C: learned recurrent expressive regulator.** Choose this if a recurrent latent (S5–S7, L2–L3) beats the best
  memory + mapping system on the preregistered H4/H8 primary criteria by the addendum's margin, with the causal
  controls intact.
- **D: specialised developmental affective architecture.** Choose this if a genuinely distinct mechanism (PAD,
  fast/slow, learned motion latent) beats both the GRU and the memory + mapping baselines under the same
  conditions. D is never chosen by default or for elegance.

Human interpretability (claim B) is reported separately. Without approved human data, every recommendation states
that interpretability is untested.

## 7. Compute and budget

- **No spending authorization exists for programme H.** Programme G's approval does not carry over.
- Default is local CPU: an i7-1355U, about 3 effective cores. The NERVA simulator measures 15× real time on one
  core (`reports/H0_AUDIT.md`).
- Paid cloud resources are requested from the user first, with a cost estimate and a measured speed-up, and only
  where local execution is impractical.
- When authorized, the same pattern as programme G applies:
  - private VMs with temporary NAT;
  - hard run-duration caps;
  - restartable jobs;
  - checkpoints and hashes;
  - teardown and audit after every phase.
- Identifiers (project, bucket, account, personal paths) are never committed.

## 8. Integrity

- Hypotheses, criteria and experiment code are committed before confirmatory runs.
- Failed runs, deviations and negative results are kept and reported.
- Human studies, ratings, compute measurements and findings are never fabricated.
- Old preregistrations (F, G and NERVA's) are never edited.
- Tests, lint and types pass at every commit. Verified work is committed and pushed incrementally.
