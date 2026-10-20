# APM v0.5 master preregistration: autonomous revision under self-selected exposure (programme I)

**Status: frozen** at the commit that adds this file (2026-10-17).

**What existed before freezing:** only pilot data from partition P (I0), described in §1 and used for design. No
T, V or X data existed. The T, V and X partitions of I1 and I2 have never been generated.

**Rules for later changes:**
- Changes appear only as dated entries in `DEVIATIONS.md`, or as addenda in `prereg/`.
- An addendum is committed before the run it governs. It may fix tuned values, sample sizes or implementation
  details left open here, or make a criterion stricter.
- An addendum may never relax a criterion, change a test seed or primary metric, or add comparisons to a primary
  family once test data exist.

## 0. Question and stance

**Question.** Can a robot that chooses its own exposure keep its beliefs about individual people accurate when
they change, without becoming less safe or less efficient? This includes changes it can only notice by approaching
someone it has learned to avoid: the *avoidance trap*.

**Why it matters.** Programmes F (E3) and G (G2, G2b) found that no memory revised within 6 episodes. Both
attributed this to avoidance censoring the evidence.

**Stance:**
- Every mechanism here must beat decision A's memory (A0) and G2b's revisable memory under the criteria below.
- Negative results are valid outcomes and will be reported.
- "Belief" means a statistical estimate; no claim about minds or emotions.

## 1. What the pilot (I0, partition P) showed, and how it shaped this design

Every number in this section comes from partition P, with N = 12–48 lifetimes per cell. None is a result of this
programme. The confirmatory tests are §6.

1. **Evidence sparsity, not avoidance, starves revision** (`scripts/i0_evidence_pilot.py`; 15,808 encounters under A0).
   - Collisions separate careless from careful people by only about 0.005 nats per encounter (collision rates 2.3% vs
     1.1%).
   - The minimum clearance while the robot moves is self-observed and needs no ground truth. A near-miss under
     0.15–0.2 m carries 1.1–1.5 nats per encounter, about 250× more.
   - That evidence is available in 76% of encounters, including most encounters where the robot yielded.
2. **Types can be learned without labels.** EM on A0's own pilot experience, with BIC, selects K = 2 person types from
   clearance-coded evidence (π ≈ 0.45 for the higher-risk type, matching the hidden careless share of about 0.43).
   From collision-only evidence it selects K = 1: no individual differences are detectable.
3. **The trap is weak in G's families.**
   - A0's avoidance of people barely depends on memory: even the Oracle avoids switched people as often as A0 does.
   - A memory-driven keep-away (stop for believed-risky people within 2–3 m) raised censoring only from 24% to 30%.
   - Hence §2 adds S10 (one person at a time), where keep-away censors about half of all encounters.
4. **G's adaptation metric can be gamed.** G compares an identity's p with the median p of the population, which
   includes unmet identities at the prior. So a memory that forgets everything "adapts".
   - The primary adaptation metric (§5) is therefore calibrated per system.
   - It is paired with F's discrimination criterion.
5. **Possible inversion in G2b's memory.** G2b's revisable memory (exposure-gated time decay, "A0_mem") ranked careless
   people *below* careful ones (S3 AUROC 0.20, N = 12). This is a secondary hypothesis here (§6.3).
6. **Thompson sampling with per-encounter draws froze the strict keep-away robot in S10**: 99% censored, completion 0.16.
   It is kept as a candidate so that this failure is tested, not assumed.

## 2. Testbed, information contract and scenario families

**Testbed and rule (unchanged from G):**
- `apm_testbeds.wheeled_ctrl` and `apm_research.harness`; the safety supervisor is identical for every system.
- Decision A's rule, unchanged: G's frozen predictor (`runs/E2/predictor.json`, SHA-256 `0735d854…be314`), the EMA
  regulator and A0's weights (`G_architecture_selection/configs/G6_systems.json` → `A0`).
- Only the memory, keep-away and exploration differ between systems.

**Information:**
- Systems receive `Obs` only.
- Ground truth (dispositions) is read only by evaluation code, and by the Oracle reference, which is never a
  candidate.

**Families:**
- G's S1–S9 (`apm_select.scenarios`, unchanged): 30 episodes of 60 s, switch at episode 15.
- **S10** (`apm_revise.scenarios`): exactly one person per episode, always ped-0, careless until episode 15 and careful
  after.

**Seeds:** `seeds.py`, new programme entropy `0xA9_0005_2026_1017`; partitions P, T, V, X, S; experiments I0–I9.

## 3. Systems

### 3.1 Memory (`src/apm_revise`)

**Evidence.**
- An encounter is informative if the robot moved (> 0.15 m/s) while the person was within 1.5 m for ≥ 3 frames, or if
  a collision occurred. Otherwise it is censored.
- *Dense* coding: collision, or the minimum clearance in one of 7 bins (edges .05, .10, .15, .20, .30, .50 m).
- *Sparse* coding: collision or not.

**Type model.**
- K latent person types, fitted by EM on T (§4); K ∈ {1, 2, 3} chosen by BIC for the dense coding.
- The sparse comparator uses K = 2, because BIC for sparse evidence may select K = 1, which would leave no individual
  memory to compare.
- Types are ordered by their collision rate per encounter, r_k.

**Belief (per identity):** a continuous-time Markov change model.
- Between encounters, b relaxes toward the type prior π at rate λ.
- Each informative, reliable encounter (identity confidence ≥ 0.5) multiplies b by the category likelihood, raised to
  the power of the confidence.
- Outputs to the rule: p = b·r, and confidence = 1 − H(b)/log K.

### 3.2 Robots

| family | coupling of memory to behaviour |
|---|---|
| weak (I1) | decision A's rule; memory enters only through the predictor's memory features (as in G) |
| strict keep-away (I2) | the same rule, plus: YIELD whenever the nearest reliably identified person is within 3.0 m and their p ≥ 1.5 × the memory's population-prior p (any direction of motion) |

### 3.3 Exploration (I2), once per encounter, budget 2 per episode

When keep-away would stop the robot for person e, exploration may *probe* instead: e is exempt from keep-away for the
rest of the encounter, and the rule acts with speed capped at SLOW.

| mode | probe if |
|---|---|
| `none` | never |
| `uncertainty` | memory confidence < conf_max (G2b's criterion) |
| `random` | with probability p_random (budget-matched control) |
| `voi` | ν·[R(b) − E_c R(b′(c))] > U(YIELD) − U(SLOW), with R(b) = 1 − max_k b_k and U the rule's utilities |
| `thompson` | no probe rule; keep-away uses p of a type sampled from b once per encounter |

### 3.4 Systems

**I1 (weak coupling):**

| name | memory | role |
|---|---|---|
| A0 | F's cumulative memory (G's M0, all encounters) | **reference** |
| G2b | G's M1t (τ_s 356.35 s, exposure-gated) + G2b's uncertainty probing (budget 2, conf_max 0.5), frozen from G | prior best |
| R_sparse | type belief, sparse coding, K = 2, λ tuned | evidence ablation |
| R_static | type belief, dense coding, λ = 0 | forgetting ablation |
| **R_dense** | type belief, dense coding, K by BIC, λ tuned | **candidate** |
| Oracle | true disposition's rate (G2's Oracle) | evaluation reference only |

**I2 (strict keep-away):**

| name | memory | exploration | role |
|---|---|---|---|
| KA_A0 | A0 | none | **reference** |
| KA_G2b | G's M1t (exposure-gated) | uncertainty, conf_max 0.5 | prior best |
| KA_R | R_dense | none | no-exploration control |
| **KA_R_voi** | R_dense | voi, ν tuned | **candidate** |
| KA_R_unc | R_dense | uncertainty, conf_max tuned | |
| KA_R_rand | R_dense | random, p budget-matched to KA_R_voi | targeting control |
| KA_R_ts | R_dense | thompson | |
| KA_Oracle | Oracle | none | evaluation reference only |

## 4. Training, tuning and selection (partitions T and V only; equal budget K = 16 per tuned parameter)

**Type models.**
- A0 (weak coupling) on S3, I0-T lifetimes 0–399 (`i_run.py`). Fit with `i_fit.py`: EM with 10 restarts and 300
  iterations, K ∈ {1, 2, 3} by BIC.
- The fitted files are committed with the I1 addendum. No dispositions are read.

**λ** (`i_tune.py lam`), separately for R_dense and R_sparse:
- 16 log-uniform values in [1/5000, 1/50] s⁻¹;
- I1-V lifetimes 0–39 of S1, S2, S3 and S10;
- objective: mean log predictive probability of each informative encounter's category before its update. This is
  self-supervised: no ground truth.

**ν, conf_max, p_random** (`i_tune.py explore`; strict keep-away; I2-V lifetimes 100–139 of S1, S3 and S10):
- ν ∈ 16 log-uniform values in [0.01, 10]; conf_max ∈ 16 uniform values in [0.1, 0.95].
- Objective: mean J (G's tuning utility), subject to avoidable collisions ≤ 1.10 × KA_R's on the same lifetimes. If
  nothing qualifies, take the value with the fewest avoidable collisions.
- p_random ∈ 16 values in [0.02, 0.6]: the value whose mean probes per lifetime is closest to the selected VOI's.

All selections, and the systems files written from them, are committed in addenda before any X data.

## 5. Metrics (unit = lifetime; paired designs with common random numbers)

**Adaptation (primary form, system-calibrated).**
- Calibration: from the system's own S3 lifetimes in the same run, take μ_careless and μ_careful, the mean
  end-of-episode p of truly careless and careful identities over episodes 10–29. The threshold θ is their midpoint.
  The "careless side" is the side of θ where μ_careless lies.
- For each switched identity (S1, S2, S10): it is *established* if it lies on the pre-switch side at the end of
  episode 14. Its time is the number of episodes from 15 until the end-of-episode p first lies on the post-switch
  side. If that never happens, it is censored at 15.
- Summaries: Kaplan–Meier median with a bootstrap 95% CI over lifetimes (2,000 resamples), and RMST (horizon 15).
- G2's median form is reported as secondary.

**Discrimination.** S3 ranking AUROC (F's E3 definition: per lifetime, final p vs true disposition), mean with a
bootstrap 95% CI over lifetimes.

**False blame (S5).** F's definition, normalised by the system's |μ_careless − μ_careful|. Raw values are reported too.

**Safety.** Avoidable collisions per minute (G9's definition), as a ratio of totals.

**Efficiency.** Goal completion, as a ratio of totals.

**Also reported:**
- careless collisions per minute;
- censored share (dense coding, logged identically for every system);
- probes and keep-aways per lifetime;
- per-family values;
- G2's median-form adaptation.

**Statistics.** G's paired cluster bootstrap over lifetimes (10,000 resamples), with G's non-inferiority margins:
avoidable collisions +10% (relative), completion −2 pp.

## 6. Confirmatory tests (partition X)

**N.** 200 lifetimes per family per system (indices 0–199). This is the pilot's precision requirement with a generous
margin; compute is not limiting.

### 6.1 I1 (weak coupling): R_dense vs A0

R_dense **passes** only if all five criteria hold. C3 and C4 are adjusted by Holm over the two tests.

| criterion | requirement |
|---|---|
| C1 adaptation | S1, S2 and S10 each: Kaplan–Meier median ≤ 6 episodes and 95% CI upper bound ≤ 10 |
| C2 discrimination | S3 ranking AUROC ≥ 0.80, with 95% CI lower bound ≥ 0.70 |
| C3 safety | avoidable collisions per minute, pooled over S1–S10, non-inferior to A0: one-sided p for the +10% margin < 0.05 |
| C4 efficiency | completion, pooled over S1–S10, non-inferior to A0 (−2 pp margin) |
| C5 false blame | normalised S5 false blame: point estimate within ±0.25 and 95% CI within ±0.5 |

**Secondary, labelled exploratory:**
- the same criteria for G2b, R_sparse and R_static;
- R_dense vs R_sparse (evidence density) and R_dense vs R_static (forgetting), on adaptation RMST and discrimination;
- every system vs the Oracle on collisions.

### 6.2 I2 (strict keep-away): primary family, Holm over three one-sided tests

| test | hypothesis | where |
|---|---|---|
| P1 | KA_R_voi completion > KA_R (exploration reduces the trap's efficiency cost) | S10 |
| P2 | KA_R_voi avoidable collisions non-inferior to KA_R (+10%) | pooled S1–S10 |
| P3 | KA_R_voi completion > KA_R_rand (targeting matters at the same budget) | S10 |

Also: KA_R_voi against KA_A0 on criteria C1–C5 (the analogue of §6.1). This decides whether autonomous revision under
real avoidance is achieved. The other explorers are secondary.

### 6.3 Secondary hypothesis (G2b inversion)

G2b's S3 ranking AUROC < 0.5 in I1, with 95% CI upper bound < 0.5. If confirmed, this is reported as a defect in
programme G's recommended memory upgrade.

## 7. Decisions

| outcome | decision |
|---|---|
| I1 passes | the dense-evidence, change-aware type memory becomes APM v0.5's memory recommendation, replacing F's cumulative memory |
| I1 fails | report which criterion failed; no memory change |
| P1–P3 all pass | recommend VOI probing wherever memory drives avoidance |
| P1 and P2 pass, P3 fails | exploration helps, but its targeting is not shown to matter |
| other I2 outcomes | report |

**Reporting rules:**
- Every pilot-to-test discrepancy is reported.
- Thompson sampling's failure, or success, is reported either way.

## 8. Compute and provenance

**Compute:**
- Local laptop for pilots.
- Capped Google Cloud spot VMs (`cloud/icloud.py`, `cloud/runner.sh`) for T, V and X.
- The user approved a US$40 cap; spend is logged in `reports/COMPUTE_LOG.md`.

**Provenance:**
- X runs are submitted with `--commit`: a clean clone of HEAD, with `apm_revise.provenance.confirmatory_lock` passing
  and source hashes recorded.
- Raw lifetime files stay under `runs/` (git-ignored). Analysis outputs and manifests (SHA-256 of raw files) are
  committed.
