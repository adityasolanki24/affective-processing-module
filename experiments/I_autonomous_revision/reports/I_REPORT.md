# Programme I report: autonomous revision under self-selected exposure (2026-10-17)

**Governed by:** `MASTER_PREREGISTRATION.md` (frozen at `cceb682`, before any T, V or X data) and
`prereg/I12_ADDENDUM.md` (`cc3e77a`, before any X data).

**Deviation:** D1. The random-probe control's range was widened at the V stage so it could be budget-matched
(`DEVIATIONS.md`).

**Results:**
- `results/I1/I1_X.json`, `results/I2/I2_X.json`, `results/V/`;
- raw-data hashes in `manifests/runs.json`.

**`dirty: True` in those records** reflects only the untracked `out/` result folder on the VM. The confirmatory lock
checks tracked files and passed.

## 1. Why revision was slow: evidence, not avoidance

| evidence the robot uses | information per encounter (pilot) | person types EM + BIC finds in the robot's own experience (T, 2,400 people) |
|---|---|---|
| collisions only (F, G) | ≈ 0.005 nats | **K = 1**: individual differences are undetectable |
| self-observed minimum clearance while moving | 1.1–1.5 nats (near-miss < 0.15–0.2 m) | **K = 3**: π = 0.575 / 0.185 / 0.240; collision rate 0.87% / 2.6% / 2.6% |

The robot discovers that people differ, and roughly how, without ever being told anyone's disposition. With collisions
as the only outcome, it would take hundreds of encounters to tell a careless person from a careful one. That explains
F's E3 failure and G2's: every memory model revised slowly because the *evidence* was too thin. Avoidance censoring
was a secondary cause.

## 2. I1: memory under decision A's rule (12,000 test lifetimes, 6 systems × 10 families × 200)

| system | revision S1 / S2 / S10 (median episodes [95% CI]) | S3 ranking AUROC | normalised false blame | avoidable collisions vs A0 | completion vs A0 |
|---|---|---|---|---|---|
| A0 (F's cumulative memory) | not reached / not reached / not reached | 0.80 [0.78, 0.83] | 0.32 | — | — |
| G2b (G's revisable memory) | not reached / 7 [6, 9] / not reached | **0.21 [0.18, 0.25]** | 0.02 | −36.5% [−39.9, −33.0] | −0.65 pp |
| R_sparse (collision-only types) | 1 / not reached / 1 (only 1 of 400 S1 identities established) | 0.50 | 0.00 | **+33.1%** [+27.7, +39.0] | +0.16 pp |
| R_static (dense, no forgetting) | 14 / not reached / not reached | 1.00 | 0.00 | −45.3% [−48.9, −41.5] | −0.10 pp |
| **R_dense** | **2 [2, 2] / 3 [3, 3] / 2 [2, 2]** | **1.00 [1.00, 1.00]** | **0.01** [−0.01, 0.03] | **−55.6% [−58.7, −52.3]** | −0.32 pp [−0.40, −0.23] |
| Oracle (reference) | 1 / 1 / 1 | 1.00 | 0.00 | −19.2% | −0.08 pp |

**Preregistered verdict: R_dense passes C1–C5.**

| criterion | result |
|---|---|
| C1 | adaptation ≤ 6 episodes in S1, S2 and S10 |
| C2 | AUROC ≥ 0.80 |
| C3 | safety non-inferior, Holm p < 0.001 (in fact, superior) |
| C4 | completion non-inferior (lower bound −0.40 pp vs the −2 pp margin) |
| C5 | false blame within bounds |

**Mechanism (exploratory).** Every system has the same rate of collisions with *careful* people, 0.039 per minute. All
of R_dense's reduction is in collisions with *careless* people (0.071 → 0.051 per minute). Idle time is unchanged and
the action mix shifts slightly toward SLOW/YIELD. The robot is not more cautious overall; it is more cautious with the
right people.

**Confidence-channel ablation (exploratory, fresh indices X 1000–1199, 6,000 lifetimes).**
- R_dense again cuts avoidable collisions by −52.4% [−55.9, −48.7] vs A0. **This replicates I1.**
- Reporting F's count-based confidence instead of posterior certainty still gives −42.1% [−46.0, −37.9].
- About four-fifths of the gain comes from more accurate beliefs; the rest from the more informative confidence signal
  the predictor reads.

**Ablations (secondary):**
- **Forgetting is necessary.** R_static never revises after a switch, despite perfect discrimination.
- **Dense evidence is necessary.** R_sparse cannot discriminate (AUROC 0.50), and is less safe than A0.
- **The Oracle is not an upper bound here.** It feeds the predictor collision rates of 0.27 for careless people, far
  outside the range of the memory input the predictor was trained on (≈ 0–0.1). The predictor reacts less to that than
  to R_dense's in-range values.

**Secondary hypothesis (§6.3), confirmed.** G2b's memory ranks people *inverted*: AUROC 0.21, CI upper bound 0.25 < 0.5.

Why it inverts:
- G2b counts only *exposed* encounters (person within 0.5 m while the robot moves) as trials.
- Careless people produce many exposed, collision-free encounters, so their estimated rate falls.
- Careful people are rarely exposed, so they stay near the prior.

G2b's lower collisions than A0 (−36.5%) therefore do not come from correct beliefs. Programme G recommended G2b as a
memory upgrade, conditional on confirmation (G6 and G8). **It should not be adopted.**

## 3. I2: the avoidance trap (strict keep-away robot, 16,000 test lifetimes, 8 systems × 10 families × 200)

The keep-away robot stops whenever a reliably identified person it believes is risky (p ≥ 1.5 × prior) is within 3 m.

| system | S10 completion | S10 censored | revision S1 / S2 / S10 | AUROC | completion (all families) | avoidable collisions vs KA_A0 | probes per lifetime |
|---|---|---|---|---|---|---|---|
| KA_A0 | 0.751 | 0.23 | not reached / not reached / not reached | 0.77 | 0.537 | — | 0 |
| KA_G2b | 0.700 | 0.30 | not reached / 7 / not reached | 0.21 | 0.554 | −22.1% | 5.5 |
| KA_R (no exploration) | **0.597** | **0.57** | 2 / 3 / 3 | 0.99 | 0.524 | −57.2% | 0 |
| **KA_R_voi** | **0.684** | **0.26** | 2 / 3 / 2 | 1.00 | 0.516 | **−61.2%** | 24.2 |
| KA_R_unc | 0.685 | 0.25 | 2 / 3 / 2 | 1.00 | 0.511 | −60.4% | 27.8 |
| KA_R_rand (budget-matched) | 0.677 | 0.30 | 2 / 3 / 2 | 1.00 | 0.514 | −61.1% | 24.0 |
| KA_R_ts (Thompson) | **0.160** | **0.99** | 2 / 3 / not reached | 0.97 | 0.432 | −60.7% | 0 |
| KA_Oracle | 0.322 | 0.86 | 1 / 1 / 1 | 1.00 | **0.203** | −57.6% | 0 |

**Primary family (Holm), all pass, each adjusted p < 0.001:**

| test | result |
|---|---|
| P1 | VOI restores S10 completion vs no exploration: **+8.7 pp [+8.0, +9.4]** |
| P2 | VOI is not less safe than no exploration: avoidable collisions **−9.3% [−16.1, −2.0]** |
| P3 | targeted beats random probing at the same budget: **+0.71 pp [+0.55, +0.88]**. Real, but small |

**Full-system criteria vs KA_A0: KA_R_voi narrowly fails C4.**
- Pooled completion is −1.80 pp [−2.18, −1.42]; the lower bound crosses the −2 pp margin.
- Probing rescues the trap world but costs about 0.8 pp in worlds where it isn't needed: pooled completion is 0.516
  vs KA_R's 0.524.
- C1, C2, C3 and C5 pass.
- KA_R, the new memory without exploration, passes all five.

**Reading:**
1. **The trap is real only when avoidance censors almost everything** (S10 with keep-away: 57% of encounters
   censored). With dense evidence the robot still revised there (median 3 episodes), from the few informative
   encounters left, but at a large efficiency cost.
2. **Value-of-information probing works, but is almost indistinguishable from "probe whenever keep-away would stop the
   robot".**
   - On V the information value exceeded the cost in nearly every eligible case, for every ν.
   - The uncertainty rule and budget-matched random probing perform within 1 pp of it.
   - Targeting adds 0.7 pp in S10.
3. **Thompson sampling with one draw per encounter is a trap of its own.** Once it draws "risky" for a person who is
   always present, the robot freezes for the whole encounter (S10: 99% censored, completion 0.16). It was kept as a
   candidate precisely to test this pilot observation, which the test confirms.
4. **Keep-away itself is the costliest policy.** With perfect knowledge (the Oracle), the keep-away robot completes only
   20% of goals. Avoiding risky people outright is far worse than decision A's graded, consequence-predicting caution,
   which costs 0.3 pp.

## 4. What this means for APM

- **Prompt, accurate revision of what the robot believes about individuals is achieved, in simulation,** by:
  1. learning person types from self-observed, dense outcome signals (near-misses), without labels;
  2. holding change-aware type beliefs that forget toward the population prior;
  3. leaving decision A's rule unchanged.

  It revises within 2–3 episodes in both directions, discriminates perfectly in a stable world, halves avoidable
  collisions and costs 0.3 pp of goal completion.
- **Exploration is only needed if memory drives outright avoidance, and then any reasonable probing rule works.**
  Better still, don't make memory drive outright avoidance.
- **Not shown:**
  - any human-relevant or real-robot result (simulation only);
  - transfer to the biped (NERVA) or any other embodiment;
  - robustness to G6's shift conditions;
  - behaviour when person types are not cleanly separable.

  The wheeled testbed's careless people are scripted to cut across a moving robot. That makes clearance an unusually
  clean signal. On a real robot the separability is unknown.
