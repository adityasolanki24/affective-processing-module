# Programme F confirmatory results: E2, E3, E4, E7 (2026-10-11)

All four ran once on partition X under the committed addenda (`prereg/E*_ADDENDUM.md`), with the confirmatory lock
passing (clean checkout, provenance in `results/E*/…run_meta.json`, raw-data hashes in `manifests/`). Deviations
during execution (machine change, a refused-then-rerun step, the E4 evaluation on a separate VM) are in
`DEVIATIONS.md`; none changed a size, seed, threshold or criterion. Analyses were run locally on the fetched raw data
with the committed `scripts/e*_analyze.py`. CIs are 95% paired cluster-bootstrap intervals (10,000 resamples).

## E2: does the existing APM improve behaviour? **Yes, by criterion A.**

N = 240 lifetimes × 10 episodes, 9 systems (`results/E2/E2_X.json`).

| comparison | collision rate r | completion Δ | idle Δ | decision |
|---|---|---|---|---|
| **B2 vs B1 (primary)** | **−25.4% [−32.0, −18.4]** | +0.3 pp [−0.1, +0.8] | +0.0 pp | **beats (A)**; Holm p_A < 0.001 |
| B1 vs B0 | −94.5% [−95.1, −93.8] | +7.5 pp [+6.7, +8.4] | +0.1 pp | beats (the testbed rewards prediction) |
| Oracle vs B1 | −7.3% [−15.5, +1.7] | +0.5 pp [+0.2, +0.9] | 0 | does not beat (explicit memory is close to the true rates) |
| B2_noPAD vs B2 | +53% [+40, +68] | −0.2 pp | — | removing PAD modulation loses most of the gain |
| B2_noTonic vs B2 | +33% [+23, +44] | −0.3 pp | — | |
| B2_noPersistence vs B2 | +42% [+30, +56] | −0.1 pp | — | |
| B2_noMemoryFrames vs B2 | +0.4% [−1.7, +2.7] | +0.0 pp | — | APM memory frames add nothing |
| B2_shuffled vs B2 (negative control) | +15% [+7, +25] | −0.3 pp | — | another lifetime's affect trace keeps part of the gain |

Reading. The hand-written affective layer, used as a state-dependent multiplier on collision cost, reduces collisions
by a quarter without costing goals or freezing. Caveats: B2 had more tuning freedom than B1 (tuned λc = 0.5 plus
k_pad = 2 vs B1's λc = 1.0), which the design accepts as a bias in B2's favour. The shuffled-affect control still
beats B1 (collision rate 0.113 vs 0.131/min), so roughly half of the gain is generic extra caution, not the
lifetime-specific affect trajectory. "Functional affect" only; no claim about emotion.

## E3: experience, changing relationships, false attribution. **Verdict: "encounter statistics only (no timely revision)".**

N = 60 lifetimes × 30 episodes per scenario family (`results/E3/e3_X.json`); B1 memory primary.

| criterion | result | pass |
|---|---|---|
| (i) S3 ranking AUROC ≥ 0.80, CI lower ≥ 0.70 | 0.889 [0.850, 0.923] | yes |
| (ii) S1/S2 median adaptation delay ≤ 6 episodes | S1: 16 (censored) [14, 16]; S2: 15.5 [9, 16] | **no** |
| (iii) S5 false attribution within ±0.03 | +0.004 [−0.001, +0.008] (naive association memory: +0.137) | yes |
| (iv) careless collisions ≤ 0.8 × B0, completion ≥ −3 pp | r = −95.9% [−96.4, −95.3], +9.4 pp | yes |

Also: under identity errors (S4) AUROC 0.80; newcomer over-generalisation ≈ 0 (B0: +0.32); restart retention exact
for B1 (memory identical, replay and closed-loop decisions identical in 60/60), memory identical but replay agreement
0.977 for B2; two robots with different histories diverge on 3.0% of decisions. Adaptation fails for every system
including the Oracle-driven one (the explicit memory is cumulative, so 15 episodes of history are slow to overturn)
and is worse when the policy avoids the identity (S2: B0, which never avoids, revises in a median 5 episodes): the
censoring limitation predicted in `AUDIT.md`.

## E4: learned vs hand-written regulation. **B3 is not a credible candidate.**

40 networks (8 variants × 5 seeds, 1.2M decisions each), N = 400 test lifetimes × 5 episodes, two-level bootstrap,
Holm over four p-values (`results/E4/E4_X.json`).

| comparison | collision rate r | completion Δ | idle Δ | decision |
|---|---|---|---|---|
| **B3 vs B4** | −11.2% [−20.4, −1.4] (Holm p 0.039) | **−4.0 pp [−5.8, −2.1]** | −1.3 pp | does not beat |
| **B3 vs B1** | −14.8% [−23.4, −5.3] (Holm p 0.007) | +1.2 pp [+0.0, +2.6] | +1.2 pp | beats (A) |
| B3 vs B2 (secondary) | +1.4% [−8.0, +11.8] | +1.0 pp | +1.2 pp | no difference shown |
| B2 vs B1 (secondary, replication on fresh lifetimes) | −16.0% [−23.0, −8.3] | +0.2 pp | 0 | beats (A) |
| B3_noMemory / B3_shuffledIdentity vs B3 | −3.6% / −1.3% (CIs span 0) | ≈ 0 | — | B3 makes no measurable use of explicit memory |
| B3_historyReset vs B3 | +27.7% [+13.5, +44.4] | −0.7 pp | — | the recurrent state matters |

Verdict as preregistered: "no evidence for a specialised regulation structure (B3 beats B1 but not B4)". The
capacity-matched GRU (B4) trades a few more collisions for clearly more goals (J 25.5 vs 24.6). B4 was still
improving at the 1.2M budget (pilot), so this comparison may understate B4. E5 and E6 stay gated (reduced versions
only, per §2).

## E7: robustness and safety independence. **All criteria pass.**

N = 100 lifetimes × 5 episodes × 15 conditions × B0/B1/B2 (`results/E7/e7_X.json`).

- (1) No exception from B1/B2 reached the robot loop under any fault (0 genuine exceptions).
- (2) Under `model_failure`, 6,036 injected failures each produced a GO fallback (B1 and B2); fallback timing is
  unit-tested.
- (3) Supervisor independence: unit test passes.
- (4) Every fault, both systems: collision-rate CI upper bound vs B0 ≤ +5%. The smallest margins are sensor outage
  (B1 −30.7% [−36.8, −24.2]; B2 −21.9% [−30.1, −13.1]) and delayed inference (B1 −84.7%). Caveat: B0 collides ≈ 20×
  more often than B1/B2 in this testbed, so (4) is a weak bar here.

## E1: does affective machinery add predictive information? **No (one inconclusive).** (completed 2026-10-12)

Run from the programme-G session under `prereg/E1_ADDENDUM.md` (wheeled X 100 lifetimes, NERVA X 160 trials; datasets
regenerated bit-identically to F's manifests; locks passed; `results/E1/*_X.json`). Relative Brier vs the baseline
selected on V (R), Holm over U, P, UP; 5 seeds; lifetime/trial clusters.

| embodiment / target | U (appraisal) | P (PAD) | UP | decision |
|---|---|---|---|---|
| wheeled collision (primary) | −0.2% [−4.1, +3.9] | −1.6% [−5.6, +2.2] | −0.8% [−4.3, +2.4] | U, UP: no meaningful information; **P: inconclusive** |
| wheeled stability_loss | +1.5% [−1.5, +4.8] | +1.3% [−1.2, +3.6] | +1.8% [−0.4, +3.8] | no meaningful information (all) |
| NERVA stability_loss (primary) | −1.1% [−2.4, +0.4] | +0.4% [−0.8, +2.1] | −0.7% [−2.2, +1.0] | no meaningful information (all) |

Secondary: entity-memory expectations add nothing (M vs R +1.9% [−1.3, +4.9]); high-rate summaries help NERVA (R vs
G −6.7%) but not wheeled; the noise-width control is worse than R (+3–10%), and the capacity control equals R.
Decision (F's rule): **PAD is no longer treated as a necessary predictive representation** — it adds no measurable
information in two of three settings and is inconclusive (point estimate −1.6%, far from the −5% margin) in the third.

## Not yet done

Reduced E5/E6 are superseded by programme G's G7 (see `../G_architecture_selection/reports/F_ACCOUNTING.md`); the §5
final report is replaced by programme G's `reports/ARCHITECTURE_DECISION.md`.
