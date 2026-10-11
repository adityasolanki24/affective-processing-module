# G9 — Validity re-test on avoidable collisions (2026-10-12)

**Why:** testbed defect D8 (`DEVIATIONS.md`): the robot can spawn overlapping a pedestrian; that contact is reported as
a collision at the first step, identically for every policy. In the rule-based systems it was 67–97% of all reported
collisions (programme F's E2: 58–78%). It dilutes every collision-rate comparison and makes non-inferiority
conclusions anti-conservative. Preregistered in `prereg/G9_ADDENDUM.md`: the same frozen systems, selections and
checkpoints; fresh lifetimes (indices 10000+); the original primary families with **avoidable collisions** (after the
first step). All runs: commit `58e1244`, locks passed, all 20 neural checkpoint hashes verified
(`results/G9/g9_avoidable.json`, `results/G9/g9_total.json`). **Where G9 and the original tests disagree, G9 governs.**

## Primary families, avoidable collisions (total-collision result on the same lifetimes in brackets)

| test | avoidable collisions [95% CI] | completion | decision (original test) |
|---|---|---|---|
| **G1 P1** PAD (R6) beats static (R1) | **+98.8% [+55.7, +156.5]** (total +19.9%) | −0.2 pp | fails (fails) |
| **G1 P2** EMA (R2) non-inferior to PAD | −73.3% [−80.6, −64.7] (−24.6%) | +0.3 pp | holds (holds) |
| **G1 P3** EMA (R2) beats static (R1) | **−46.9% [−62.2, −27.2]**, p_Holm 0.0004 (−9.6%) | +0.1 pp | **holds** (failed: −7.1%) |
| **G3 Q1** EMA-state predictor beats stateless | +94.6% [+32.6, +190.6] | −0.6 pp | fails (fails) |
| **G3 Q2** fast+slow beats EMA-state | −68.1% [−80.6, −50.9] | −0.4 pp | holds (holds) |
| **G4** A2 GRU (BC→PPO) beats A0 | +83.5% [+17.1, +224.0] | +6.7 pp | fails (fails) |
| **G4** A3 GRU+memory (BC→PPO) beats A0 | +87.0% [+12.8, +232.2] | +6.6 pp | fails (fails) |
| **G4** A4 residual (PPO) beats A0 | +43.5% [−8.4, +140.0] | +7.3 pp | fails (fails) |
| **G5** A3 BC→PPO beats A3 PPO | −44.9% [−64.3, −20.8] | +0.5 pp | holds (holds) |
| **G5** best GRU non-inferior to A0 | +87.0% | +6.6 pp | fails (fails) |
| **G8** remove persistent regulator (static) | +11.1% [−18.9, +51.4] | −0.5 pp | **not removable** (removable) |
| **G8** remove ensemble (1 member) | +42.2% [0.0, +102.8] | −0.2 pp | **not removable** (removable) |
| **G8** remove entity memory | +224% [+141, +352] | −1.0 pp | needed (needed) |
| **G8** hand-written predictor | +1382% | −5.3 pp | needed (needed) |
| **G8** G2b memory, λ retuned to 1.21 | +144% [+83, +237] | +0.1 pp | not added (not added) |
| **G8 stage 2** A0_min (static + 1 member) | +164% [+85, +307] | −0.2 pp | not non-inferior (not) |

Secondary (avoidable): R1 vs F's B1 −58.5% (equal tuning); R7 / R8 (PAD replaced by appraisal / filter) vs R6 −62% /
−65% and vs R1 −25% / −30%; learned scalar R5 vs R1 +40%; stateful predictors vs stateless: LDS +211%, GRU +157%,
PAD inputs +97%, fast+slow **−37.8% [−61.8, −3.7]** (−1.0 pp completion); A0 (F's predictor) vs G3's retrained
stateless predictor +48.6% (the retrained predictor is better); APM v0.1 (A1) vs A0 +278%; A4 vs A2 −21.8%
[−47.5, +15.6].

## G6 under the corrected metric

A0 is very safe: 5 avoidable collisions in 150 nominal lifetimes (750 robot-minutes). With so few events, avoidable
non-inferiority cannot be *established* in most held-out conditions (CIs of several hundred percent), so **no system
is eligible** under the corrected metric — the eligibility test is underpowered, not failed by a large effect.
Point estimates: A0_mem is better than A0 in 11 of 13 held-out conditions and never worse; significantly so where
events are frequent: fast pedestrians −23.1% [−34.7, −9.6], slow actuation −15.7% [−26.5, −3.6], identity uncertainty
−60.0% [−85.7, −21.4]. R1 (+27% to +350%), APM v0.1 (+125% to +625%) and the neural cells (A4 up to +155%; slow
actuation +72%, estimator ×0.5 +100%, sensor noise +155%) are worse in most conditions.

## What changes

1. **Persistent risk regulation does help** once the dilution is removed: the EMA-filtered risk weight (R2) has 47%
   fewer avoidable collisions than an equally tuned static penalty, and removing it from A0 is not non-inferior. G1's
   original "persistence is not needed" conclusion is withdrawn. (The two differ in calibrated λ as well as in the
   filter, so this establishes the regulated configuration, not the filter in isolation.)
2. **The 5-member ensemble is needed** (removing it: +42%, CI lower bound 0); G8's original "removable" is withdrawn.
3. **PAD is clearly harmful** (+99% avoidable vs a fairly tuned static penalty; +278% vs A0).
4. **Neural policies are less safe than A0** (+44% to +87% avoidable collisions) while completing ≈ 7 pp more goals.
5. A **slow, lifetime-persistent context state in the predictor** (G3 fast+slow) reduced avoidable collisions by 38%
   (secondary; Q1's validation-chosen EMA state was worse). This is the one stateful mechanism worth a follow-up.
