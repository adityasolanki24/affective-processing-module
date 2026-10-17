# APM v0.5 decision: autonomous revision (programme I, 2026-10-17)

## Decisions, by the preregistered rules (`MASTER_PREREGISTRATION.md` §7)

### 1. I1 passed: the memory changes

**APM v0.5's entity memory is the dense-evidence, change-aware type memory (R_dense).** It replaces F's cumulative
collision memory, which decision A had kept.

Evidence: 12,000 test lifetimes, replicated on 6,000 fresh ones.
- Revision after a change: median 2–3 episodes, against "never" for A0.
- Perfect discrimination in a stable world.
- Avoidable collisions −56% [−59, −52], all of it with careless people.
- Completion −0.3 pp.

### 2. P1–P3 passed: VOI probing is recommended where memory drives outright avoidance

**Recommendation:** wherever memory drives *outright avoidance*, add value-of-information probing (budget 2 per
episode).

**Qualifications:**
- **It narrowly fails efficiency non-inferiority against the plain keep-away robot.** The completion lower bound is
  −2.18 pp against the −2 pp margin. Probing costs about 0.8 pp where it isn't needed and gains 8.7 pp in the trap
  world.
- **Targeting adds little:** 0.7 pp over random probing at the same budget.
- **The stronger recommendation is not to build outright keep-away on memory at all.** Decision A's graded,
  consequence-predicting use of memory gives the same safety gain without the trap.

### 3. G2b's memory is withdrawn as a recommendation

Its beliefs are inverted (AUROC 0.21). Programme G recommended it as a memory upgrade pending confirmation; it should
not be adopted.

### 4. Do not use Thompson sampling with per-encounter draws for avoidance decisions

It freezes the robot (S10: completion 0.16).

## Updated architecture (decision A + I's memory)

```
 Obs (20 Hz) ─┬─► risk filter s_t (EMA of TTC threat; G's A0)                     ┐
              │                                                                    │
              └─► encounter tracker (per person track; robot-observable only)      │
                    collision? · min clearance while moving · observable frames    │
                         │ at encounter end, if informative and identity reliable  │
                         ▼                                                          │
                  TYPE BELIEF per identity b ∈ Δ^K (K = 3, learned)                 │
                    between encounters:  b ← π + (b − π)·e^(−λΔt)                   │
                    at an encounter:     b ← b ⊙ E[:, c]^w   (normalised)           │
                         │ p = b·r,  confidence = 1 − H(b)/log K                    │
                         ▼                                                          ▼
          consequence predictor f_θ(x, a) (G's A0, unchanged) ──► a* = argmax progress − λc·m_t·P_col − λs·P_slip
                                                                        │
                                       independent safety supervisor (unchanged) ──► motion
```

**Learned offline from the robot's own unlabelled experience** (EM, K by BIC):
- π: type prior;
- E: outcome-category probabilities per type. Categories: collision, or minimum clearance in
  [0, .05, .10, .15, .20, .30, .50, ∞) m;
- r: collision rate per encounter for each type.

**Tuned:** λ = 7.2 × 10⁻⁴ s⁻¹ (time constant ≈ 23 min of experience), tuned self-supervised on predictive
log-likelihood.

**Memory cost:** K + 1 = 4 floats per known identity. Compute is negligible next to the predictor.

**Per embodiment:**
- The outcome coding needs a self-observable graded signal of how close an encounter came to a bad outcome. Clearance
  is the one here.
- Types and λ are re-learned from that robot's own experience; no labels are needed.

## What remains open

1. **Second embodiment.** On NERVA the analogue of clearance is graded perturbation, such as peak base speed (H's
   jolts). The same method should apply unchanged; that is untested.
2. **Robustness.** G6's shift conditions, identity errors beyond S4, and types that are not cleanly separable. The
   testbed's scripted pedestrians make clearance unusually informative.
3. **Online type learning.** Types are fitted offline from a stable world. A robot that refits as it goes, or adds
   types, is the natural next step. Deployment-time learning still needs G7's powered promotion gate.
4. **Integration into the core.** Make this memory, and decision A's rule, the default runtime in `src/apm`, replacing
   the PAD path that F and G falsified.
5. **Human legibility of expression** (H2): unchanged; needs an ethics determination.

## Cost

About US$2 of the US$40 cloud cap, on 128-core spot VMs, roughly 18× faster than the laptop. All resources were deleted
at the end of the session.
