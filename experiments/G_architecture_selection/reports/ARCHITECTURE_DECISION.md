# APM v0.3 architecture decision (programme G, 2026-10-12)

**Decision: A — simple predictive regulator.** A learned action-conditioned consequence predictor (one small MLP),
explicit outcome-grounded entity memory, and a fairly calibrated **static** collision/slip penalty in a transparent
argmax rule. No PAD, no persistent affective state, no recurrent regulator, no learned utility corrections, no
continual weight updates in deployment.

Every claim below comes from a preregistered, locked, once-run confirmatory test unless it is marked *secondary* or
*exploratory*. Reports: `G0_AUDIT.md`, `G1_REPORT.md`, `G2_REPORT.md` (incl. G2b), `G3_REPORT.md`, `G45_REPORT.md`,
`G6_REPORT.md`, `G7_REPORT.md`, `G8_REPORT.md`; programme F's E1 in `../../F_behavioral_learning/results/CONFIRMATORY_SUMMARY.md`.

## Answers to the ten questions

1. **Is PAD necessary for useful robot behaviour? No.** Under equal tuning budgets, APM v0.1's PAD-modulated rule
   (R6) has 25.7% *more* collisions than a static penalty (R1) [CI +16.7, +35.4] (G1, primary P1 fails), and again
   +25.1% vs A0 in G4. Programme F's E2 gain (−25.4%) is reproduced entirely by giving B1 the same tuning budget
   (R1 vs F's B1: −24.6% [−30.2, −18.8]). PAD adds no predictive information (E1: no meaningful information on NERVA
   and wheeled stability; inconclusive, −1.6%, for wheeled collision; G3: PAD inputs −0.4% Brier, +7.9% closed-loop
   collisions). Replacing PAD by the instantaneous appraisal threat or a plain filter removes the harm (G1, R7/R8:
   −26% / −24% vs R6).
2. **Can a simpler persistent risk regulator reproduce its benefits? Yes — and persistence itself is not needed.**
   A simple EMA regulator is non-inferior to PAD with 26% fewer collisions (G1 P2 holds), but its state carries no
   detectable value (shuffled/shifted traces within ±2%), no persistent family beats the fairly tuned static penalty
   by the practical margin (G1 P3: −7.1%, short of −10%), and removing the regulator from A0 is non-inferior (G8:
   +1.6%, one-sided upper +6.1%). The benefit is a *calibrated* risk weight, not a persistent state.
3. **Which memory system adapts fastest without losing reliability?** None met the preregistered adaptation
   criterion (≤ 6 episodes after a disposition change), because avoidance removes the evidence (≈ 69% of encounters
   carry none). The fastest revision came from **exposure-gated evidence + evidence-free time decay (τ ≈ 6 min) +
   budgeted uncertainty-guided probing** (G2b, secondary: risky→safe median 6 episodes [2, 13], restricted mean 7.1
   vs 10.7 for F's memory; safe→risky 8; collisions −10% vs F's memory, no false blame). Exposure gating alone cuts
   collisions 10–14% (G2, secondary). In the final equal-retuning comparison this assembled memory was *not* better
   than F's cumulative memory (G8: +22.8% collisions after retuning chose a less cautious penalty), so it is not
   adopted as default; it is the recommended direction for memory work.
4. **Does explicit long-term memory contribute beyond recurrent state? Yes in the transparent architecture, no in a
   GRU.** Removing entity memory from A0 costs +28.9% collisions [+18.8, +40.7] (G8). Recurrent state does not
   substitute: no stateful consequence predictor beats the stateless one (G3: EMA +9.6%, GRU +18%, LDS +21%,
   fast+slow −2.2% n.s.; none adds predictive information). Inside an end-to-end GRU policy, explicit memory inputs add
   nothing (A3 vs A2: +2.6% [−4.5, +10.5]).
5. **Is a specialised neural regulator better than an ordinary GRU? No.** The learned residual regulator (A4, F's B3
   network anchored on A0) and the ordinary GRU are indistinguishable (A4 vs A2: −1.2% collisions [−9.2, +7.2],
   +0.6 pp completion [−1.1, +2.2]); neither beats A0 under the preregistered rule (G4).
6. **Is model-based planning better than direct learned action selection? Under the safety-first criterion, yes.**
   A0 is one-step model-based selection with a learned consequence model. Direct learned policies complete ≈ 7 pp more
   goals at about the same collisions per metre but 5–9% more collisions per minute (CIs up to +13–17%), so none
   beats A0 (G4). Multi-step search (A6) was not justified by any G1–G3 result and was not tested.
7. **Which training method offers the strongest sample efficiency?** Imitation of the A0 teacher (BC, DAgger)
   reproduces A0 within ±3% using 0.3M interactions (8× less than RL) but cannot exceed it; **BC → PPO** gives the best
   learned policies (highest validation J and learning-curve area; 17% fewer collisions than PPO from scratch at
   equal 2.4M interactions, G5 primary). RL policies converge to GO/YIELD-only behaviour, which is the objective's
   optimum in this testbed, not a pathology.
8. **Which mechanism allows actual continual adaptation?** Within a lifetime: the explicit memory (dynamic by design;
   faster with time decay and probing). Across environments: retraining the predictor *with replay* learned real
   improvements (G7: −30% collisions, +16% J in a new pedestrian regime) but increased collisions by 7% in another and
   drifted on the original task; a 20-lifetime promotion gate blocked every update (safe, no learning) and correctly
   rejected corrupted updates; retraining without replay collapsed. **No validated continual weight learning**
   exists yet; shipping one needs a statistically powered gate.
9. **Which parts generalise across embodiments?** The software module (rule + memory + AffectLink interface) is
   portable. Only `stability_loss` has shared semantics between the wheeled robot and NERVA; affective variables add
   no predictive information on either (E1), and programme F's Milestone D found no data-efficiency gain from learned
   transfer. The learned predictor is robot-specific and is trained per embodiment; no social transfer to NERVA is
   claimed (no people or identities in its telemetry).
10. **Smallest architecture worth implementing on embedded hardware:** see "Recommended architecture" (≈ 7.3k
    parameters, one MLP forward per action per 0.5 s decision, two floats per known identity).

## Ranked table of surviving architectures (nominal confirmatory tests)

(filled from G4, G6 and G8; see the section below)

## Recommended architecture for APM v0.3

(filled after G8 stage 2)
