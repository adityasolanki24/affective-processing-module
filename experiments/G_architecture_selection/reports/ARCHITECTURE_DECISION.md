# APM v0.3 architecture decision (programme G, 2026-10-12)

## Decision: **A — simple predictive regulator**

A learned action-conditioned consequence predictor (small MLP ensemble), explicit outcome-grounded entity memory,
and a calibrated collision/slip penalty with a short risk filter, combined by a transparent argmax rule. **No PAD, no
appraisal-driven affect, no recurrent regulator network, no learned utility corrections, no deployment-time weight
updates.** B (standard learned recurrent policy), C (specialised affective regulator) and D (modular with a learned
utility adjustment) were tested and did not justify themselves; E is not needed because the evidence is consistent
across experiments.

Evidence base: preregistered, locked, once-run confirmatory tests (G1–G8), and the **G9 validity re-test**, which
removes unavoidable spawn-overlap collisions (testbed defect D8; `G9_REPORT.md`) and governs wherever it differs
from the original test. Reports: `G0_AUDIT.md`, `G1_REPORT.md`, `G2_REPORT.md` (incl. G2b), `G3_REPORT.md`,
`G45_REPORT.md`, `G6_REPORT.md`, `G7_REPORT.md`, `G8_REPORT.md`, `G9_REPORT.md`; programme F's E1 in
`../../F_behavioral_learning/results/CONFIRMATORY_SUMMARY.md`. "Avoidable" = collisions after the first step.

## Answers to the ten questions

1. **Is PAD necessary for useful robot behaviour? No — it is harmful.** With equal tuning budgets, APM v0.1's
   PAD-modulated rule has **+98.8% avoidable collisions** [+55.7, +156.5] vs a static penalty (G9-G1 P1) and +278% vs
   A0 (G9-G4); 15× the compute (9.3 ms vs 0.6 ms per decision). Programme F's E2 advantage of B2 over B1 is fully
   explained by B2's 9× larger tuning budget: re-tuning B1 alone gives −58.5% avoidable collisions (G9; −24.6% total
   in G1). PAD carries no predictive information (E1: no meaningful information for NERVA and wheeled stability,
   inconclusive −1.6% for wheeled collision; G3: PAD inputs +97% avoidable closed-loop collisions). Replacing PAD by
   the instantaneous appraisal threat or a plain filter removes the harm (R7/R8: −62% / −65% vs PAD).
2. **Can a simpler persistent risk regulator reproduce its benefits? Yes, and more.** A 4-parameter EMA risk filter
   (R2: threat smoothed with τ = 0.62 s, gain k = 0.45, collision impulse c = 0.017, λc 3.95, λs 0.99) has
   **−73% avoidable collisions vs PAD** (G9 P2) and **−47% [−62, −27] vs an equally tuned static penalty** (G9 P3,
   p_Holm 0.0004). Removing it from A0 is not non-inferior (+11% [−19, +51]). The configuration earns its place; whether
   the gain comes from the filter or from its calibrated λ is not separable with these data. Other persistent
   regulators (hysteresis, Bayesian hazard, learned scalar) were not better than the static penalty.
3. **Which memory system adapts fastest without losing reliability?** None met the preregistered criterion (revision
   within 6 episodes), because avoidance removes the evidence (≈ 69% of person encounters carry none). Fastest:
   **exposure-gated evidence + evidence-free time decay (τ ≈ 6 min) + budgeted uncertainty-guided probing** (G2b:
   risky→safe median 6 [2, 13], restricted mean 7.1 vs 10.7 episodes for F's memory; safe→risky 8; −10% collisions;
   no false blame). With A0's calibrated penalty it was never worse than A0 under 13 held-out shifts and significantly
   better under faster pedestrians (−23%), sluggish actuation (−16%) and identity uncertainty (−60%) (G9-G6), but its
   nominal effect is inconclusive and, when the penalty was re-tuned on J (which lowered it), it was worse (+144%,
   G8). Recommended as the memory upgrade **with the penalty held at A0's calibrated value**; not yet confirmed as the
   default.
4. **Does explicit long-term memory contribute beyond recurrent state? Yes in the transparent architecture.** Removing
   entity memory from A0 costs **+224% avoidable collisions** [+141, +352] (G9-G8). No recurrent predictor state
   substitutes for it (G3: EMA, LDS, GRU states worse than stateless). Inside an end-to-end GRU policy, explicit memory
   inputs add nothing measurable (A3 vs A2).
5. **Is a specialised neural regulator better than an ordinary GRU? No.** The learned residual regulator (A4, fast +
   slow latent anchored on A0) and the ordinary GRU are indistinguishable (A4 vs A2: −22% avoidable [−48, +16],
   +0.5 pp completion); neither beats A0.
6. **Is model-based planning better than direct learned action selection? For safety, yes.** The direct learned policies
   have 44–87% *more* avoidable collisions than A0 (one-step selection with a learned consequence model), in exchange
   for ≈ 7 pp more completed goals (G9-G4), and degrade more under shift (G6). Multi-step search was not justified
   by G1–G3 and was not tested.
7. **Which training method offers the strongest sample efficiency?** Imitation of the A0 teacher (BC, DAgger)
   reproduces A0 with 0.3M interactions (8× less than RL) but cannot exceed it; **BC → PPO** produces the best learned
   policies (−45% avoidable collisions vs PPO from scratch at the same 2.4M interactions; highest learning-curve area).
   RL converges to GO/YIELD-only policies, which is the objective's optimum in this testbed, not a pathology.
8. **Which mechanism allows actual continual adaptation?** Within a lifetime, the explicit memory (faster with time
   decay and probing). Across environments, retraining the predictor *with replay* learned real gains (−30%
   collisions, +16% J in a new regime) but regressed safety elsewhere (+7%); the 20-lifetime promotion gate blocked
   every update (safe, no learning) and rejected corrupted updates; no-replay retraining collapsed (G7). **No validated
   deployment-time weight learning yet** — a statistically powered gate is required first.
9. **Which parts generalise across embodiments?** The software architecture (rule + memory + AffectLink interface)
   and the regeneration pipeline (datasets reproduced bit-for-bit on new machines for both robots). Only
   `stability_loss` has shared semantics; affective variables add no information on either robot (E1); learned
   transfer did not improve data efficiency (F's Milestone D). The predictor is trained per embodiment; no social
   transfer to NERVA is claimed (its telemetry has no people or identities).
10. **Smallest architecture worth implementing on embedded hardware:** A0 as specified below — ≈ 36.5k parameters
    (5 × 7.3k), ≈ 0.6 ms per 0.5 s decision on a laptop CPU, 2 floats per known identity plus a type pool. Smaller
    variants were tested and rejected: one ensemble member (+42% avoidable, CI lower bound 0), no regulator
    (inconclusive, +11% [−19, +51]), both together (+164%), a hand-written consequence model (+1382%).

## Ranked table of evaluated architectures (nominal, G9 re-test lifetimes)

| rank | architecture | params | avoidable coll./min | severe/min | completion | latency µs/decision | robustness (G6/G9) | confidence | main failure mode | complexity justified? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **A0**: predictor ×5 + entity memory + EMA risk penalty (R2) | 36.5k | **0.011** | 0.0015–0.0020 | 0.601–0.619 | 630 | reference | primary + G9 | slow revision of individual risk after a change (G2) | **yes** (every removal worse) |
| 1b | A0 + revisable memory (exposure gating, time decay, probing), A0's penalty | 36.5k + 2/identity | ≈ A0 (nominal inconclusive) | — | −1 pp | 630 | best: never worse, −16% to −60% under 3 shifts | G6 (secondary per condition) | small completion cost; penalty must stay calibrated | promising; confirm before default |
| 3 | R1: predictor ×5 + memory + static penalty | 36.5k | 0.020 | 0.0048 | 0.618 | 560 | worse than A0 in most shifts | G1/G9 primary | less caution under hazard shifts | — (dominated by A0) |
| 4 | A4: learned residual (fast+slow GRU) on A0 | 14.8k + A0 | 0.017 | 0.0020 | **0.681** | 1,300 | not robust (slow actuation +72%) | G4/G9 primary | more exposure → more collisions; shift fragility | no (≈ ordinary GRU; not > A0) |
| 5 | A2/A3: ordinary GRU actor-critic (BC→PPO) | 15.2k | 0.021 | 0.0033–0.0050 | 0.674–0.676 | 1,200 | not robust | G4/G9 primary | as A4 | no |
| 6 | G3 fast+slow consequence predictor + rule | 19.2k ×5 | 0.006 | 0.0008 | 0.605 | 6,200 | not tested | secondary only | −1 pp completion; 10× compute | open question (secondary −38%) |
| 7 | A1 / R6: APM v0.1 PAD regulator | 36.5k + APM core | 0.040–0.044 | 0.0083–0.0085 | 0.606–0.616 | 8,200–9,300 | least robust (up to 7× A0) | G1/G4/G9 primary | PAD over-weighting oscillates and mis-times caution | **no** |

## Recommended architecture for APM v0.3

```
 sensors / tracker ──► Obs (20 Hz) ──┬──────────────────────────────────────────────┐
                                     │                                              │
                     ┌───────────────▼──────────────┐        ┌──────────────────────▼─────────────┐
                     │ Entity memory (per identity)  │        │ Risk filter  s_t (EMA of TTC threat) │
                     │ Beta counts of observable     │        │ m_t = 1 + k·s_t                       │
                     │ trials, type-pool prior       │        └──────────────────────┬─────────────┘
                     └───────────────┬──────────────┘                               │
                                     │ p(collision | identity), confidence          │
              ┌──────────────────────▼───────────────────────┐                     │
              │ Decision features x_t (41)                    │                     │
              └──────────────────────┬───────────────────────┘                     │
                                     │ every 0.5 s                                   │
              ┌──────────────────────▼───────────────────────┐                     │
              │ Consequence predictor f_θ(x_t, a), a ∈ 4      │  (MLP 45-64-64-3, ×5)│
              │ → P_col(a), P_slip(a), progress(a)            │                     │
              └──────────────────────┬───────────────────────┘                     │
                                     ▼                                              ▼
                 a* = argmax_a  progress(a) − λc·m_t·P_col(a) − λs·P_slip(a)     (transparent rule)
                                     │
                                     ▼
              independent safety supervisor (unchanged, never depends on APM) ──► motion
```

**Update equations.**
- Threat (each observation, 20 Hz): x_t = 1 − Π_i (1 − TTC-estimate_i).
- Risk filter: s ← s + (1 − e^(−Δt/τ))(x_t − s); on a collision s ← s + c(1 − s); m_t = 1 + k·s.
  Selected values: τ = 0.62 s, k = 0.45, c = 0.017; λc = 3.95, λs = 0.99.
- Entity memory (per identity; at the end of each encounter): evidence only if a collision was physically possible
  (person within 0.5 m while moving) or occurred. Counts n, k (weighted by identity confidence) decay continuously,
  n, k ← e^(−Δt/τ_m)·(n, k), τ_m ≈ 360 s (recommended upgrade; F's cumulative counts are the confirmed default);
  p = (k + 2·p₀)/(n + 2), confidence n/(n + 2), p₀ = type-pool mean.
- Optional probing (recommended upgrade): if the rule would YIELD/DETOUR for the nearest person, that person is
  isolated (nobody else within 2 m), reliably identified and memory confidence < 0.5, choose SLOW instead
  (≤ 2 probes per episode). The safety supervisor is untouched.

**Learning objectives.**
- Predictor: supervised, per robot; inputs x_t and the action actually taken; targets = what actually happened within
  2 s (collision, slip: binary cross-entropy; goal progress: squared error); exploration data with randomised safe
  action variation; early stopping on held-out lifetimes; 5 seeds. No labels for actions not taken.
- Memory: closed-form Bayesian counting (no gradient learning).
- Penalty calibration (λc, λs, filter): tuned once on validation lifetimes with an equal search budget; recommended
  change: calibrate to an explicit **safety target** (e.g. maximise completion subject to an avoidable-collision
  budget, a Lagrangian/constrained formulation), because tuning on J alone lowers caution (G8).
- No deployment-time weight updates until a promotion gate with a powered statistical test exists (G7).

**Data flow and deployment constraints.** Observations at 20 Hz update the filter and memory; decisions every 0.5 s;
one ensemble forward per action (4 × 5 MLP evaluations ≈ 0.6 ms on a laptop CPU, NumPy only, no GPU); memory state
2 floats per known identity (bounded record table); the advisory output (action preference) never bypasses the robot's
own safety supervisor; on any APM exception the robot falls back to its ordinary controller within one decision period
(F's E7, still valid). Per embodiment: retrain the predictor; keep the rule, memory and interface.

## Limitations and next steps

- **Testbed:** defect D8 (spawn overlaps) diluted all original collision-rate comparisons; G9 corrects the primary
  families on fresh lifetimes, but secondary analyses of G1–G8 remain on the diluted metric. The next testbed version
  should place the robot with person clearance. Arena size could not be varied. The pedestrian model makes stopping
  dominate slowing, which shapes every learned policy.
- **Power:** A0 is so safe nominally (≈ 0.011 avoidable collisions per minute) that non-inferiority on avoidable
  collisions needs far larger N than this programme used; G6 eligibility was underpowered under the corrected metric.
- **Open questions worth a dedicated test:** (i) the revisable memory with a fixed calibrated penalty as the default;
  (ii) a slow, lifetime-persistent context state in the consequence predictor (G3 secondary −38% avoidable);
  (iii) a safety-constrained learned efficiency layer (the learned policies' +7 pp completion at per-distance safety
  parity); (iv) a powered promotion gate for predictor updates with replay.
- **Novelty:** none claimed. A0 is standard one-step model-predictive selection with a learned consequence model,
  Beta-Bernoulli identity memory and an adaptive risk weight (`LITERATURE.md` §1, §2, §5, §10).
