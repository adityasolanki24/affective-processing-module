# J6 report: does expression need to be learned? (exploratory, 2026-10-18)

**Governed by:** `prereg/J6_ADDENDUM.md` (committed at 82a65f9 before J6-X), DEVIATIONS D2 (exploratory, because
J5's gate failed) and D3 (proximity value).

**Data:** J6-X, 960 lifetimes (6 systems × 4 histories × 40, J5 seeds at fresh indices 40–79): **28,800 closed-loop
NERVA encounters**. 0 falls, 0 safety trips, 100% arrival in every system.

**Results:** `results/J6/J6_X.json` (utilities, comparisons) and `results/J6/J6_X_loop.json` (J5's six questions per
system). Being exploratory, nothing here adopts an architecture.

## Headline

1. **Learning from the robot's own consequences alone removes expression.**
   - With time and perturbation as the only costs, "stop at 0.6 m, walk fast, no hesitation" is optimal for every
     belief (D3).
   - LEARNED is fast (4.4 s per approach vs L0's 15.4 s) and gets few contacts. Its stopping distance does not depend
     on what it has learned (ρ = 0.04).
   - Experience-dependent expression only appears when the task also values coming close. **Expressive
     differentiation is a consequence of a task trade-off, not something the physical utility asks for by itself.**
2. **With that trade-off, a simple learned mapping beats L0 on every measure:**
   - utility, time, contacts, push size and expressive range;
   - while keeping evidence dependence.
3. **A recurrent mapping adds nothing material.** GRU_V reproduces LEARNED_V's choice in 97.6% of encounters.
4. **The avoidance trap gets worse, not better.** No mapping revises pushy → calm within the lifetime.

## Reading rules (fixed in the addendum)

| system | E1 own utility vs L0 | E2 safety: falls; contacts vs L0 | E4 evidence dependence: ρ; stopping-distance effect | "improves on L0"? |
|---|---|---|---|---|
| LEARNED (U) | −11.20 s [−11.35, −11.04] ✅ | 0; −0.20 [−0.23, −0.17] ✅ | **0.04**; 0.000 m ❌ | **no**: it abandons evidence dependence |
| LEARNED_V (U_V) | −11.91 [−12.44, −11.37] ✅ | 0; −0.040 [−0.069, −0.012] ✅ | 0.87; **0.258 m** [0.237, 0.277] ✅ | **yes** |
| RESIDUAL_V (U_V) | −1.71 [−2.21, −1.20] ✅ | 0; **+0.120** [+0.103, +0.138] ❌ | 0.83; 0.202 m ✅ | **no**: more contacts (L0's slow, hesitant approach is kept) |
| GRU_V (U_V) | −11.99 [−12.49, −11.47] ✅ | 0; −0.044 [−0.073, −0.016] ✅ | 0.87; 0.265 m ✅ | yes |

**Is a persistent recurrent state necessary?** The rule is that GRU_V must beat LEARNED_V on U_V.
- GRU_V − LEARNED_V = **−0.088 s [−0.185, −0.003]**: the rule is **nominally met**.
- This is 0.5% of U_V, with the CI touching zero, and no multiplicity correction.
- GRU_V was trained only to imitate LEARNED_V's choices, and agrees with them in 97.6% of encounters. It cannot carry
  information the explicit memory lacks.
- The rule had no practical-significance margin. That is a weakness of how it was written, and it is disclosed here
  rather than fixed after the fact.

**Reading: a persistent recurrent state is not necessary.** The explicit memory plus a 5-parameter stateless mapping
does the same job.

## The loop, per system (J5's questions on J6-X)

| system | Q1 history AUROC | Q2 stopping-distance effect | Q3 max push vs FROZEN (risky_B, late B1) | Q4 revision late_B / change_B | Q5 recovery |
|---|---|---|---|---|---|
| CL (L0) | 0.96 | 0.041 m | −0.009 [−0.014, −0.004] ✅ | 3 / never | not testable (< 0.05 m) |
| LEARNED | 0.91 | 0.000 m | −0.031 ✅ | never / never | not testable |
| **LEARNED_V** | 0.99 | **0.258 m** | **−0.027 [−0.031, −0.023]** ✅; contacts −0.96 | **2** / never (4 of 40) | **never** (8 of 40) |
| GRU_V | 0.99 | 0.265 m | −0.028 ✅ | 2 / never | 9 encounters (9 of 40) |
| RESIDUAL_V | 1.00 | 0.202 m | −0.016 ✅ | 2 / never | never |

**Is the full loop closed by a learned mapping?** The addendum's rule needs E3 plus J5's Q1, Q2 and Q4.
- LEARNED_V passes E3, Q1 and Q2, and calm → pushy revision.
- It **fails Q4** for pushy → calm (change_B).
- **So the full loop is still not demonstrated.**

What is demonstrated, with a learned mapping:
- experience → belief → expression → changed physical outcomes;
- in one direction, revision when an entity becomes risky.

**Replication note.** On these fresh seeds, CL's Q3 primary measure is significant (−0.009, upper bound −0.004). In
J5-X it was not (−0.003, n.s.). J5's confirmatory verdict stands; this is an exploratory replication suggesting L0's
effect on push size is real but small.

## Why pushy → calm never happens: the avoidance trap is physical

- Once B1 is believed pushy, LEARNED_V stops at 0.6 m.
- There, even a pushy entity's strong-evidence rate is 0.23 against a calm one's 0.11. Each calm meeting moves the
  belief very little, and λ = 0.02 per encounter forgets slowly.
- The more decisively a mapping avoids, the less it can learn that avoidance has become unnecessary.
- L0's compressed range (0.40–0.53 m) keeps the robot close enough to see the change, but too close to avoid much.

This is the exploration–exploitation problem on a body. A mapping that maximises utility under the current belief
does not value information. Programme I's value-of-information probing is the natural next step, now on NERVA.

## Complexity

| mapping | fitted parameters | supervision |
|---|---|---|
| L0 | 0 | designed |
| LEARNED / LEARNED_V | 5 (time model) + the price and value as explicit design values | the robot's own arrival times (J6-T) |
| RESIDUAL_V | 5 | as LEARNED_V |
| GRU_V | about 1,800 | imitation of LEARNED_V |

No human judgement and no simulator truth were used as targets. The "value of coming close" is a task parameter that
a designer, or eventually a person, must supply. It is not learnable from the robot's physical consequences.

## Uncertain

- All of it is exploratory, and the price and value were chosen by inspecting the mapping (D3).
- Whether people read LEARNED_V's far/near stopping as legible or appropriate is untested. H2's pilot suggested weak
  legibility for H's styles.
- LEARNED_V drops L0's hesitation and slow tempo entirely. Those are expressive channels H designed; here they only
  cost time.

## Is the next phase justified?

Phase 4 can proceed. J6 supports, as an exploratory finding, that:
- a small, explicit, decision-theoretic mapping with stated task values beats the fixed lookup;
- recurrent states add nothing.

Any adoption needs its own preregistered confirmation, and pushy → calm revision needs an information-seeking
mechanism.
