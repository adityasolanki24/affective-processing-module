# J5 report: closed-loop expressive development on NERVA (2026-10-18)

**Governed by:** `prereg/J5_ADDENDUM.md`.

**Data:** J5-X, 960 lifetimes (6 systems × 4 histories × 40), **28,800 physically simulated closed-loop NERVA
encounters**.

**Results:** `results/J5/J5_X.json`.

## Verdict: **the full loop is not demonstrated with H's fixed L0 mapping.**

| question | CL (type memory + L0) | criterion |
|---|---|---|
| Q1 outcomes → beliefs | history AUROC **0.99**; belief effect +0.19 | ✅ |
| Q2 beliefs → expression | ρ(belief, stopping distance) **0.87**; stopping-distance gap risky − safe **0.046 m** | ✅ (small) |
| Q3 expression → outcomes | max push CL − FROZEN −0.003 [−0.009, +0.002]; contacts **+0.23** [+0.02, +0.44] | ❌ |
| Q4 revision | calm → pushy: median 4 B1 encounters ✅; **pushy → calm: never** within 8 | ❌ |
| Q5 recovery from excess caution | gap < 0.05 m | not testable |
| Q6 safety and task | CL: 0 falls, 0 trips, 100% arrival; **+8.1 s per approach** vs neutral | ✅ |

## What the loop does and does not do

1. **Beliefs track measured experience in the closed loop.** The CL robot learns from its own IMU, via the
   efference-gated, sensing-paused evidence, whether an entity pushes. History AUROC 0.99.
2. **Expression follows belief, but L0 compresses it.** Learned-type beliefs span about 0.25–0.5: even calm entities
   trigger the noisy detector at times. L0 was designed for H's 0.05–0.85 scale, so stopping distance moves by only
   0.05 m.
3. **On this body, cautious-looking expression is counterproductive.** L0's caution slows the approach and adds
   hesitation, which lengthens exposure within the entity's reach. CL robots get *more* contacts than the neutral
   frozen robot, while the small extra distance barely reduces push size.
4. **The avoidance trap appears physically.**
   - Once an entity is believed pushy, the robot stands farther away, where even a pushy entity rarely reaches. "No
     contact" there is weak evidence.
   - With λ = 0.02 per encounter, it never revises pushy → calm within the lifetime.
   - NOFEEDBACK and FROZEN behave the same way; RANDHIST "revises" only because its beliefs are not entity-specific.
5. **Self-motion must be separated from perturbation.** Pilots showed:
   - Without an efference-copy gate, the robot reads its own expressive head and body motion as being pushed.
   - Without sensing pauses, a continuously expressive robot cannot sense at all.

   These are design requirements for any embodied expressive learner.

## Ablations

| system | belief effect | history AUROC | stopping-distance gap |
|---|---|---|---|
| NOMEM | 0 | 0.50 | 0 |
| RANDHIST | 0.10 | 0.87 | 0.025 m |
| NOFEEDBACK | 0.14 | 0.92 | 0.037 m |
| H_MEM (H's memory on the robot's detector) | 0.30 | 0.92 | 0.078 m |

H_MEM's wider p-scale gives the most expression differentiation, but it also never revises pushy → calm.

## Safety finding

- Two falls occurred in RANDHIST (2 of 4,800 encounters), both with a pushy entity landing **3–5 separate pushes** of
  about 0.07 m/s in one encounter.
- Each push was inside H's validated single-push envelope (≤ 0.10 m/s). Repeated pushes combined into a fall, and the
  safety monitor tripped too late.
- H1b's envelope covers single pushes only. **Repeated sub-envelope perturbations are an uncertified risk for
  NERVA's controller.**
- CL itself had no falls.
