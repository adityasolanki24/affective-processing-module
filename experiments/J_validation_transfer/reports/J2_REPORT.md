# J2 report: stress tests of the near-miss memory (2026-10-18)

**Governed by:**
- `MASTER_PREREGISTRATION.md` §3 and `prereg/J2_ADDENDUM.md`;
- deviation D1 (an analysis bug, corrected; `DEVIATIONS.md`).

**Data:**
- J2-X: 65,000 closed-loop lifetimes (5 systems × 13 conditions × 5 families × 200);
- 9,600 policy-group lifetimes.

**Results:** `results/J2/J2_X.json`.

## Verdict: **not promoted as a universal default.** A *conditional* improvement.

| | promoted? | failing conditions |
|---|---|---|
| EU-I (programme I's memory) | no | overlap_25, overlap_40, sensor_noise_0.20 |
| EC (selected on V; context-conditioned, fitted on mixed-policy J2-T) | no | overlap_25, overlap_40 |

### EU-I per condition (vs A0 in the same condition)

| condition | S3 AUROC | careful misclassified | avoidable collisions vs A0 [95% CI] | completion | decisive adaptation S1 / S2 |
|---|---|---|---|---|---|
| nominal | 1.00 | 0.01 | −31% [−39, −22] | +0.02 pp | 3 / 3 |
| overlap_25 | 0.87 | **0.29** | −35% | −0.41 pp | 4 / 4 |
| overlap_40 | **0.67** | **0.46** | −36% | −0.59 pp | 5 / 4 |
| erratic (novel pattern) | 1.00 | 0.01 | −25% | −0.35 pp | 3 / 3 |
| drift (gradual change) | 1.00 | 0.01 | −35% | +0.02 pp | 6 / 3 |
| identity_uncertain | 0.87 | 0.13 | −42% | −0.26 pp | 3 / 5 |
| sensor_noise_0.10 | 0.98 | 0.07 | −33% | −0.38 pp | 3 / 3 |
| sensor_noise_0.20 | 0.91 | **0.29** | −28% | −0.73 pp | 4 / 2 |
| crowd_dense | 1.00 | 0.00 | −24% | −0.36 pp | 2 / 4 |
| obstacles_dense | 1.00 | 0.01 | −24% | +0.09 pp | 3 / 4 |
| fast_pedestrians | 1.00 | 0.00 | **+2% [−0, +5]** | −0.26 pp | 2 / 3 |
| speed_limit_0.6 | 1.00 | 0.00 | −26% | −0.17 pp | 3 / 4 |
| robot_fast | 1.00 | 0.01 | −14% | −0.10 pp | 3 / 3 |

A0's own S3 AUROC is 0.60–1.00 across conditions (0.80 nominal).

## Answers

1. **Does perfect discrimination survive realistic variation? No.**
   - It holds under novel patterns, gradual drift, crowding, obstacles, robot speed changes and moderate sensor noise.
   - It degrades with **overlapping behaviour**: when a person acts like the other type in 25–40% of approaches,
     AUROC falls to 0.87 / 0.67, and 29–46% of careful people are misclassified.
   - It also degrades with **heavy clearance noise** (σ = 0.2 m): 29% misclassified.

   This is a real limit of individual-level belief from noisy, overlapping evidence. A0 degrades too (to 0.70 / 0.60).
2. **The closed-loop safety benefit is robust:** −14% to −42% avoidable collisions vs A0 in 12 of 13 conditions.
   - It vanishes with fast pedestrians (+2%), where A0 itself discriminates perfectly (AUROC 1.00).
   - Completion costs are ≤ 0.73 pp in every condition (non-inferior).
3. **Is clearance informative about the individual after controlling for the robot's action and situation? Yes.**
   - Plug-in I(Z; C), in nats per encounter, under four driving policies:

     | policy | unconditional | given action band | given context |
     |---|---|---|---|
     | A0 | 0.40 | 0.41 | 0.41 |
     | always-GO (B0) | 0.45 | 0.40 | 0.45 |
     | cautious | 0.39 | 0.41 | 0.40 |
     | R1 | 0.41 | 0.42 | 0.42 |

   - Under B0, conditioning on the robot's speed removes about 0.05 nats: the part of clearance the robot itself
     causes.
   - **No false attribution from self-caused near-misses:** every memory observing the always-GO robot still ranks
     people perfectly (AUROC 1.00, 0% misclassified).
4. **Action- and context-conditioned evidence did not help.**
   - Fitted on mixed-policy experience, EM learned a policy-artefact "person type" (collision rate 0.60) from
     the always-GO driver.
   - All three J2-T models (EU-T, EA, EC) give weaker closed-loop safety than EU-I, which was fitted on the robot's
     own A0 experience. EC nominal: −3%, not significant.
   - **Lesson: a memory's evidence model should be fitted on the robot's own policy's experience.** Pooling other
     policies' data contaminates the types.

## Consequence

The near-miss memory remains a strong improvement over A0 in most conditions, but it is **not** a universal default.

Before default use, a deployment needs:
- a check that people's behaviour is separable enough (low overlap);
- clearance sensing noise well below 0.2 m.

A runtime safeguard follows directly: report confidence honestly and fall back to population-level caution when the
evidence model's predictive likelihood is poor. This is not tested here.
