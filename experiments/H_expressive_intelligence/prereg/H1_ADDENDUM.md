# H1 addendum: NERVA's expressive movement repertoire (fixed before the H1 test data, 2026-10-12)

## Scope

**Question.** Does the H0b interface produce a repertoire of distinct, reliably controllable movement qualities
that:
- are recognisable in task geometries never seen in training;
- are recognisable at matched speed and under perturbations;
- keep stability and task completion?

**Interface.** NERVA's validated neutral controller with the admitted channels:
- `results/H0/X/admitted_interface.json`;
- the H0b adapter at commit `24aa42a` and later;
- the secondary E1′ configuration (rule S) in one descriptive arm.

**Runner and analysis:** `scripts/h1_repertoire.py` and `scripts/h1_analyze.py` at the commit that adds this file.

**This is claim A only.** Whether people perceive the styles is H2.

## Inputs fixed before the test

**Styles.**
- X0 = neutral.
- X1 = the six designed styles in `apm_express.styles.DESIGNED`: cautious, assertive, curious, hesitant, relaxed,
  energetic.
- X2 = independent uniform random intents.

**X3 (learned motion-style latent): not run.**
- No reference-motion data exist for this robot beyond Placo gait references.
- A latent learned from our own parametric generator cannot span more than the generator itself.
- Unsupervised disentanglement is not identifiable without supervision (Locatello et al. 2019).
- Instead, an exploratory PCA of X2's non-speed features reports how many dimensions of variation the interface
  actually produces.

**Speed matching (calibrated on partition V, `results/H1/V/tempos.json`).**
- Tempos: cautious 0.25, assertive 0.70, curious 0.25, hesitant 0.70, relaxed 0.40, energetic 0.25.
- Calibration speeds at these tempos: 0.0278–0.0303 m/s, all within ±5% of their mean.
- **Measured on V: the progress guarantee floors slow styles near 0.029 m/s.** So the shared speed range runs from
  ≈ 0.028 m/s (floor) to 0.068 m/s (full tempo).

**Task geometries.** Target distance {0.9, 1.3} m × bearing {0, +0.7, −0.7, +1.6} rad (8). Episodes of 60 s.
Latency alternates by trial. Initial joint noise 0.02 rad.

**Shift partition S:** 4 new geometries (1.8 m ahead; 1.8 m at 2.4 rad; 0.7 m at −2.4 rad; 1.3 m behind at 3.1 rad).

## Arms (partition X unless stated)

| arm | trials | content |
|---|---|---|
| core | 7 styles × 8 geometries × 5 seeds = 280 | X0 + X1 |
| matched | 6 × 8 × 5 = 240 | X1 at the calibrated tempos |
| random | 80 | X2, 80 random intents |
| push | 7 × 8 × 2 = 112 | X0 + X1, one push at m* = 0.15 m/s, random time in [4, 10] s and random direction |
| switch | 6 pairs × 4 geometries × 4 seeds = 96 | within-episode style change at 6 s (cautious↔assertive, hesitant↔energetic, relaxed↔curious) |
| shift (S) | 7 × 4 × 2 = 56 | core styles on the shift geometries |
| secondary (E1′) | 7 × 8 × 2 = 112 | core on the E1′ configuration (descriptive) |

## Analysis

- **Classifier:** shrinkage LDA (shrinkage 0.1), standardised features, leave-one-geometry-out cross-validation.
- **Feature sets:**
  - **non-speed** (primary) = temporal + posture + acceleration + spatial; it drops progress/path speed, arrival
    time, moving fraction and power;
  - **non-speed-non-posture** = temporal + acceleration + spatial (secondary). This checks distinctness without the
    head-at-rest channel. The pilot showed that the head channel alone separates the designed styles.
- **Matched-speed classification:** within the matched arm (6 classes).
- **Perturbation transfer:** a classifier trained on the core arm, tested on the push arm (non-speed).
- **Shift transfer:** a classifier trained on X core, tested on S.

## H1 gate (all required)

| criterion | definition |
|---|---|
| G1 safety | 0 falls and 0 monitor trips in the core, matched, random and switch arms; 0 falls in the push arm (m* = 0.15 m/s) |
| G2 task | arrival (final distance within ±0.15 m of the stopping distance) ≥ 95% for every style in core and matched |
| G3 distinctness | core, non-speed: balanced accuracy ≥ 0.80, and ≥ 4 of the 6 designed styles with recall ≥ 0.80 |
| G4 matched speed | matched arm, non-speed: balanced accuracy ≥ 0.70, and ≥ 4 of 6 styles with recall ≥ 0.70. Each style's mean progress speed within ±15% of the matched mean, else G4 is reported as "speed not matched" and fails |
| G5 perturbation robustness | push-arm transfer (non-speed) balanced accuracy ≥ 0.70 |

- **H1 passes** if G1–G5 hold.
- **The repertoire** = the designed styles with recall ≥ 0.80 in core and ≥ 0.70 in matched.
- **Reported, not gated:**
  - per-group accuracies, including non-speed-non-posture;
  - shift transfer;
  - X1 vs X2: falls, arrival, SPARC smoothness, power and diversity;
  - PCA dimensionality of X2;
  - the E1′ secondary arm;
  - switch safety.
- **No claim** about perceived meaning (H2 pending).
