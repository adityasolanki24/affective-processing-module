# H3/H4 addendum: internal-state representations and experience-shaped expression (fixed before H3/H4 test data)

**Prerequisite:** the H1 gate must pass (`reports/H1_REPORT.md`). If it does not, H3/H4 are not run.

**Code:** this addendum governs `scripts/h34_experience.py`, `scripts/h34_states.py`, `scripts/h34_probes.py` and
`scripts/h34_analyze.py` at the commit that adds it. Pilot data (partition P, `results/H34/P/`) were used only for
the design choices in §1.

## 1. Developmental testbed (`apm_express.develop`)

**Lifetimes.** A lifetime is 48 encounters. At each encounter the robot meets:
- one of 6 identities (A1, A2, B1, B2, C1, C2; kinds A/B/C, all visible);
- in one of 2 visible zones (plain, marked).

**Jolts.** The entity jolts the robot with the world's hidden probability:
- a real 0.15 m/s push in a random direction;
- 0.15 m/s is the standing-validated magnitude from H0b, so it is recoverable.

**Histories** (the experimental manipulation, defined only by the jolt probabilities):

| history | jolt probability |
|---|---|
| safe | all 0.05 |
| risky_B | kind B 0.85, others 0.05 |
| risky_A | kind A 0.85 (symmetry control) |
| unpredictable | all 0.35 |
| change_B | B 0.85 for encounters 0–23, then 0.05 (revision) |
| late_B | B 0.05 then 0.85 (acquisition) |
| zone_B | B 0.85 only in the marked zone (context) |

**Observations** (all the robot gets): identity tag, kind, zone, and the measured disturbance.
- Observed outcome y = 1 if peak horizontal base speed in the 4 s after the encounter is ≥ 0.04 m/s.
- Fixed from the pilot: 672 encounters; jolted min 0.066 m/s; non-jolted max 0.018 m/s; 0 false or missed events.
- Measured peak tilt does **not** separate them at this magnitude (38% of jolts exceed 0.5°), so tilt is not used.
- The magnitude input is peak speed / 0.1 m/s.

**Design simplification.** Experience is measured with neutral motion, so every model sees an identical stream.
Expression is communicative only: it does not change what the robot experiences.

## 2. Partitions (H4 seeds, `seeds.py`)

| partition | lifetimes per history | total | use |
|---|---|---|---|
| T | 40 | 280 | train learned models |
| V | 10 | 70 | tune every family, equal budget |
| X | 30 | 210 | test |

## 3. Systems

**Intent-level systems.** Every system observes the same streams. Expression is the shared fixed lookup L0
(`apm_express.states.lookup_intent`):

    κ = clip(p + 0.3·(g − 0.05))
    intent = blend(relaxed, cautious, κ)
    hesitation += 0.6·u
    scanning   += 0.5·u

PAD uses its own fixed mapping (`pad_intent`).

| system | description |
|---|---|
| S0 | no state (current observation only, which carries no risk cue): the **current-risk-only baseline** |
| S1 | fixed style (relaxed) |
| S2 | scalar EMA of outcomes |
| S3 | small continuous state (global risk, volatility, identity novelty) |
| S4 | PAD (global; appraisal-driven) |
| M_id | **A0 explicit memory**: per-identity Beta + kind pool prior + global EMA risk filter |
| M_ctx | memory keyed by identity × zone |
| M_id_cp | change-point identity beliefs (programme G's G2 BOCPD) |
| S5 | GRU latent |
| S6 | fast/slow GRU latent |
| S7 | GRU + explicit memory inputs |
| oracle_lookup | **human-designed history-to-style lookup**: it is told the history and the true rule (control, not a candidate) |

**Equal tuning budget.**
- Designed and memory families: 8 configurations of the smoothing α each, chosen by V log loss of p.
- Learned families: 8 configurations each (hidden 16/32 × lr 1e-3/3e-3 × epochs 40/80), on seed 0, chosen by V
  loss; then 5 seeds form an ensemble.
- Learning objective: **world prediction only**, i.e. the next observed outcome. No expression labels exist.

**Simplicity order:** S0 < S1 < S2 < S3 < M_id < M_ctx < M_id_cp < S4 (specialised 3-D) < S5 < S6 < S7.

## 4. Evaluation situations (identical for every robot)

**Probes:**
- B1/plain, B1/marked, A1/plain, C1/plain;
- B3 (a novel identity of a known kind);
- D1 (an unknown kind).

Probes are read every 4 encounters and at the end of the lifetime.

**Truth.** The world's hidden probability at that time, used for evaluation only. D1 has no truth and is used for
disposition only.

## 5. Metrics

| code | metric |
|---|---|
| E1 (primary) | **Evidence consistency:** Spearman ρ between caution κ and the truth, over all end-of-lifetime probes with truth (B1×2 zones, A1, C1, B3) and all X lifetimes |
| E2 (primary, the "most important test") | **History effect:** mean κ(B1/plain | risky_B) − κ(B1/plain | safe) at lifetime end, paired by lifetime index; 95% CI from a bootstrap over lifetime pairs |
| E3 | **Causal controls** on E2: <br>• shuffled outcomes (outcomes permuted across encounters) <br>• randomised identity tags (kind kept) <br>• state reset before evaluation <br>• S0 (current-risk-only) and S1 (fixed style) <br>• oracle lookup (reference) |
| E4 | **Revision** (change_B): the number of encounters after the switch until κ(B1/plain) falls below the midpoint of its value at k = 24 and the safe level, with Kaplan–Meier median |
| E5 | **Acquisition** (late_B): the same, upward |
| E6 | **Kind generalisation:** κ(B3 novel) under risky_B minus under safe |
| E7 | **Context:** κ(B1/marked) − κ(B1/plain) under zone_B |
| E8 | **Disposition:** κ(D1 unknown kind) under unpredictable minus under safe (reported, no "correct" value) |

## 6. Physical "most important test" (`h34_probes.py`)

**Setup:**
- For each X lifetime index i, two robots with **identical** initial state, weights, simulator seed and situation;
- robot 1 lived history risky_B, robot 2 history safe;
- both approach B1 in the plain zone (geometry 1.0 m ahead);
- each uses its system's end-of-lifetime intent through the H0b interface;
- 30 pairs per system.

**Systems tested physically:**
- M_id (A0 + memory + fixed lookup, the baseline);
- the best learned state by E1 (S5–S7);
- S4 (PAD);
- S0 and S1 (controls).

**Measured:**
- motion features;
- safety (falls, trips);
- arrival;
- a leave-one-pair-out classifier predicting the history from non-speed motion features.

## 7. H4 gate (developmental-expression claim, preregistered)

A system supports claim C (in simulation; interpretability untested) **only if all** hold:
1. E2 > 0.20 with 95% CI lower bound > 0;
2. E1 ρ ≥ 0.50;
3. under shuffled outcomes the E2 CI includes 0, or |E2| < 0.05;
4. under state reset, E2 = 0;
5. physical history classification ≥ 0.80, with 0 falls and ≥ 95% arrival in its probe trials.

S0 and S1 must show E2 = 0 (a check of the design). If no system passes, the H4 gate fails: **no developmental
claim.**

## 8. H3 selection and the "beats the baseline" rule

**H3 selection.** The smallest system (§3 order) whose E1 is non-inferior to the best E1. Non-inferiority: the
one-sided 95% bootstrap lower bound of ρ(system) − ρ(best) is > −0.05. E2, E4, E6 and E7 are reported for every
system.

**A learned or specialised state "beats" A0 + memory + fixed lookup (M_id) only if** both:
- its E1 exceeds M_id's with a 95% CI lower bound > 0;
- it is not worse on E2, E4 and E6 (point estimates no lower).

Otherwise the conventional baseline is retained.
