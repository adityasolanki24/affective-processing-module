# Programme J final report: validation, NERVA transfer and closed-loop expressive development (2026-10-18)

**Branch:** `research/apm-v0.6-validation-transfer`.

**Phase reports:** `J1_REPORT.md`, `J2_REPORT.md`, `J4_REPORT.md`, `J5_REPORT.md` and `J6_REPORT.md`. J3 is in
`prereg/J3_ADDENDUM.md` and `RESEARCH_LOG.md`.

**Other records:** departures in `DEVIATIONS.md` (D1–D3); compute in `COMPUTE_LOG.md`.

Everything is simulation only; no humans, no hardware.

## Architecture decision (Phase 4): **E. Insufficient evidence to change the architecture; keep the simpler baseline.**

**The baseline kept:**
- explicit dynamic memory (programme I's type beliefs, with emissions learned per body), plus
- H's fixed lookup L0, into H's bounded expression adapter, on NERVA's untouched neutral controller and safety
  supervisor.

This is option A's structure. A is **not** declared to meet the criteria: J5 showed the fixed mapping does not close the
loop.

**Why not A:** with L0, the loop is not demonstrated (J5 Q3, Q4 fail).
- L0 compresses learned beliefs into a 0.04 m difference in stopping distance.
- Its "cautious" tempo and hesitation lengthen exposure.

**Why not B (learned mapping), yet:**
- J6 is exploratory (D2). Its best mapping depends on a task value chosen after the mapping was inspected (D3).
- That mapping still fails pushy → calm revision.

B is the **leading candidate**: a 5-parameter decision-theoretic mapping, with explicit task values, that beats L0 on
utility, contacts, push size and expressive range. It needs its own preregistered confirmation.

**Why not C or D (learned regulatory state, recurrent):** recurrent states add nothing.
- J4: the GRU memory is worse (AUROC 0.86 vs 1.00) and does not fix revision.
- J6: the GRU mapping copies the stateless mapping in 97.6% of choices. Its 0.09 s utility "gain" is nominal and
  practically nil.
- No neural architecture is presented as novel here.

**What would change the decision to B:** a confirmatory study, on fresh seeds, of the decision-theoretic mapping with a
priori task values. It should include an information-seeking term (programme I's value-of-information probing, ported
to NERVA), and pass J5's Q1–Q6, including pushy → calm revision.

## Answers to the eight questions

### 1. Do programme I's findings survive independent calibration and distribution shifts?

**Calibration: yes** (J1, 14,000 new lifetimes; thresholds frozen on V).
- R_dense adapts in 2 / 3 / 2 episodes (decisive 3 / 4 / 2), AUROC 1.00.
- Avoidable collisions −52.2% [−55.4, −48.8]; completion −0.25 pp.
- The decay-only control fails the adaptation measures, so they do not reward mere forgetting.

**Qualifications:**
- **The safety gain is not caused by revision.** R_decay, which stops learning at the switch, gives the same −55%.
  The benefit comes from holding discriminating type beliefs.
- Decision A's rule barely uses memory: the read-out moves 0.45 → 0.49.

**Shifts: only conditionally** (J2, 13 stress conditions). Programme I's memory (EU_I) is **not promoted as a universal
default**:
- it fails with 25% and 40% behavioural overlap between person types, and with heavy sensor noise (0.20);
- its safety benefit holds in 12 of 13 conditions (−14% to −42%), but vanishes with fast pedestrians;
- the context-conditioned alternative (EC) also fails both overlaps;
- self-caused near-misses caused no false attribution.

### 2. Does memory learning work on NERVA with robot-observable evidence?

**Partly.** Learning and retention work; revision does not.
- **J3:** NERVA's own IMU carries learnable evidence. Graded peak acceleration gives 0.55 nats vs 0.001 for an a-priori
  push detector.
  - EM finds K = 2 types without labels.
  - Two disclosed pilot iterations were needed: a self-calibrated detector, and a 4th "pushy" type within the safe
    push envelope.
  - Types below the sensing range (calm vs gentle) are not learnable, by construction.
  - No privileged simulator state (`base_linvel`) is used as an observation.
- **J4 (fails C2 and C3):**
  - history AUROC 1.00, specificity 0, generalisation to an unmet entity of the same kind +0.17, exact
    save/restore;
  - but the readout is compressed (caution effect +0.285 vs H's +0.387);
  - pushy → calm revision is essentially never reached within 24 encounters (12 of 163), for every memory.
- **Self-motion must be separated from perturbation:**
  - Without an efference-copy gate, the robot reads its own expressive motion as being pushed, and beliefs saturate.
  - Without sensing pauses, a continuously expressive robot cannot sense at all.

### 3. Can NERVA revise experience-dependent expressive tendencies?

**In one direction only.**
- Calm → pushy revision happens in 2–4 meetings: J4 type memories 4 blocks; J5 CL 4 encounters; J6 learned mappings 2.
- Pushy → calm essentially never happens within a lifetime, under any memory or mapping.
- **Causes:**
  - **evidence asymmetry:** a calm entity still trips the detector 21% of the time vs 52% for a pushy one;
  - **the physical avoidance trap:** a robot that stands back after bad experience rarely gets contact either way, so
    it cannot learn that the entity changed. The better a mapping avoids, the deeper the trap.

### 4. Do expressive actions change future experience?

**Yes, but with L0 the effect is small and partly counterproductive.**
- **J5 (confirmatory), L0:**
  - max push CL − FROZEN −0.003 (n.s.); contacts **+0.23**, because L0's slow, hesitant approach lengthens exposure;
  - each approach costs 8 s.
  - On J6-X's fresh seeds, L0's push effect is small but significant: −0.009 [−0.014, −0.004]. This is an
    exploratory replication.
- **J6 (exploratory), learned mapping:**
  - max push −0.027 [−0.031, −0.023] and about 1 fewer contact per late meeting with the risky entity, vs neutral;
  - 7.5 s faster than L0.
  - What the robot experiences next depends on how it chose to express.
- **Safety:**
  - 0 falls and 0 trips in every confirmatory system.
  - One uncertified risk was found: **repeated sub-envelope pushes** (3–5 in one encounter) caused 2 falls in an
    ablation (J5 RANDHIST). H's validated envelope covers single pushes only.

### 5. Does a learned mapping improve on the fixed mapping?

**Exploratory yes, with a crucial condition** (J6).
- **Time and perturbation costs alone:** the optimal mapping is "stop far, walk fast" for every belief. It is better on
  utility and safety, but has **no experience-dependent expression at all** (ρ 0.04).
- **With a stated task value for coming close:** the decision-theoretic mapping (LEARNED_V, 5 fitted parameters)
  beats L0 on:
  - utility: −11.9 s;
  - contacts: −0.04;
  - push size;
  - expressive range: 0.258 m vs 0.041 m;
  - and it keeps evidence dependence (ρ 0.87).
- A residual around L0 is worse than L0 on contacts, because it keeps L0's slow, hesitant approach.
- **The value of coming close is not learnable from physical consequences.** A designer or a person must supply it.
- Hesitation and slow tempo, H's main expressive channels, only cost time and exposure on this body. A utility-based
  mapping drops them.

### 6. Are persistent recurrent states necessary?

**No.**
- J4's recurrent memory is less discriminating (AUROC 0.86 vs 1.00), with a smaller history effect. Its faster
  forgetting buys only slightly more revision.
- J6's recurrent mapping reproduces the stateless mapping's choices 97.6% of the time.
- An explicit belief plus a stateless mapping with one forgetting rate does the same job, and is inspectable and
  restorable.

### 7. Does the architecture work across embodiments?

**The memory layer transfers; the expression layer does not.**
- **Transfers:** the same memory code (EM-learned types, change-aware beliefs, kind pooling, restart persistence) works
  on a wheeled robot (near-miss clearance) and on NERVA (IMU acceleration). Only the evidence model is re-learned per
  body, without labels.
- **Does not transfer:**
  - H's fixed L0 mapping, designed on H's belief scale, under-expresses NERVA's learned beliefs, and its caution styles
    are counterproductive physically;
  - revision speed is a property of the body's evidence channel, not of the memory.
- **Cross-body requirements:** the efference-copy gate and sensing pauses are general requirements for any robot that
  both expresses and learns from its own body's sensors.

### 8. What remains unsupported without humans or hardware?

- **Legibility and appropriateness.**
  - Whether people read stopping far vs near, or any of these behaviours, as meaningful, appropriate or emotional.
  - The user's H2 self-pilot suggested weak legibility of H's styles. No human data were collected in J, and none were
    simulated.
- **Real people.** Simulated "types" are stochastic lunge processes. Real people change for reasons, are many, and are
  not identity-tagged.
- **Real sensors and contact.** IMU noise, real contact dynamics and identity tracking errors on hardware. J2 showed
  identity overlap and sensor noise are exactly where the wheeled memory breaks.
- **Safety:** repeated sub-envelope perturbations on the physical NERVA. This is an uncertified risk; do not test it
  on hardware without a dedicated protocol.
- **Task values:** the value of approaching, and the price of a push, are human or task decisions.
- **Emotion:** nothing here shows, or should be described as, emotion. "Risky" and "pushy" are simulator processes,
  and the robot's state is a belief about perturbation.

## Phase-by-phase status

| phase | passed | failed | uncertain | next phase justified? |
|---|---|---|---|---|
| 0: audit | I's work preserved; PR #3 unmerged | — | — | yes |
| 1: J1, J2 | I1 under frozen calibration; validity control | universal default (overlap, noise); revision is not the safety cause | the decision pathway from memory | yes: transfer of the memory principle |
| 2: J3, J4 | evidence from IMU; EM types; experience dependence; persistence | J4 C2 (compressed readout), C3 (revision) | readout rescaling (post hoc, not claimed) | yes, with revision flagged |
| 3: J5, J6 | Q1, Q2, Q6; a learned mapping improves utility, safety and range (exploratory) | J5 Q3 (primary), Q4; pushy → calm everywhere | task values; legibility | Phase 4 only (decision E) |
| 4 | — | no option meets every criterion | — | a confirmatory B study with information-seeking is the proposed next programme |

## PR #3 (programme I): is merging justified?

**Yes, for programme I's stated claims, with qualifications recorded.**
- I1's adaptation, discrimination, safety and efficiency results survive independent calibration and new seeds (J1).
- The qualifications should accompany the merge. J's reports are linked from J's README; programme I's documents are
  not edited.
  1. The collision benefit comes from discriminating beliefs, not from revision (J1).
  2. The memory is not a universal default: it fails with ≥ 25% behavioural overlap and heavy sensor noise (J2).
  3. Decision A's rule barely uses memory (J1).
  4. G2b's memory is inverted and must not be adopted (J1 replicates it).

**The merge is not done: it awaits your approval.**

## Deliverables

| item | where |
|---|---|
| master preregistration, matrix, addenda | `MASTER_PREREGISTRATION.md`, `EXPERIMENT_MATRIX.md`, `prereg/J1–J6` |
| code and tests | `src/apm_transfer/` (removable); `scripts/`; `tests/test_j_transfer.py`, `test_j2_stress.py`, `test_j4_nerva_memory.py` |
| pilot and confirmatory results, statistics | `results/J1–J6` |
| behavioural comparison plots | `figures/` (`scripts/j_plots.py`) |
| paired NERVA videos (experience → subsequent expression) | `videos/` with `README.md` and `manifest.json` (re-simulation matches the logs exactly) |
| architecture comparison | this report; `J6_REPORT.md` (complexity table) |
| cloud and cost logs | `reports/COMPUTE_LOG.md`: about US$7–10 of the US$40 cap; final audit clean |
| raw-data hashes | `manifests/raw_data.json` (`scripts/j_manifest.py`) |
| deviations | `DEVIATIONS.md`: D1 (J2 analysis bug), D2 (J6 exploratory), D3 (J6 proximity value, GRU alignment) |

## Reproducing

- Each confirmatory addendum names the commit, inputs (git-normalised SHA-256) and exact command.
- **Wheeled runs:** `scripts/j_run.py`.
- **NERVA runs:** `scripts/j5_closed_loop.py`, `scripts/j4_lifetimes.py`, run in the NERVA venv with `NERVA_ROOT` set
  and `PYTHONPATH="src;<NERVA>"`, or on the cloud via `cloud/jcloud.py submit … --commit`.
- **Analyses:** `j1_analyze.py`, `j2_analyze.py`, `j4_analyze.py`, `j5_analyze.py`, `j6_analyze.py`.
- **Figures:** `j_plots.py`. **Videos:** `j_videos.py`.
- Raw data are git-ignored under `runs/J/`, hashed in `manifests/raw_data.json`, and kept in the project bucket.
