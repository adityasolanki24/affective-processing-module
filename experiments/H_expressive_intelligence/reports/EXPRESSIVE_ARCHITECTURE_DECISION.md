# APM v0.4 expressive architecture decision (programme H, 2026-10-13)

## Decision: **B — experience-conditioned expressive mapping**

**What it is.** Explicit, outcome-grounded memory feeding a small, fixed, transparent mapping into a bounded,
embodiment-specific expression adapter on a validated controller. This is sufficient to make identical robots
with different experiences move reliably and coherently differently in the same situation. In simulation, the
lived history was recoverable from motion in 98–100% of cases, with 0 falls.

**What was not shown.** No recurrent latent, specialised (fast/slow, PAD) architecture or learned mapping beat it
under the preregistered rules. So C and D are not supported.

**Why not A.** Developmental expression was demonstrated beyond fixed styles (H4b, H8).

**Why not E.** Every gate needed for B eventually passed: H0 → H0b, H1 → H1b, H4 → H4b. Each failure stays on record
with its diagnosis.

**The main caveat.** **Human interpretability (claim B) is untested.** The H2 protocol and 72 anonymised stimulus
clips are ready; no participant has been recruited, pending an ethics determination and your authorization.
Everything below is simulation only.

## Evidence base

| experiment | outcome | report |
|---|---|---|
| H0 interface | **failed**: standing push falls; slow-probe arrival | `H0_REPORT.md` |
| H0b re-test | **passed**: dwell 0.4 s, head box at rest ±0.4 rad, progress guarantee, standing push tolerance 0.15 m/s | `H0b_REPORT.md` |
| H1 repertoire | **failed G1**: 1/112 falls at 0.15 m/s after a stop. G2–G5 passed: distinctness 1.00 from non-speed features; matched speed 1.00 | `H1_REPORT.md` |
| H1b | **passed**: stop-transition tolerance 0.10 m/s; 0/168 falls | `H1b_REPORT.md` |
| H2 perception | **pending**: protocol, power simulation, 72 clips; no participants | `../prereg/H2_PROTOCOL.md` |
| H3 states | smallest adequate: explicit context memory; nothing beats A0 memory; PAD weakest | `H34_REPORT.md` |
| H4 developmental | **preregistered gate failed** (mis-specified shuffle control) | `H34_REPORT.md` |
| H4b developmental | **passed** with the corrected pooled-shuffle control on fresh data: A0 memory E2 +0.83, physical history classification 1.00, 0 falls; also GRU, fast/slow and GRU + memory with smaller effects | `H34_REPORT.md` |
| H5 learned mapping | learned mappings better calibrated to designer references (L2 GRU preferred), but compress history effects (0.49–0.59 vs 0.84) | `H5_REPORT.md` |
| H6 dispositions | revisable dispositions from global states; the scalar EMA is simplest; PAD not better | `H34_REPORT.md` |
| H7 embodiment | measured qualities correspond across biped and kinematic wheeled base (median ρ 0.71, transfer 0.79); interpretability untested | `H7_REPORT.md` |
| H8 integration | baseline retained; integrated variant gains revision and context but loses history effect | `H8_REPORT.md` |

**Scale:**
- ≈ 8,000 simulated NERVA motion trials;
- ≈ 40,000 physically measured experience encounters;
- 1,260 physical "two-robot" episodes.

All local CPU; **US$0 cloud spend.**

## Answers to the twelve questions

1. **What movement styles can NERVA reliably express?** Six designed qualities plus neutral (cautious, assertive,
   curious, hesitant, relaxed, energetic).
   - They are carried by command timing (pauses, stop-go, retreat), path shape, stopping distance, body orientation
     and head posture at rest.
   - They are distinguishable from non-speed motion in unseen geometries (1.00), at matched speed (1.00), under
     pushes (0.98) and in new geometries (0.98), with 100% task completion.
   - **Not available on the validated controller:** cadence, stride, crouch, head motion while walking, torso pitch
     beyond ≈ 2° (E1′).
   - **Speed range** is 0.029–0.065 m/s.
   - **Push tolerance** during expression is 0.10 m/s, versus 0.30 m/s for steady walking.
2. **Which can human observers recognise?** **Unknown.** No human data (H2 pending).
3. **Does experience reliably change expressive motion?** **Yes, in simulation.**
   - Identical robots with identical present observations, after 48 encounters of different actual experience,
     express different caution: E2 +0.83 [0.79, 0.87] for A0 memory.
   - Their lived history is recoverable from their motion in 98–100% of pairs, across three situations including a
     never-seen entity and a new geometry.
   - There were 0 falls in 1,260 such episodes.
4. **Is the change caused by history rather than current risk?** **Yes.**
   - The present observations were identical and carry no risk cue.
   - The current-risk-only baseline (S0) and fixed style (S1) show exactly 0.
   - Pooling and randomly reassigning the two robots' outcomes removes the effect (+0.04 [−0.02, 0.11]).
   - State reset removes it.
   - Within a lifetime, entity specificity disappears when outcomes are shuffled (+0.79 → −0.02).
   - Caveat: the preregistered H4 control was mis-specified and its gate failed; the conclusion rests on H4b and H8.
5. **What internal state representation works best?** **Explicit outcome-grounded memory:**
   - per-identity Beta beliefs with a kind-level pool prior, plus a global EMA.
   - It gives the strongest, most specific and best-generalising (kind) history effects at negligible cost.
   - Context keying (identity × zone) is needed when outcomes depend on context.
   - Change-point beliefs revise faster.
   - A global scalar EMA is the simplest source of revisable dispositions.
   - Learned recurrent states revise fastest but express history weakly. They also absorb training-set priors: S7
     learned "kind B is risky" from the training distribution.
6. **Is PAD useful for expression?** **No evidence that it is.** As a global state it:
   - is less evidence-consistent (0.41) than every memory or learned state;
   - expresses smaller history effects (+0.23) than a plain scalar EMA (+0.31);
   - revises dispositions more slowly (20 vs 8 encounters);
   - cannot represent entity-specific expectations.

   This parallels v0.3 (harmful for regulation), but it is a separate finding about expression.
7. **Does explicit long-term memory improve expressive development?** **Yes.** Removing memory (global state only)
   drops evidence consistency from 0.73 to 0.41 and the history effect from 0.78 to 0.36, and removes kind
   generalisation and context sensitivity (H8).
8. **Does a learned expression policy beat a scripted mapping?**
   - **For calibration to designer references, yes:** H5 CE 0.126 (GRU) vs 0.167, faster revision.
   - **For expressing what the robot has experienced, no:** the history effect is 0.49–0.59 vs 0.84.
   - **In H8's preregistered comparison the scripted lookup was retained.**
   - The learned mappings used privileged simulator truth, which a deployed robot would not have.
9. **Does a specialised neural network beat an ordinary recurrent model?** **No.**
   - Fast/slow is within the non-inferiority margin of the GRU on evidence consistency, with a smaller history
     effect.
   - GRU + memory equals GRU as a mapping (H5).
   - The ordinary GRU is preferred among learned options, and the explicit memory is preferred overall.
10. **Can the mechanism generalise across embodiments?** **Partly, as a hand-designed contract.**
    - The same `ExpressionIntent` produces corresponding measured qualities on the biped and a kinematic wheeled
      base: median ρ 0.71; a biped-trained classifier recognises wheeled styles at 0.79.
    - Gaze and pause structure do not correspond.
    - There is no learned transfer. The wheeled body is kinematic. Interpretability is untested.
11. **What is the minimal computational architecture?** **No neural network is required:**
    - a few counts per entity and kind, one global scalar, an 8-parameter mapping, the adapter and the monitor;
    - ≈ 2 µs per experience update and ≈ 0.2 ms per expression on a laptop CPU.
12. **What is still untested or unsupported?**
    - Human interpretability (claim B).
    - Any hardware behaviour.
    - Other kinds of experience: only push outcomes while standing were used, and expression did not change what was
      experienced.
    - Learning the expression mapping from human judgements.
    - The right weight for combining dispositions with entity memory.
    - Context keying's slightly non-null control in H8.
    - The NERVA controller's low push tolerance when standing or just stopped.
    - Anything about emotions, feelings or consciousness.

## Recommended APM architecture (v0.4)

```
 observations ──► A0 world/consequence layer (v0.3, frozen) ──► regulation (v0.3, frozen) ──► task decisions
      │                                                                                        │
      │ outcome of each encounter (measured)                                                  │ task goal
      ▼                                                                                        ▼
 EXPLICIT MEMORY  ── per identity i: (n_i, k_i)   per kind c: (n_c, k_c)   global g ──► READOUT (p, u, g)
                                                                                         │
                                                                    FIXED LOOKUP L0 (8 numbers)
                                                                                         ▼
                                                                ExpressionIntent (8 bounded channels)
                                                                                         │
             EmbodimentExpressionAdapter (biped: 7 validated commands, dwell ≥ 0.4 s, head only at rest,
             |offsets| ≤ 0.4 rad, ≤ 0.75 rad/s, progress ≥ 0.025 m/s, retreat ≤ 25%)  | wheeled: (v, ω) limits
                                                                                         ▼
                       ValidatedMotionController (NERVA neutral candidate, its own contract; frozen)
                                                                                         ▲
                      SafetyMonitor (physical state only: tilt ≥ 25°, |ω| ≥ 5 rad/s, joint margin < 0.02 rad,
                      25-step force saturation) ─► neutral rest fallback, overrides everything above
```

### State-update equations

Each observed outcome is y ∈ {0, 1}, perturbed or not as measured; weight w = 1; p₀ = 0.05; s = 2.
- **Kind pool:** n_c ← n_c + w, k_c ← k_c + w·y, and p_c = (k_c + s·p₀) / (n_c + s).
- **Identity:** n_i ← n_i + w, k_i ← k_i + w·y, and p_i = (k_i + s·p_c) / (n_i + s), conf_i = n_i / (n_i + s).
  - Change-point variant: Bernoulli BOCPD with hazard 0.05, the pool as prior.
  - Context variant: keys (i, zone) and (c, zone).
- **Global:** g ← g + α·(y − g). α was 0.02 as tuned; H6 supports α ≈ 0.07 for revisable dispositions.
- **Readout** for a probe:
  - p = p_i if the identity is known, else p_c;
  - u = 1 − conf_i, or 1 − 0.5·conf_c for an unseen identity.

### Expression mapping L0 (fixed)

    κ = clip(p + 0.3·(g − p₀), 0, 1)
    intent = (1 − κ)·relaxed + κ·cautious
    hesitation += 0.6·u
    scanning   += 0.5·u
    clip every channel to its range

Channel values (relaxed → cautious):
- tempo 0.75 → 0.5;
- hesitation 0 → 0.4;
- proximity 0.6 → 0;
- directness 0.6 → 0.7;
- gaze 0 → 1;
- scanning 0.15 → 0.3;
- lean 0 → −1;
- head elevation 0 → −0.5.

### Neural models

**None in the recommended architecture.** Tested alternatives (parameters):

| model | parameters |
|---|---|
| GRU state (S5) | 2,657 |
| fast/slow (S6) | 2,273 |
| GRU + memory (S7) | 2,753 |
| L1 MLP mapping | 648 |
| L2 GRU mapping | 6,568 |
| L3 GRU + memory mapping | 6,920 |

All were trained with Adam on world-prediction (S5–S7) or designer-reference (L1–L3) losses.

### Learning objectives

- **World learning:** the memory's Beta updates. Learned alternatives use binary cross-entropy on the next measured
  outcome.
- **Regulatory learning:** v0.3, unchanged.
- **Expressive learning:** none in the recommendation. The mapping is fixed and transparent. If it is ever learned,
  learn it from approved human judgements (H2), not from invented emotion labels.

### Interface, safety and latency requirements

**`ExpressionIntent`.** Eight bounded channels; NaN maps to neutral:
- tempo, hesitation, proximity, directness ∈ [0, 1];
- gaze, lean, head elevation ∈ [−1, 1];
- scanning ∈ [0, 1].

**Adapters.** Accept only intents and emit only validated controller requests. The biped adapter enforces:
- dwell;
- head-at-rest;
- rate limits;
- the progress guarantee;
- the retreat cap.

**The monitor is independent.** Its signature takes no expressive inputs (tested). Fallback is neutral rest.

**Certified push tolerance: 0.10 m/s** (stop transitions). It is 0.15 m/s at steady rest and 0.30 m/s while
walking. The expression layer must not be deployed where larger disturbances are expected, until the controller is
improved.

**Latency.** The controller runs at 50 Hz. Expression costs ≤ 1 ms per update, well inside the 20 ms control
period.

## Supported claims (simulation)

- **A (expressive motion):** six distinct, reliably controllable movement qualities on NERVA's validated
  controller, safe within the stated envelope.
- **C (developmental expression):** actual experienced outcomes change expressive motion in identical situations.
  The change is caused by history, specific to entity and kind, revisable, and keeps the task and stability intact.
- **Revisable dispositions** toward the unfamiliar arise from a simple global state.
- **The architectural point:** explicit memory + fixed mapping is sufficient and was not beaten by recurrent or
  specialised models.

## Unsupported or untested claims

- B (interpretability) for any style, body or history.
- Subjective emotion, feelings, personality, consciousness.
- Hardware.
- Learned cross-embodiment transfer.
- Benefits of PAD.
- Superiority of learned expression for communication.
- Generalisation to experiences other than physical perturbation.

## Roadmap

1. **Ethics determination, then the H2 study.** The protocol, power analysis and stimuli are ready. Extend the stimuli
   to the two-robot H4b pairs (can people tell which robot had which history?).
2. **NERVA controller robustness while standing and just after stopping.** This currently limits safe expression
   to 0.10 m/s pushes. It is a controller task (training with standing pushes), with its own preregistration and
   budget.
3. **Memory upgrades with re-checked controls:** change-point beliefs for revision, context keying where context
   matters, and a global state weighted to give revisable dispositions. Choose the weight on V with an explicit
   expressive criterion, then test on fresh X.
4. **Closed-loop experience,** where expression changes what is experienced (keeping distance changes exposure).
   Richer outcome types: task success, social responses.
5. **Learning from human judgements,** but only after H2 shows which differences people perceive.
6. **Hardware,** only with your separate authorization and with NERVA's missing hardware e-stop and torque limits in
   place.
