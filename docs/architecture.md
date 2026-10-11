# Architecture

```
 robot sensors / simulator / planner                      (robot side; any body, any middleware)
        │
        ▼
 embodiment adapter  ── robot-specific: which signals exist, calibrations, outcome detectors
        │  AffectLink v1 messages (versioned, units, uncertainty, provenance)
        ▼
 ┌──────────────────────────── APM core (robot-independent, NumPy only) ───────────────────────────┐
 │ StreamGate      validation, ordering, duplicates, staleness, resets                             │
 │ Context         bounded current view: tracks, self margins (calibrated), goals, live predictions │
 │ Appraisal       explicit frames: hypothesis / memory prediction / observed outcome               │
 │ Affect          tonic attractor p* = tanh(W u) + decaying phasic impulses → persistent PAD       │
 │ Tendencies      per-target advisory approach/avoid/orient/explore/disengage/seek-assistance      │
 │ Memory          entity & context outcome statistics (from OutcomeSignal only), episodes          │
 │ Learning (opt.) outcome predictors (statistical, GRU), replay, shadow learner, promotion         │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
        │  AffectState, AppraisalFrame, ActionTendency, AttentionBias, ExpressionIntent,
        │  MemoryExpectation, Diagnostics (advisory)
        ▼
 robot planner / expression layer (may ignore everything; owns safety)
```

## Concepts kept separate

| concept | where | what it is |
|---|---|---|
| perception | robot + adapter → `EntityObservation`, `RelationObservation`, `SemanticCue` | what the robot observes |
| prediction | `OutcomeHypothesis` (robot's predictors); memory's own predictions; `apm.learning` predictors | what might happen |
| outcome | `OutcomeSignal` | what actually happened |
| appraisal | `apm.core.appraisal` → `AppraisalFrame` | what a predicted or actual outcome means relative to goals and self state |
| affect | `apm.core.affect` → `AffectState` | persistent internal control state (PAD readout) |
| action tendencies | `apm.core.tendencies` → `ActionTendency` | advisory preferences about possible behaviours |
| personality-like adaptation | `apm.memory` (and learned predictors) | relatively stable differences in expectation that grow from each robot's experience |

## Key design choices

- **Attractor, not accumulator.** The current tonic context sets a bounded target. Repeating an unchanged
  observation only re-asserts the same target, so it cannot accumulate. Phasic impulses come only from new
  outcomes, each counted once.
- **Rate invariance.** Affect is integrated exactly over timestamp differences, piecewise constant in the
  context. Running the same scene at 10 Hz and at 100 Hz gives the same state (tested).
- **Bounded memory.** Every store has a capacity: 64 tracks, 128 hypotheses, 1,024 entity records and 2,048
  episodes.
- **Memory predicts but is never taught by its predictions.** Entity statistics produce internal hypotheses for
  tracked entities, appraised like any prediction. Only observed outcomes change the statistics.
- **PAD is experimental.** Three numbers are a compact, interpretable readout, not a claim about the true
  dimensionality of machine affect. Stage-2 experiments test whether PAD adds predictive or behavioural value
  beyond outcome memory and prediction alone.

## Reuse from NERVA

APM is an independent implementation. These ideas were carried over from the NERVA project (the first
embodiment) as design lessons, with no code copied:
- learn from measured outcomes rather than elicited emotion;
- keep risk estimates separate from outcomes;
- make sleep or replay add no evidence;
- appraisal frames should refer to explicit hypotheses;
- self state should lower controllability and raise relevance.

Not carried over:
- discrete emotion categories (NERVA's Model A);
- NERVA's additive drive dynamics (replaced by the attractor);
- appearance-embedding identity matching (an adapter concern);
- anything robot-specific.
