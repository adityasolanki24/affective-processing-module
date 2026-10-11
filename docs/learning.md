# Learning

The learning stack (`apm.learning`) is optional; the core never imports it. Its job is to predict **robot-relevant
consequences**, not emotion labels.

## What is learned, and from what

- **Experience:** an episode's AffectLink stream.
  - Canonical, masked features from `SelfState`, `GoalState` and `EntityObservation`: stability and actuation
    margins and their rates, energy margin, nearest-entity proximity and closing speed, goal type.
  - The outcomes that actually occurred (`OutcomeSignal`).
- **Labels:** "did outcome kind K occur within the next H seconds?", taken from observed outcomes only.
  - Predictions (`OutcomeHypothesis`) are never labels.
  - Samples during an ongoing event and after a fall are excluded.
  - No counterfactual credit: if the robot avoided something and nothing happened, nothing is learned about what
    would have happened.
- **No raw video or joint vectors.** That's what lets the same feature definition work on a biped and a wheeled
  robot.

## Models (all behind the same fit/predict interface)

| model | description |
|---|---|
| `GoalRate` | outcome memory alone: event rate per goal type |
| `MarginBins` | empirical frequency by stability-margin bin × goal type |
| `Logistic` | L2 logistic regression on the current features |
| `TiltExtrapolation` | the NERVA adapter's heuristic, as a reference |
| `GRUPredictor` | GRU (64 hidden) + 32-unit MLP head over a 2 s history; 17,857 parameters for 16 inputs |

Replacing the GRU with a Transformer, an active-inference model or a neuromorphic implementation means adding
another class with the same interface. Robots and AffectLink are unaffected.

## Replay and shadow learning (`apm.learning.replay`)

```
validated model ──► live predictions
      └──► experience buffer ──► shadow learner (replay mix: new + older) ──► frozen evaluation ──► promote / reject
                                                                     rollback ◄── previous validated model
```

- The live model never changes after a gradient step.
- **Promotion rule (Milestone C):** validation Brier must improve by ≥ 2% relative AND the protected validation
  subset (older experience) may regress by ≤ 5%. Otherwise the shadow is discarded.
- **Rollback:** every promotion keeps the previous model.
- Experience is kept in the buffer even when a shadow is rejected.

## What has been measured

See [experiments/C_learning/README.md](../experiments/C_learning/README.md) and
[experiments/D_portability](../experiments/D_portability/).

| question | NERVA biped result |
|---|---|
| calibration | GRU calibrated (ECE 0.002) |
| ranking | better than statistical baselines (AUROC 0.79 vs 0.74) |
| proper score | **not** better by the preregistered margin (Brier 0.0166 vs 0.0167, criterion C1 failed) |
| context generalization | failed (C4) |
| affect inputs | adding the APM's affect and appraisal inputs improved prediction (Brier 0.0152), confounded by higher-rate information (see that README) |
| shadow learning | rejected a corrupted update and rolled back correctly |
| incremental gain | essentially none in this run |

On the wheeled robot (Milestone D):
- the GRU beat logistic regression on slip prediction (Brier −17% at 240 episodes) and collision prediction (−38%);
- pretraining on the biped did not improve sample efficiency.

## Limits

- One simulated embodiment per dataset, one seed per model in Milestone C.
- Hand-set appraisal and affect constants.
- The GRU is evaluated in Python/PyTorch on a development PC. Embedded inference (ONNX Runtime / TFLite class) is
  untested; see docs/hardware.md.
