# Writing an embodiment adapter

An adapter translates one robot's signals into AffectLink and consumes APM's advisory outputs. It lives at the
integration boundary (`apm.adapters.<robot>` for reusable adapters, or in the robot's own code); the core never
imports it.

1. **Declare the body** with an `EmbodimentProfile`:
   - `self_fields`: only the margins you can actually provide;
   - `outcome_kinds`: only outcomes you can actually detect;
   - `MarginCalibration` entries for raw quantities (nominal, critical, meaning).
2. **Self state.** Send `SelfState` periodically, about 5–20 Hz, as `Quantity` values (fraction or calibrated raw
   units). Mark anything unavailable as `MISSING`/`UNSUPPORTED`. Never send 0 for "don't know".
3. **Entities.** Send `EntityObservation` with your tracker's temporary `track_id`. Add a persistent `identity` and
   `identity_confidence` only if your perception can re-identify.
4. **Goals.** Send `GoalState` when goals change and periodically while active.
5. **Predictions.** If the robot has predictors (time to contact, slip risk), send `OutcomeHypothesis` with an
   explicit kind, probability and horizon.
6. **Outcomes.** Detect what actually happened and send `OutcomeSignal`:
   - `confirmed=True` for direct measurement, `False` for an estimate;
   - set `evidence_quality` honestly;
   - use a unique `outcome_id`.
   Document every detector's threshold.
7. **Headers.** Use `apm.adapters.base.Emitter(source)`. On a robot restart, call `restart()` and send `Reset`.
8. **Outputs.** Read `ActionTendency`, `AttentionBias` and `ExpressionIntent` as suggestions. Never route them
   around the robot's safety layer.

Examples: `apm.adapters.nerva` (biped; Milestone B) and `apm.adapters.wheeled` (Milestone D).
