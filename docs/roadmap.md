# Roadmap

| milestone | content | gate | status |
|---|---|---|---|
| A | independent repository, AffectLink v1, APM-0 core, standalone demo | runs standalone; tests, lint, types pass | done |
| B | NERVA adapter, real stored telemetry, one bounded live rollout, observe-only | real simulator data flows through the unmodified core; replayable logs | done |
| C | statistical baseline, compact GRU outcome predictor, replay, shadow learner with promotion/rollback | held-out evaluation, calibration, forgetting and failure analysis recorded | done (C1, C4 failed; see results) |
| D | minimal wheeled robot, its adapter, portability and preregistered transfer experiment | the same core runs unmodified; transfer judged against preregistered criteria | done (portability yes; learned transfer no) |
| E | embedded hardware design, BOM candidates, failure behaviour, local resource benchmark | estimates separated from measurements; no hardware purchased | design + PC benchmark; no hardware |
| later | C/ROS 2/CAN FD transports, learned latent affect (only if it beats the baseline), human evaluation, licence and public release | each needs its own preregistration or authorization | open |

## Next decisive experiment

A fair test of the affect hypothesis on both bodies.
1. Give the plain predictor the same high-rate summary information the appraisal vector carries.
2. Test whether *any* part of the affective machinery (appraisal vector, PAD, memory predictions) improves
   prediction or behaviour beyond that.
3. Run multiple seeds per robot, with the calibration of the baseline appraisal made embodiment-aware through the
   profile.

Preregister before running.
