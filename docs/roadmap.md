# Roadmap

| milestone | content | gate |
|---|---|---|
| A | independent repository, AffectLink v1, APM-0 core, standalone demo | runs standalone; tests, lint, types pass |
| B | NERVA adapter, real stored telemetry, one bounded live rollout, observe-only | real simulator data flows through the unmodified core; replayable logs |
| C | statistical baseline, compact GRU outcome predictor, replay, shadow learner with promotion/rollback | held-out evaluation, calibration, forgetting and failure analysis recorded |
| D | minimal wheeled robot, its adapter, portability and preregistered transfer experiment | the same core runs unmodified; transfer judged against preregistered criteria |
| E | embedded hardware design, BOM candidates, failure behaviour, local resource benchmark | estimates separated from measurements; no hardware purchased |
| later | C/ROS 2/CAN FD transports, learned latent affect (only if it beats the baseline), human evaluation, licence and public release | each needs its own preregistration or authorization |
