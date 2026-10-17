# Roadmap

| milestone | content | gate | status |
|---|---|---|---|
| A | independent repository, AffectLink v1, APM-0 core, standalone demo | runs standalone; tests, lint, types pass | done |
| B | NERVA adapter, real stored telemetry, one bounded live rollout, observe-only | real simulator data flows through the unmodified core; replayable logs | done |
| C | statistical baseline, compact GRU outcome predictor, replay, shadow learner with promotion/rollback | held-out evaluation, calibration, forgetting and failure analysis recorded | done (C1, C4 failed; see results) |
| D | minimal wheeled robot, its adapter, portability and preregistered transfer experiment | the same core runs unmodified; transfer judged against preregistered criteria | done (portability yes; learned transfer no) |
| E | embedded hardware design, BOM candidates, failure behaviour, local resource benchmark | estimates separated from measurements; no hardware purchased | design + PC benchmark; no hardware |
| F (v0.2) | behavioural learning with the v0.1 core | preregistered E0–E7 | done |
| G (v0.3) | architecture selection | decision A: predictor + entity memory + EMA risk penalty | done |
| H (v0.4) | developmental expressive intelligence | decision B: memory → fixed lookup → bounded motion adapter | done (human study pending ethics) |
| I (v0.5) | autonomous revision: escaping the avoidance trap | preregistered; see `experiments/I_*` | next |
| later | decision-A runtime replacing the PAD core; C/ROS 2/CAN FD transports; powered promotion gate for deployment-time learning; human evaluation (H2); licence and public release | each needs its own preregistration or authorization | open |

## Next decisive experiment

The affect-hypothesis test proposed here at v0.1 was run in programmes F and G; PAD did not survive it.

The next question is **autonomous revision**: a robot that has learned to avoid an entity stops collecting
evidence about it, so it cannot notice when the entity becomes safe (or keeps being trusted after it becomes risky
without being observed). This is programme I. It must show that a robot choosing its own exposure can keep its
beliefs accurate after the world changes, at a bounded safety cost, and beat decision A's memory and G2b's
revisable memory under preregistered criteria.
