# Safety boundaries

The APM is **never a safety authority and never a motor controller**.

- **Outputs are advisory.** Tendencies, attention, expression intent and expectations are offers to the robot's
  own planner. The robot decides whether to use them.
- **The robot owns safety.** Collision avoidance, fall protection, emergency stops and joint limits must work
  identically with the APM absent, late, wrong, or adversarial.
- **No new safety dependency.** No robot function may wait on, or fail because of, an APM output.
- **No foundation models in the loop.** Neither the core affect nor any learned predictor sits in the motor or
  safety path. Interpreted cues from a language or vision model enter only as `SemanticCue` with confidence and
  provenance. The baseline records them but does not let them drive affect.

## Failure behaviour (required of every integration)

| event | robot behaviour | APM behaviour |
|---|---|---|
| APM loses power or crashes | continues its ordinary task/motor control; ignores missing advisory outputs after a timeout | on restart: restores the last saved state, starts a new epoch, waits for fresh self state |
| network cable unplugged / link down | same as above | inputs go stale; `Diagnostics.health = stale_inputs`; affect relaxes toward baseline; no memory updates from absent data |
| messages arrive late | n/a | rejected beyond `max_age_s`; counted in diagnostics |
| storage corrupt | n/a | load fails its version/structure checks; the APM starts with empty memory and reports it, rather than using partial state |
| neural learner fails or diverges | n/a | the validated inference model keeps serving; the shadow learner is discarded; rollback to the last validated model |
| host robot reboots | sends `Reset` with a new epoch | clears that source's context and ordering state; keeps learned memory |

## Learning safeguards

- **Teaching signal.** Memory and learned predictors are trained only on observed outcomes (`OutcomeSignal`),
  never on the APM's own affect and never on predicted risks. This prevents self-reinforcing fear-like loops.
- **No counterfactual credit.** If the robot avoided a person and nothing happened, nothing about "what would have
  happened" is learned.
- **Shadow learning.** A learned model changes live behaviour only after frozen evaluation and explicit promotion;
  rollback is always available (docs/learning.md).
