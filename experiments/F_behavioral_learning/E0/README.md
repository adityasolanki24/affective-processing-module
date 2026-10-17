# E0: experimental integrity and the controllable wheeled interface — gate PASSED

Hypothesis (preregistered): the simulation and logging infrastructure can support valid, causal comparisons between
decision-making systems. Runner: `scripts/e0_integrity.py` (pilot partition only). Results:
`results/E0/e0_results.json`.

## What was built

| component | file |
|---|---|
| intervention-capable testbed (separate RNG streams, hidden pedestrian dispositions, goal deadlines, safety supervisor, contact episodes) | `src/apm_testbeds/wheeled_ctrl.py` |
| AffectLink adapter for it (retracts stale TTC predictions) | `src/apm/adapters/wheeled_ctrl.py` |
| harness: lifetimes, common random numbers, GO fallback on any policy failure, metrics | `src/apm_research/harness.py` |
| explicit outcome memory (the APM `EntityMemory`, same lifecycle) | `src/apm_research/memory.py` |
| decision features shared by all systems | `src/apm_research/features.py` |
| action-conditioned consequence predictor | `src/apm_research/predictor.py` |
| B0, exploration, B1, B2 (+ ablation flags), Oracle | `src/apm_research/policies.py` |
| provenance and the confirmatory-run lock | `src/apm_research/provenance.py` |

The v0.1 testbed and adapter are unchanged. The core changed only for the D1/D2 fixes (`AUDIT.md`).

## Gate tests (`tests/test_wheeled_ctrl.py`, `tests/test_v02_defects.py`): 19 / 19 pass

| preregistered requirement | test |
|---|---|
| interventions change subsequent motion and outcomes (GO vs SLOW is a real trade-off) | `test_actions_change_subsequent_motion_and_outcomes` |
| same legal actions for everyone; illegal actions refused; harness falls back to GO | `test_illegal_actions_are_refused_and_the_harness_falls_back` |
| one safety supervisor, independent of the policy; replay is exact | `test_safety_supervisor_is_policy_independent` |
| APM failure leaves navigation and safety working | `test_apm_failure_falls_back_to_ordinary_navigation` |
| standing still cannot succeed | `test_always_yield_does_not_succeed` |
| no hidden state or future information in observations; policies hold no simulator reference | `test_observations_hold_no_hidden_state_or_future_information`, `test_decision_systems_hold_no_reference_to_the_simulator` |
| common random numbers (population, goals, layout independent of the policy) | `test_goal_sequence_and_population_do_not_depend_on_the_policy` |
| no duplicate or missed collision events vs ground-truth contact logs | `test_collision_events_match_ground_truth_contact_episodes` |
| adapter messages valid, none rejected, no stale prediction held (D3 fixed) | `test_adapter_messages_validate_retract_stale_predictions_and_none_are_rejected` |
| explicit memory identical to the APM's memory on the same stream | `test_explicit_memory_matches_the_apm_memory_on_the_same_stream` |
| predictor labels strictly after the decision; censored windows dropped | `test_predictor_labels_are_strictly_after_the_decision` |
| disjoint partitions that never reproduce v0.1 seeds | `test_partitions_are_disjoint_and_never_reproduce_old_seeds` |

## Measurements (pilot partition, not results about any hypothesis)

**Testbed behaviour under fixed actions** (40 two-episode lifetimes, population 40% careless):

| policy | goal completion | collisions / min | J per lifetime |
|---|---|---|---|
| always GO (= B0) | 0.47 | 2.79 | −4.6 |
| always SLOW | 0.27 | 0.09 | +3.3 |
| always YIELD | 0.007 | 0.09 | −0.1 |

97% of person collisions under GO involve careless pedestrians, who look identical to careful ones. So the testbed
has a genuine speed/safety trade-off, and knowing *who* is careless is potentially useful.

**Robot-side TTC estimator (the v0.1 formula, retracted when closing stops):** an alarm (p ≥ 0.5) on 36% of
control steps; only 8% of p ≥ 0.5 predictions are followed by a collision with that entity within 1 s (Brier 0.41).
The estimator is very poorly calibrated in this denser world. All systems receive it equally; it is the input
that drives the APM's threat appraisal.

**Throughput (this container, 1 core):** simulator 21,000 steps/s; one 10-episode lifetime takes 0.65 s (B0),
0.9 s (B1), 8.1 s (B2, dominated by the APM core at 20 Hz); predictor training 104,000 samples/s (CPU); peak RSS
0.7 GB. A full E2 (≈ 400 lifetimes × 8 systems) is ≈ 6 CPU-hours, about 1.5 h on 4 cores. **Cloud is not needed for
Stage I**; a GPU would not help these NumPy/Python loops.

## Deviations (recorded in `DEVIATIONS.md`)

Testbed constants were calibrated on E0 pilot runs of fixed-action policies, before any B1/B2 comparison: goal
deadline `4 s + d/0.35 m/s` (was `3 s + d/0.4`); the safety brake is a last resort at 0.05 m clearance in ±60° (was
0.15 m, ±70°), because at 0.15 m it removed almost all collisions and left nothing to compare; severe collision at
≥ 0.2 m/s (was 0.4, which never occurred because navigation already slows near people); careless pedestrians cut
across a *moving* robot's path, and everyone walks around a stopped robot (in the first version careless people
walked into a stationary robot, so no action could avoid them).

## Gate decision

**PASS.** Behavioural comparisons may proceed. Limitations: the pedestrian model is simple and hand-designed; the
testbed rewards caution near careless people by construction, so results are about this testbed, not about real
crowds.
