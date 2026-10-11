# Deviations from MASTER_PREREGISTRATION.md (programme G)

Every change to the preregistered plan is recorded here, dated, with its reason and whether it happened before or
after the affected data were seen.

## 2026-10-12 — G1 pilot started minutes before this preregistration was committed

The G1 pilot (partition P, 40 lifetimes, pilot specs `configs/G1_pilot_specs.json`) was started locally at 13:16 UTC,
about 10 minutes before the commit that freezes `MASTER_PREREGISTRATION.md`. No pilot output had been inspected
when the preregistration was committed (the runner had written only partial base lifetimes). Pilot data are used only
for debugging, timing and the sizing rule of §6.

## 2026-10-12 — code added after the G1 addendum commit (behaviour-neutral for G1)

`scripts/g_common.py` gained an optional `memory` field (G2 belief memories) and `src/apm_select/` gained
`memory.py`, `policies.py` and `scenarios.py` for G2. G1 specs never contain `memory`, so G1 behaviour is unchanged
(the G1 integrity tests still pass, including exact equality with programme F's B1 and B2). The G1 test runs from a
later commit than the addendum's; its run metadata records that commit and all source hashes.

## 2026-10-12 — G2 selection written from recorded validation data

On the tuning VM, `g2_tune.py` finished every simulation (stage A table and stage B validation lifetimes were
written) but crashed when writing `configs/G2_selected.json` (a NumPy boolean in the verdict). The cast was fixed and
the identical selection rule was applied locally to the recorded outputs (`g2_tune.py --select-only`); no simulation
was repeated. Result: no memory meets (i)–(iii) on validation (all fail S1 adaptation), so no M\* is selected; as
preregistered, the X run evaluates all memories (secondary only).

## 2026-10-12 — validity problem found after confirmatory runs: unavoidable spawn collisions (testbed defect D8)

While checking an anomaly in G6 (identical collision counts for all systems under `speed_limit_0.6`), collision
logs showed that the frozen testbed places the robot after the pedestrians and checks its clearance only against
obstacles (`WheeledCtrlWorld._free_point`), so the robot can **spawn overlapping a person**. The contact is reported
at the first step (t = 0.05 s), is never severe (the robot is stationary), and is identical for every policy under
common random numbers. It is common: 67–97% of all collisions of the rule-based systems in a 40-lifetime check (G0-P)
and 51–78% in programme F's E2 test (183 spawn collisions per system in E2-X).

Consequences: severe-collision results and all non-collision metrics are unaffected. Collision-*rate* comparisons
are diluted toward "no difference" by a constant, policy-independent component — conservative for superiority tests,
**anti-conservative for non-inferiority tests** (G1 P2, G8 removals, G6 eligibility, G5's GRU test). As required by
§7, the confirmatory results are kept and reported unchanged; nothing is silently re-run. A separate, preregistered
validity re-test (G9, `prereg/G9_ADDENDUM.md`) repeats the primary families with the same frozen selections and
models on fresh lifetimes using **avoidable collisions** (contacts after the first step); runners now record
`spawn_collisions` and `avoidable_collisions` (`apm_select.instrument.collision_split`, tested). The testbed itself is
not modified (F and G depend on it); a fixed testbed is recommended for any future programme.
