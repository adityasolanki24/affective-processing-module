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
