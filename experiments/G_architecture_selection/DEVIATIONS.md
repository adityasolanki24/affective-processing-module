# Deviations from MASTER_PREREGISTRATION.md (programme G)

Every change to the preregistered plan is recorded here, dated, with its reason and whether it happened before or
after the affected data were seen.

## 2026-10-12 — G1 pilot started minutes before this preregistration was committed

The G1 pilot (partition P, 40 lifetimes, pilot specs `configs/G1_pilot_specs.json`) was started locally at 13:16 UTC,
about 10 minutes before the commit that freezes `MASTER_PREREGISTRATION.md`. No pilot output had been inspected
when the preregistration was committed (the runner had written only partial base lifetimes). Pilot data are used only
for debugging, timing and the sizing rule of §6.
