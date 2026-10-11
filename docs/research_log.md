# Research log

Newest first. Measurements only from runs actually executed; hypotheses labelled.

## 2026-10-11: Milestone A (repository, AffectLink v1, APM-0 core)
- Implemented the schema, codec, validation and core with 25 tests (synthetic inputs).
- The standalone synthetic scenario exposed four implementation defects, all fixed before the milestone closed:
  1. phasic gain large enough to drive valence to the −1 bound after one collision;
  2. event streams wrongly marked stale;
  3. strain from comfortable margins;
  4. memory expectations with no effect on appraisal.
  Memory now raises internal predictions for tracked entities, which are never learned as outcomes.
- [design] All appraisal and affect constants are hand-set, transparent baseline values, not fitted to any data.
