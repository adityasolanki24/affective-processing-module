# Deviations from MASTER_PREREGISTRATION.md

Every change to the preregistered plan is recorded here, dated, with its reason and whether it happened before or
after the affected data were seen.

## 2026-10-11 — testbed calibration during E0 (before any B1/B2 comparison data)

Made on E0 pilot runs of fixed-action policies (always GO / SLOW / YIELD / DETOUR), before any system comparison:

| item | preregistered | implemented | reason |
|---|---|---|---|
| goal deadline | 3 s + d / 0.4 m/s | 4 s + d / 0.35 m/s | at 0.4 m/s even an unobstructed robot missed many deadlines among people |
| safety brake | 0.15 m clearance, ±70° | 0.05 m clearance, ±60° (last resort) | at 0.15 m the brake prevented ≈ all collisions under every policy, leaving no behavioural question |
| severe collision | robot speed ≥ 0.4 m/s | ≥ 0.2 m/s | navigation already slows to ≤ 0.24 m/s near people, so 0.4 never occurred |
| careless pedestrians | "keep going and drift toward the robot when close" | cut across a *moving* robot's path (aim 0.6 m ahead of it); all pedestrians walk around a *stopped* robot | the first version walked into a stationary robot, so collisions were unavoidable by any action |
| navigation speed near people | v0.1 rule (nearest entity) | obstacle and person proximity factors separate (person factor ≥ 0.3 within 0.4 m) | people were slowing the robot to a crawl everywhere |

None of these change a hypothesis, metric, criterion or seed.
