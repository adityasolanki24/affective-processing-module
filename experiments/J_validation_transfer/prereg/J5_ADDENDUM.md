# J5 addendum: closed-loop expressive development on NERVA (committed before any J5-X data, 2026-10-18)

This addendum supplements `MASTER_PREREGISTRATION.md` §4.3. J5 runs ahead of J4's confirmatory test because J4's bank
was lost to a spot preemption. J5 does not depend on J4's result: it uses the J3-T-fitted, distance-conditioned NERVA
type model, and the J3 gate passed.

## Pilot findings that shaped the protocol (partition P; disclosed; data kept)

1. **The robot mistook its own expressive motion for being pushed.** After arrival, gaze turns, scanning and head
   motion gave high-acceleration evidence with zero contacts. Closed-loop beliefs saturated, and the history AUROC
   was 0.57.
   - Head motion alone raises the IMU peak to 1.1–1.9 (still: about 0.45).
   - **Fix: an efference-copy gate.** Frames during, and 0.6 s after, the robot's own commanded motion are excluded.
2. **With the gate alone, an expressive robot (L0 at moderate caution) had zero quiet frames.** Continuous scanning
   makes the robot unable to sense.
   - **Fix: a sensing pause.** After arrival the robot holds still for the 4 s observation window, then expression
     resumes.
   - Expression still sets stopping distance, approach tempo, hesitation and path.
   - Encounters with < 25 quiet frames are unobservable: no memory update.
3. **Pilot 3 (both fixes; n = 8):**
   - CL history AUROC 1.00;
   - belief ↔ stopping distance ρ 0.87, but the risky − safe stopping-distance gap is only 0.04 m: L0 compresses beliefs
     of 0.25–0.5;
   - max push CL − FROZEN −0.013 [−0.022, −0.004], contacts unchanged;
   - revision calm → pushy in 1 encounter; pushy → calm not reached;
   - 0 falls, 0 trips, 100% arrival; +8.2 s per approach vs neutral.

## Fixed values

**Type model:** `configs/J5_types_distance.json`:
- J3-T, conditioned on stopping-distance band (0.2 / 0.4 / 0.6 m; edges 0.3, 0.5);
- K = 2 by BIC;
- acc8 edges `configs/J4_edges.json`.

**λ = 0.02 per encounter**, selected on J5-V (CL, 15 lifetimes × 4 histories) by self-supervised predictive
log-likelihood (`results/J5/J5_V_lambda.json`; the objective is nearly flat: −2.044 to −2.035).

**World and protocol:** `nerva_world.closed_loop_encounter` as committed:
- the gate (0.6 s hold) and the sensing pause;
- H's admitted interface (`configs/H_admitted_interface.json`, a frozen copy);
- H's `SafetyMonitor` and neutral fallback.

**Lifetimes:**
- 30 encounters with A1, B1 and C1; switch at 15;
- histories risky_B, change_B, late_B, safe;
- systems CL, FROZEN, NOMEM, NOFEEDBACK, RANDHIST, H_MEM.

**Test:** J5-X indices 0–39: **960 lifetimes**, 28,800 closed-loop NERVA encounters.

## Criteria (`j5_analyze.py`; LATE = 10; calibration from J5-V CL)

| question | criterion |
|---|---|
| Q1 outcomes → beliefs | CL history AUROC (risky_B vs safe, B1 belief at its last encounter) ≥ 0.80, and belief-effect 95% CI lower bound > 0 |
| Q2 beliefs → expression | CL Spearman(belief, stopping distance) ≥ 0.5, and stopping-distance effect (risky_B − safe, after LATE) CI lower bound > 0 |
| Q3 expression → outcomes | **primary:** CL − FROZEN mean max push per B1 encounter (risky_B, after LATE), 95% CI upper bound < 0. Secondary: contacts, detections |
| Q4 revision | CL Kaplan–Meier median ≤ 4 B1 encounters (horizon 8) in **both** late_B and change_B |
| Q5 recovery from excess caution | change_B: CL stopping distance back within 0.05 m of the safe-history mean, median ≤ 4 B1 encounters. **Testable only if** the Q2 effect ≥ 0.05 m; otherwise "not testable" |
| Q6 safety and task | 0 falls in every system; safety trips ≤ 1% of encounters; CL arrival rate ≥ 0.99. The CL − FROZEN arrival-time difference is reported |

**The full loop is demonstrated** only if Q1–Q4 and Q6 pass, and Q5 passes. Per the master preregistration, a history
effect alone is insufficient.

**Ablations, reported:**
- NOMEM: no history effect expected;
- RANDHIST: specificity, measured as the B1 − A1 belief difference within risky_B, CL vs RANDHIST;
- NOFEEDBACK: revision;
- H_MEM: H's memory on the robot-observable detector.
