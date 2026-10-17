# H0 report: the safe expressive-action interface (2026-10-12)

## Verdict: **H0 gate FAILS** (criteria d and e)

**Setup:**
- Confirmatory run on partition X, governed by `prereg/H0_ADDENDUM.md`;
- commit `2ecad67`, locks passed (tracked tree clean, NERVA `726c8c6` clean);
- 1,240 simulated trials, local CPU;
- results in `results/H0/X/`; the gate is `results/H0/X/gate.json`.

As preregistered, **H1 does not start on this interface.** The two blockers were diagnosed (below) and are addressed
by a new addendum, `prereg/H0b_ADDENDUM.md`. Its re-test runs on fresh seeds; this failure stays on record.

| criterion | result |
|---|---|
| (a) reproduction of NERVA's stable behaviour | **pass.** Neutral long motor gate, neutral candidate: 160/160 trial records byte-identical to NERVA's Linux run, gate passes. Same for E1′ at e = 0: 160/160, passes. E1′ style evaluation: 378/378 trials identical, and the same verdict (pitch controllable but small; crouch ignored; fails its own criteria 3, 4, 5, 8). Summaries identical except wall-clock time. |
| (b) command sequencing admitted | **pass.** Neutral candidate: 0 falls and 0 trips in 120 random-switching trials at dwell 0.4, 0.6 and 1.0 s (with and without latency), so d* = 0.4 s. E1′: 2/40 falls at 0.4 s, so d* = 0.6 s. |
| (c) ≥ 2 channel families beyond mean speed | **pass.** Temporal (pauses), spatial (path, proximity, orientation) and posture (head scanning, head elevation at rest) were all realised in 100% of paired seeds. These spatial features were computed before the gait-sway fix (see Deviations); the paired comparisons are relative and the conclusion holds. |
| (d) monitor and fallback | **fail.** 4/40 enforced episodes fell at the 0.30 m/s push. (ii) Non-worsening holds: discordant pairs 0 enforced-only vs 1 shadow-only. (iii) Every fallback step was neutral. |
| (e) nominal adapter | **fail.** 0 falls and 0 trips in 100 primary trials; head offsets never exceeded 0.005 rad while moving. But "slow_pausing" arrived within ±0.15 m of its stopping distance in only 65% of trials (criterion ≥ 95%); every other probe 100%. |

## What H0 established

### Admitted channels (`results/H0/X/admitted_interface.json`)

- **Command sequencing** on the validated neutral controller, minimum dwell 0.4 s. This carries:
  - timing: pauses, stop-go, retreat;
  - path shape;
  - stopping distance;
  - orientation.
- **Head offsets at rest only**, inside the box |head_pitch| ≤ 0.4 rad, |head_yaw| ≤ 0.4 rad, at ≤ 0.75 rad/s.
  - Failing points: face-down 0.4 with yaw −0.8 (falls), and 0.8 rad yaw sweeps at face-down pitch.
  - **Untested corners must never be assumed safe.** The pilot dry run fell exactly there.
- **Head offsets while walking: not admitted.** Speed ratios 0.12–0.58 at ±0.4 yaw or face-up 0.2. Face-down 0.2
  alone was harmless (1.19).
- **E1′ (secondary):**
  - pitch changes, both square waves and ramps, were safe in 160/160 trials;
  - the torso moves by only 1.0–1.9° while walking and ≈ 0° at rest (in line with NERVA's record);
  - head box ±0.2 × ±0.4;
  - dwell 0.6 s;
  - nominal suite: 0 falls, 100% arrival.

### Measured probe effects (primary, 20 seeds each)

| probe | arrival | progress (m/s) | pauses / m | final distance (m) | final bearing (rad) | face-yaw SD at rest | face elevation at rest |
|---|---|---|---|---|---|---|---|
| neutral | 1.00 | 0.069 | 0.0 | 0.40 | 0.23 | 0.01 | 0.45 |
| slow_pausing | **0.65** | 0.012 | 32.5 | 0.63 | 0.18 | 0.04 | 0.42 |
| indirect_scanning | 1.00 | 0.047 | 2.0 | 0.29 | 0.14 | 0.34 | 0.39 |
| averted_lean_back | 1.00 | 0.026 | 14.8 | 0.41 | 0.98 | 0.08 | −0.06 |
| lean_forward_close | 1.00 | 0.069 | 0.0 | 0.19 | 0.14 | 0.01 | 0.91 |

## Diagnosis of the failures

### (d) Pushes

All 8 falls at 0.30 m/s, 4 enforced and 4 shadow on the same seeds, share three properties:
- the push came from behind (180° and 225°);
- the robot was **standing (rest command) during an expressive pause**;
- the same push on the same seed during plain neutral walking caused no fall (10/10).

**A standing-push sweep** (`results/H0b/pilot/rest_push.json`, partition P, 3 seeds × 8 directions) shows the
cause is a property of the controller, not of the expression layer:
- **Neutral candidate:** standing still, it falls from backward pushes (135–225°) of ≥ 0.20 m/s (0.20: 3/24;
  0.25: 8/24; 0.30: 9/24) and survives ≤ 0.15 m/s in all directions.
- **E1′:** falls from 0.15 m/s.
- NERVA's long motor gate pushes only during steady walking, so **push robustness at rest was never validated**.
- Expressive pauses increase time spent standing, and so increase exposure.
- The safety fallback cannot help: its action is "stand at rest", which is the vulnerable state. Fallback and shadow
  fell on the same seeds.

### (e) Arrival

- The slow_pausing probe (tempo 0.4, hesitation 0.8) progresses at 0.009–0.02 m/s.
- The 1 s forward bursts at tempo 0.4 barely start the gait.
- With pauses and retreats, 1.4 m targets are not reached within 60 s.
- **The adapter has no guarantee of minimum progress.** That is an interface design flaw: expression must not
  compromise task completion.

## Consequences

1. **Controller safety finding (for NERVA).** The validated neutral controller is not robust to 0.20 m/s pushes
   from behind while standing.
   - This holds with or without APM.
   - It affects any use of `rest`, including the end of every approach.
   - It is reported, not fixed: NERVA's controller is not modified in programme H.
2. **H0b** (`prereg/H0b_ADDENDUM.md`), new seeds:
   - characterise standing-push robustness on X;
   - specify the push requirement as the magnitude validated for *both* walking and standing;
   - add a minimum-progress guarantee and efficient forward bursts to the adapter;
   - re-test the nominal and fallback suites.
   The admitted dwell, head box and pitch channel from this run are kept: those suites do not go through the adapter
   and are unaffected by the changes.

## Deviations (also in `DEVIATIONS.md`)

- **Reproduction comparison.** `h0_analyze.py reproduce` compared whole summary files, including NERVA's wall-clock
  `wall_seconds`. That is a timing, not a result. The comparison now excludes it.
  - Before the fix, the check reported "not identical", solely because of this field.
  - The fix was made before the gate was computed.
- **Spatial features and gait sway.** The spatial features in `features.py` used raw base positions, including
  lateral gait sway, so straight walking scored a straightness of ≈ 0.37.
  - Fixed for all later experiments by averaging positions over one gait period (27 steps), with a test.
  - H0's criterion (c) compares probes with each other and would not change direction. The stored H0 features are
    the uncorrected ones.
