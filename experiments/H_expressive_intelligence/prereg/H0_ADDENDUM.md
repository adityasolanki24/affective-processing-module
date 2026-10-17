# H0 addendum: the safe expressive-action interface (fixed before the confirmatory run, 2026-10-12)

This addendum governs H0's confirmatory run on partition **X**. It was written after:
- the audit and the NERVA gate reproductions (`reports/H0_AUDIT.md`);
- the exploratory pilots on partition P (`results/H0/pilot/`, `results/H0/pilot_dryrun/`; `RESEARCH_LOG.md`).

Nothing here is changed after X data exist. Runners: `scripts/h0_interface.py` (simulation) and
`scripts/h0_analyze.py` (admission and gate), both at the commit that adds this file.

## What the pilots changed (design decisions, before X)

1. **Primary controller = NERVA's validated neutral candidate** (`610b1c59…`).
   - E1′ fell in pilots: random switching at 0.4 s dwell with latency (1/3); body scanning in the nominal suite
     (2/4); head yaw −0.8 at rest (2/2).
   - Its pitch channel moves the torso by only about 1.5–2.5° while walking and not at all at rest.
   - E1′ is evaluated as a **secondary** configuration and is used in H1 only if it passes rule S below.
2. **Head offsets only while resting.**
   - Walking with head offsets cut forward speed to 0–76% (yaw ±0.4) and added turning, as NERVA's RQ1c found for
     the upstream policy.
   - The adapter therefore returns the head to neutral (|offset| ≤ 0.02 rad) before any locomotion command
     (`head_while_moving=False`).
   - The controller rate-limits head and pitch changes independently of the adapter.
3. **The head envelope must be validated at its corners.**
   - The pilot dry run granted a ±0.4 × ±0.8 box from axis points.
   - The untested corner (face down 0.4, yaw −0.8) made the robot fall at rest (2/2).
   - The X grid therefore tests every corner, plus yaw sweeps at every candidate pitch.
4. **Retreats are capped** (cumulative backward time ≤ 25% of forward time). Strong hesitation had stopped arrival
   within 45 s. Nominal episodes last 60 s.
5. **Monitor trip thresholds** beyond NERVA's tilt/angular-speed rule:
   - joint-limit margin < 0.02 rad (pilot nominal minimum 0.10–0.19 rad);
   - force saturation for ≥ 25 consecutive control steps (pilot maximum 7).
6. **Large pushes (0.6 / 0.9 m/s) exceed the controller's validated robustness.**
   - The gate validated 0.30 m/s. Pilots fell in 6/16 and 12/16 trials, with and without fallback, and had no
     discordant pairs.
   - The fallback is therefore specified as *stopping expression and returning to neutral*, not as fall
     prevention. Its criterion is non-worsening, below.

## Confirmatory suites (partition X)

All suites use: H0 seeds `sim_seed("H0", "X", index)`; NERVA's backlash scene; training observation noise; initial
joint noise 0.02 rad.

| suite | grid | seeds | duration | monitor |
|---|---|---|---|---|
| switching | random sequences of the 7 commands; dwell {0.4, 0.6, 1.0} s; neutral and E1′; latency off/on | 20 per cell (240 trials) | 30 s | shadow |
| head | **rest:** 0; head_pitch ±0.2, ±0.4; head_yaw ±0.4, ±0.8; all 16 corners (±0.2/±0.4 × ±0.4/±0.8); yaw sweeps (0.3 Hz) of amplitude 0.4 and 0.8 at head_pitch {−0.4, −0.2, 0, 0.2, 0.4}. **Walking forward:** head_pitch ±0.2, head_yaw ±0.4, and 0. Neutral and E1′. Offsets ramp in over 1 s | 5 per point (400 trials) | 20 s | shadow |
| pitch | E1′; forward, left, turn left, rest; e_pitch square wave (half-period 1 s) and ramp (1 /s) between −1 and +1; latency off/on | 10 per cell (160 trials) | 30 s | shadow |
| fallback | admitted interface; probe "slow_pausing"; push at t = 6 s of 0.3, 0.6, 0.9 m/s in 8 directions; monitor enforced vs shadow (paired seeds) | 5 per cell (240 trials) | 20 s | both |
| nominal | admitted interface (primary) and the secondary E1′ configuration; 5 probes (neutral, slow_pausing, indirect_scanning, averted_lean_back, lean_forward_close); target 1.0–1.4 m at bearing {0, ±0.6, 1.4} rad | 20 per probe per configuration (200 trials) | 60 s | enforced |

**Order:**
1. switching, head, pitch;
2. `h0_analyze.py admit` → `results/H0/X/admitted_interface.json`;
3. fallback and nominal on that interface (`--channels`);
4. `h0_analyze.py gate`.

The NERVA reproductions (`h0_analyze.py reproduce`) are checked separately; they use NERVA's own fixed seeds.

## Admission rules (fixed)

- **Command dwell (per policy).** The admitted minimum dwell d* is the smallest d in {0.4, 0.6, 1.0} such that
  every trial at d and at every larger tested dwell has no fall (tilt > 45° or non-finite state) and no monitor
  trip. If none qualifies, command sequencing is not admitted for that policy.
- **Head-at-rest box (per policy).** A box (|head_pitch| ≤ P, |head_yaw| ≤ Y), with P ∈ {0, 0.2, 0.4} and
  Y ∈ {0, 0.4, 0.8}, is admitted only if every grid point inside it passes in all 5 seeds. The grid points are:
  - axes;
  - all corners;
  - yaw sweeps with amplitude ≤ Y at every pitch within ±P.

  "Passes" means:
  - no fall and no trip;
  - NERVA's rest rules: smoothed horizontal speed RMS ≤ 0.02 m/s, yaw RMS ≤ 0.15 rad/s, displacement ≤ 0.10 m
    over 5–20 s.

  The admitted box maximises Y/0.8 + P/0.4 (ties: larger Y). Head rate 0.75 rad/s.
- **Head while walking** is admitted only if every walking point has a forward-speed ratio in [0.75, 1.33] of the
  zero-offset mean, yaw RMS ≤ 0.2 rad/s and no fall. Even if admitted, H1 keeps `head_while_moving=False`; the
  result is reported.
- **E1′ pitch channel:** admitted if the pitch suite has no fall and no trip (pitch rate 1 /s in the adapter).

## H0 gate (all required; primary configuration)

| criterion | definition |
|---|---|
| (a) reproduction | NERVA's neutral long motor gate reproduces for the neutral candidate and for E1′ at e = 0 (done: 160/160 trials byte-identical each), and E1′'s style evaluation reproduces (trials and summary identical to NERVA's record) |
| (b) sequencing | a command dwell d* ≤ 1.0 s is admitted for the neutral candidate |
| (c) channel families beyond mean speed | in the nominal suite, ≥ 2 of the families {temporal, spatial, posture} are realised. A channel is realised when the probe differs from the reference probe in the designed direction in ≥ 95% of paired seeds. The families are: **temporal:** slow_pausing has more pauses per metre than neutral; **spatial** (any of): indirect_scanning has a less straight path, lean_forward_close stops closer, averted_lean_back ends with a larger bearing to the target; **posture** (any, only if a head box is admitted): indirect_scanning has a larger face-yaw SD at rest, or lean_forward_close has a higher face elevation at rest than averted_lean_back |
| (d) monitor and fallback | (i) no fall in any enforced trial at the validated 0.30 m/s push; (ii) at 0.6 / 0.9 m/s, the number of seed pairs where only the enforced arm fell ≤ the number where only the shadow arm fell; (iii) in every enforced trial, every fallback step sends neutral rest with zero pitch and head offsets |
| (e) nominal adapter | no fall and no monitor trip in any primary nominal trial; for every probe, ≥ 95% of trials end within ±0.15 m of the probe's stopping distance; head offsets ≤ 0.02 rad whenever a locomotion command is active |

- **H0 passes** if (a)–(e) all hold. If it fails, H1 does not start; the blocker is reported and addressed first
  (MASTER §2).
- **Rule S (secondary E1′ configuration in H1).** E1′ is used in an additional H1 arm only if all of these hold:
  - its dwell is admitted;
  - its pitch channel is admitted;
  - its nominal suite has no fall and no trip;
  - every probe's arrival is ≥ 95%.

## Not claimed by H0

- Any expressive *quality* or human readability (H1/H2).
- Push robustness above 0.30 m/s.
- Hardware safety: NERVA has no hardware e-stop or torque/temperature limits.
