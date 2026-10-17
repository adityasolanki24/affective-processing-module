# J3 addendum: NERVA evidence world fixed from pilots (committed before any J3-T/V/X data, 2026-10-18)

This addendum supplements `MASTER_PREREGISTRATION.md` §4.1. Nothing in it relaxes a criterion.

## What the pilots showed (partition P; reported in `RESEARCH_LOG.md`)

**Pilot 1** (1,350 encounters; types calm, gentle and forceful; peak features; 6-encounter EM units): **gate failed.**
- Most contacts were below the IMU's sensing range (median push 0.023 m/s).
- The a-priori gyro detector (0.35 rad/s) never fired.
- EM found K = 1.

**Pilot 2** (1,800 encounters; disclosed changes):
- evidence v2: self-calibrated detector, filtered impact energy, settle baseline;
- a 4th type, *pushy* (frequent moderate pushes, within the 0.10 m/s envelope);
- 12-encounter EM units.

Results:

| coding | mean pairwise KL | calm vs pushy at 0.2 / 0.4 m |
|---|---|---|
| graded acceleration | **0.50 nats** | 1.74 / 1.19 |
| self-calibrated detector | 0.13 | |
| a-priori detector | 0.003 | |
| falls | 0 | |

- EM on acceleration bins found K = 3 without labels: {calm + gentle} (indistinguishable: below the sensing range),
  forceful, and pushy.
- At 0.6 m, information collapses because most entities cannot reach the robot.

## Fixed for J3-T, J4 and J5

**World:**
- entity types calm, gentle, forceful, pushy (`nerva_world.TYPES`, as committed);
- push cap 0.10 m/s;
- standing encounters: 2 s settle, 4 s dwell;
- closed-loop encounters: evidence window from 1 s after arrival, for 4 s.

**Distances (J3-T):** 0.2, 0.4 and 0.6 m.

**Primary coding: `acc8`.**
- 8 bins of peak horizontal acceleration deviation (`Evidence.peak_acc`).
- Bin edges are the pooled quantiles of J3-T, computed without labels and frozen in the J4 addendum.
- `energy8` and `detected` are reported as secondary.

**J3 gate, evaluated on J3-T** (`j3_info.py`; units of 12 encounters of one type at one distance; EM K = 1..4 by BIC):
- acc8 mean pairwise KL ≥ 5 × the a-priori binary detector's (as preregistered), and > falls;
- also reported: the ratio to the self-calibrated detector (stricter);
- EM selects K ≥ 2.

## J3-T generation

- 4 types × 3 distances × 500 encounters = 6,000 standing encounters.
- Seeds: J3/T indices 0–2,999 (calm, gentle) and 10,000–12,999 (forceful, pushy), on two VMs.

## J4 bank (generated now, used only after the J4 addendum)

- 4 types × 0.4 m × 1,000 standing encounters.
- Seeds: J3/X indices 0–3,999.
- J4 lifetimes draw from this bank, by type, at the neutral stopping distance.
