# Programme G compute plan (2026-10-12) — requires the user's explicit spending approval

## Audit of the current situation

- **Prior authorisation:** US$10 total for the remaining programme-**F** compute (F `HANDOFF.md`). It does not cover
  programme G, and the F remainder is not treated as a G budget.
- **Actual F spend:** not visible from this login. The project's billing account is not readable with the current
  credentials and the Budget API is disabled, so no actual cost can be confirmed here (I have not changed either).
  F's estimate from list prices was ≈ US$3–5; please check Billing → Reports in the console.
- **Live resources:** none (no VMs, disks, addresses, routers or NAT; the `apm-f/` bucket prefix is deleted).
- **Quota (us-central1):** 200 vCPUs (N2D 200, N2 200, C2D 100); n2d stock-outs occur, so the launcher falls back
  across zones and machine families.

## Local vs cloud (measured)

| measurement | local laptop (i7-1355U, 15 W) | cloud (n2d, from F's runs) |
|---|---|---|
| one 10-episode lifetime, B2 family (APM at 20 Hz) | 9–11 s single process | ≈ 11 vCPU-s under full load |
| one 10-episode lifetime, simple regulators | ≈ 2 s | ≈ 2.5 vCPU-s |
| sustained parallel throughput | **≈ 3 effective cores** (G1 pilot: 480 lifetimes, 869 s, 6 workers) | 96 vCPUs per n2d-highcpu-96 |

Cloud is ≈ 30× faster per wall-clock hour than the laptop for this workload (CPU-bound simulation; no GPU needed —
the networks are ≈ 15k parameters and the simulator dominates; F's E4 measured this).

## Phase plan, estimates and caps (on-demand n2d ≈ US$2.9/h per 96 vCPUs; NAT ≈ US$0.05/h)

| phase | jobs | vCPU-hours | wall time | expected | worst case (hard caps) |
|---|---|---|---|---|---|
| **1a** G1 tuning (7 families × 48 configs × 30 lifetimes) + R5 ES | V/T only | ≈ 18 | ≈ 15 min on 96 vCPUs | US$0.8 | 45-min cap: US$2.2 |
| **1b** G1 test X (N = 280, 15 systems) + shift S (4 × 100) | after 1a is committed | ≈ 13 | ≈ 12 min | US$0.6 | 40-min cap: US$2.0 |
| **1c** F's E1 completion (data regeneration + ≈ 120 small models), in parallel with 1a on a second VM | independent | ≈ 20 | ≈ 25 min | US$1.0 | 60-min cap: US$3.0 |
| **2** G2 memory + G3 state (pilots local; tuning and tests in cloud) | after G1 | ≈ 60–100 | 2 × ≈ 30 min | US$2–3.5 | US$5 |
| **3** G4 tournament + G5 training methods (≤ 4 neural candidates × 5 seeds, ≈ 3M decisions each; 6 methods) | after G2/G3 | ≈ 250–300 | several 1-h VMs in parallel | US$8–10 | US$15 |
| **4** G6 robustness + G7 lifelong/cross-embodiment | after G4/G5 | ≈ 60 | ≈ 1 h | US$2 | US$3 |
| **5** G8 ablations | last | ≈ 30 | ≈ 30 min | US$1 | US$2 |
| **total** | | ≈ 450–550 | | **≈ US$16–19** | **US$30** |

Independent jobs run on separate VMs at the same time (within the 200-vCPU quota); dependent ones wait (1b after 1a;
2 after 1; 3 after 2; …).

## Safeguards (unchanged from F, extended)

Private VMs behind a temporary Cloud NAT (the organisation forbids external IPs); project/bucket/service account
from environment variables only, never committed; hard `--max-run-duration` with automatic deletion and self-deletion
at job end; outputs synced to a private bucket prefix every 2 minutes (resumable: a lost VM loses at most 2 minutes
and the job can be restarted from completed files); teardown and audit after every phase (no VMs, disks, addresses,
routers, NAT or bucket objects left); a running spend log in `reports/COST_LOG.md` from list prices, to be
reconciled with console billing.

## Requested approval

**US$30 total for programme G**, released phase by phase (1: US$6 cap, 2: US$5, 3: US$14, 4–5: US$5 — any unused
amount rolls forward), stopping and reporting before any phase would exceed its cap.
