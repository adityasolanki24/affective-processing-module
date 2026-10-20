# Programme I compute log

**Cap:** US$40, approved by the user on 2026-10-17. Machines: Google Cloud spot VMs of 128–180 vCPUs. Several may run
in parallel for independent experiments.

**Pricing basis:** cost estimates use the published us-central1 spot price for n2d. They are rough, so reconcile in
the billing console.

| date (UTC) | resource | duration | est. cost | purpose |
|---|---|---|---|---|
| 2026-10-17 | NAT gateway (us-central1) | up from 09:10 | ~US$0.01/h | package installs for private VMs |
| 2026-10-17 | apm-i-1 n2d-highcpu-128 spot (first attempt; SSH not possible, deleted) | ~12 min | ~US$0.2 | — |
| 2026-10-17 | apm-i-1 n2d-highcpu-128 spot (job runner) | from 09:23 | ~US$1.0/h | S10 pilot (1,200 lifetimes in 63 s), then T/V/X |
| 2026-10-17 | NAT gateway (us-east1) | up from 09:54 | ~US$0.01/h | package installs for apm-i-2 |
| 2026-10-17 | apm-i-2 n2d-highcpu-128 spot (job runner, us-east1) | from 09:54 | ~US$1.0/h | I2 confirmatory run, in parallel with I1 |

**Jobs on apm-i-1:**
- S10 pilot: 1,200 lifetimes in 63 s;
- T: 400 lifetimes in 23 s, plus the EM fit;
- V tuning: about 12,000 lifetimes in 9 min;
- D1 re-selection: 1,920 lifetimes in 97 s;
- I1-X: 12,000 lifetimes.

**Jobs on apm-i-2:** I2-X, 16,000 lifetimes.

## Session totals (2026-10-17, all resources deleted at 10:31 UTC; audit clean)

| resource | hours (approx.) |
|---|---|
| apm-i-1 (128 vCPU spot) | 0.1 (first attempt) + 1.15 |
| apm-i-2 (128 vCPU spot) | 0.5 |
| NAT us-central1 | 1.35 |
| NAT us-east1 | 0.5 |
| bucket prefix `apm-i/` | 118 MB, left in place: job archives and outputs; costs well under 1 cent per month |

**Estimated spend: about US$2** (spot n2d ~US$0.8–1.2 per VM-hour, plus NAT and egress cents). That is about 5% of the
US$40 cap. Reconcile in the billing console.

**Throughput:** about 1,100 lifetimes per minute per VM, roughly 18× the laptop. 46,000+ lifetimes in total for T, V and X.
