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
