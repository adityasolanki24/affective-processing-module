# Programme J compute log

**Cap:** US$40, approved by the user on 2026-10-17. Programme I's separate authorization is not used.

**Machines:** spot n2d-highcpu-128 VMs. Several may run in parallel for independent experiments. Every session ends
with `jcloud.py audit` clean.

| date (UTC) | resource | duration | est. cost | purpose |
|---|---|---|---|---|
| 2026-10-17 | NAT us-central1 | from 11:12 | ~US$0.01/h | package installs for apm-j-1 |
| 2026-10-17 | apm-j-1 n2d-highcpu-128 spot (us-central1-a; 4 h cap) | from 11:12 | ~US$1.0/h | J1-V, J1-X, J2-T, J3 pilot 1 (+ NERVA env), J2-V, J2-X group A, J3 pilot 2 |
| 2026-10-17 | NAT us-east1, us-east4 | from 12:25 | ~US$0.02/h | apm-j-2, apm-j-3 |
| 2026-10-17 | apm-j-2 n2d-highcpu-96 spot (us-east1-b; the 128-core type was stocked out; 3 h cap) | from 12:28 | ~US$0.75/h | J2-X group B |
| 2026-10-17 | apm-j-3 n2d-highcpu-128 spot (us-east4-a; 3 h cap) | from 12:26 | ~US$1.0/h | J2-X group C + policy group, J2 analysis |
| 2026-10-18 | apm-j-2 n2d-highcpu-96 spot (us-east1-b), second instance | 13:08–about 13:40 | ~US$0.4 | J4 bank. **Preempted by Google (spot) and auto-deleted before finishing; output lost; re-queued on apm-j-1** |
| 2026-10-17 | apm-j-1 (continued) | to about 14:45 (its 4 h cap; self-deleted) | ≈ US$3.5 total | J3-T a, J5-V, J5-V2, J5 pilot 3, J4 bank (J4bank2) |
| 2026-10-17 | apm-j-3 (continued) | to 14:57 (deleted) | ≈ US$2.5 total | J3-T b, J5 pilots, J5-V3, J5-X, J6-T, J6-X |
| 2026-10-17 | NATs and routers us-central1, us-east1, us-east4 | deleted | < US$0.2 | |

## Programme J total

- **Estimated spend:** about **US$7–10** of the US$40 cap.
  - VM-hours at about US$1.0/h for spot 128-vCPU (96-vCPU about US$0.75/h);
  - apm-j-2 is counted at its 3 h cap as an upper bound.
  - The billing console is authoritative; these are estimates from creation and deletion times.
- **Laptop:** J4 lifetimes, λ tuning and analyses; J6 time-model and GRU fitting; figures; paired videos.
- **Final audit (14:58 UTC):** VMs none, disks none, routers none, addresses none. Bucket results under `apm-j/` are kept.
