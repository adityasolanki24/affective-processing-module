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
