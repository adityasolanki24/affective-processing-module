# Programme K compute log

**Cap:** US$40, approved by the user on 2026-10-18. Live billing is not readable (the Budget API is disabled), so
costs are estimates from VM hours: spot n2d-highcpu-128 at about US$1.0/h, NAT under US$0.02/h.

| date (UTC) | resource | from–to | est. cost | purpose |
|---|---|---|---|---|
| 2026-10-18 | apm-k-1 n2d-highcpu-128 spot, us-central1-a + NAT | 00:41–~02:10, **preempted** (K3-X lost, D2) | ≈ US$1.5 | K-T exploration (2 failed setups, then KT3), pilot, K3-X (lost) |
| 2026-10-18 | apm-k-2 n2d-highcpu-128 spot, us-east4-a + NAT | 01:26–02:16 | ≈ US$0.9 | K5-X, K3b, K4 controls (pre- and post-fix), K5-X2 |
| 2026-10-18 | apm-k-gpu g2-standard-8 (1× L4) spot | 01:32–04:20 | ≈ US$1.0 | humanoid walking policy (2 failed setups, then 2.1 h training) |
| 2026-10-18 | apm-k-3, apm-k-4 n2d-highcpu-128 **on-demand** (after the preemption) | 02:31–03:12 | ≈ US$4.8 | K3-X rerun, K3-X2 |
| 2026-10-18 | apm-k-5, apm-k-6 n2d-highcpu-128 on-demand | ≈ 04:28–04:55, 05:20–05:31 | ≈ US$2.1 | K6-T exploration, K6-X |
| 2026-10-18 | NATs (us-central1, us-east4) | on and off with the VMs | < US$0.3 | |

**Programme K total:** ≈ US$11 estimated, upper bound ≈ US$14, of the US$40 cap. Final audit (05:31 UTC): no VMs,
disks, routers or addresses. Bucket results and inputs (`apm-j/inputs/t1_model.mjb`) are kept.
| 2026-10-18 | laptop | – | – | K1 diagnosis, model fitting, wheeled exploration and pilot, analyses, demo rendering |
