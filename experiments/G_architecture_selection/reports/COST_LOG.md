# Programme G cost log

Estimates from on-demand list prices (us-central1, per vCPU-hour incl. memory: n2d-highcpu 0.0299, n2-highcpu 0.0354,
n2d-standard 0.0422, n2-standard 0.0486, c2d-highcpu 0.0357, e2-highcpu 0.0247 US$), VM time = logged job window +
3 min boot/install. Actual billing is not readable from this login (see COMPUTE_PLAN.md); please reconcile in the
console. Programme F's earlier spend (≈ US$3–5, unconfirmed) is not included. All VMs self-deleted; NAT and bucket
objects were removed; final audit 2026-10-12 17:53 UTC: no VMs, disks, addresses, routers or `apm-g/` objects.

| job | machine | window (UTC) | VM min | US$ |
|---|---|---|---|---|
| G1 tuning + R5 ES | n2d-highcpu-96 | 13:49–14:06 | 20 | 0.94 |
| F E1 stage 1 (wheeled) | n2d-standard-32 | 13:52–14:28 | 39 | 0.87 |
| F E1 stage 1 (NERVA) | n2d-standard-32 | 13:58–14:05 | 10 | 0.23 |
| G1 test X+S | n2d-highcpu-96 | 14:09–14:18 | 12 | 0.55 |
| F E1 stage 2 (wheeled) | n2-standard-64 | 14:42–15:09 | 30 | 1.58 |
| F E1 stage 2 (NERVA rerun) | n2d-standard-32 | 14:50–14:58 | 11 | 0.25 |
| G2 tuning | n2d-highcpu-96 | 14:46–14:51 | 8 | 0.37 |
| G2 test | e2-highcpu-32 | 15:01–15:20 | 22 | 0.29 |
| G3 stage A | n2d-highcpu-96 | 15:07–15:14 | 10 | 0.46 |
| G3 stage B | c2d-highcpu-56 | 15:30–15:37 | 10 | 0.33 |
| G2b tuning | n2-highcpu-64 | 15:31–15:37 | 8 | 0.31 |
| G2b test | n2d-highcpu-48 | 15:43–15:51 | 11 | 0.26 |
| G4/G5 training A | n2d-highcpu-64 | 15:34–16:21 | 49 | 1.56 |
| G4/G5 training B | n2d-highcpu-64 | 15:39–16:24 | 47 | 1.50 |
| G4/G5 test | n2-highcpu-64 | 16:28–16:38 | 13 | 0.48 |
| G7 | n2d-highcpu-16 | 15:54–16:02 | 12 | 0.09 |
| G6 | n2d-highcpu-64 | 16:29–16:55 | 29 | 0.93 |
| G8 tuning | n2-highcpu-64 | 16:17–16:22 | 9 | 0.34 |
| G8 stage 1 | n2d-highcpu-64 | 16:31–16:33 | 5 | 0.16 |
| G8 stage 2 | n2d-highcpu-64 | 16:46–16:47 | 5 | 0.15 |
| G9 rules re-test | n2d-highcpu-64 | 17:16–17:29 | 16 | 0.52 |
| G9 models re-test | c2d-highcpu-56 | 17:16–17:40 | 27 | 0.91 |
| Cloud NAT (13:47–14:35, 14:40–17:52) | — | — | 240 | 0.18 |

| phase | cap (US$) | estimated (US$) |
|---|---|---|
| 1 (G1 + F's E1) | 6 | 4.42 |
| 2 (G2, G2b, G3) | 5 | 2.03 |
| 3 (G4/G5) | 14 | 3.54 |
| 4–5 (G6, G7, G8, G9) | 5 | 3.10 |
| **total incl. NAT** | **30** | **13.29** |
