# Programme K compute log

**Cap:** US$40, approved by the user on 2026-10-18. Live billing is not readable (the Budget API is disabled), so
costs are estimates from VM hours: spot n2d-highcpu-128 at about US$1.0/h, NAT under US$0.02/h.

| date (UTC) | resource | from–to | est. cost | purpose |
|---|---|---|---|---|
| 2026-10-18 | apm-k-1 n2d-highcpu-128 spot, us-central1-a (4 h cap) + NAT | 00:41– | ~US$1.0/h | K-T exploration (2 failed setups, then KT3), pilot, K3-X |
| 2026-10-18 | apm-k-2 n2d-highcpu-128 spot, us-east4-a (3 h cap) + NAT | 01:26– | ~US$1.0/h | K5-X (wheeled), K3b (tasks) |
| 2026-10-18 | laptop | – | – | K1 diagnosis, model fitting, wheeled exploration and pilot, analyses, demo rendering |
