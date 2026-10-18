# Roadmap

Only the milestones still needed to demonstrate autonomous expressive development. History:
[RESEARCH_HISTORY.md](RESEARCH_HISTORY.md).

| # | milestone | done when | status |
|---|---|---|---|
| K1 | **Diagnose the recovery bottleneck** | it is measured what a distanced NERVA can observe about a changed entity: now, with a plausible extra sensor, and only through contact | in progress |
| K2 | **Safe information gathering** | the smallest mechanism that lets beliefs recover, using only signals the robot can sense, within the safety envelope | open |
| K3 | **Independent confirmation** | recovery (both directions) and the decision-theoretic mapping with task values frozen in advance, on fresh seeds, vs L0 and baselines | open |
| K4 | **Continuous demonstration** | two identical NERVA robots, different histories, one continuous lifetime each, change → revision → recovery, with recorded-trace video and plots | open |
| K5 | **Second embodiment** | the same memory, mapping and information gathering, unchanged, on a physically different robot with its own adapter and safety layer | open |
| M1 | NERVA motor-control milestone | stable under repeated sub-envelope pushes; richer validated motion primitives (speed range, turning dynamics) | needs NERVA controller work (separate) |
| H2 | human legibility | preregistered study run after an ethics determination | blocked on ethics |
| R1 | production runtime | the validated architecture behind AffectLink, replacing the PAD core | after K5 |
