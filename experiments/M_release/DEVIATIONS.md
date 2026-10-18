# APM v1.0 deviations from PROTOCOL.md

Every departure from the frozen protocol, with its reason. Entries are appended; none is ever removed.

## D1 (2026-10-19): partition T exploration ran short (fitting only; no V or X data affected)

- **Planned (§2).** 48 RANDINTENT lifetimes per robot: T indices 0–11 (warehouse) and 12–23 (home), histories
  risky_B and safe.
- **What happened.** The jobs ran three Isaac processes per VM. Each process ran about 2.5× slower than alone,
  and the jobs hit their timeouts (100–120 min) with these lifetimes complete:
  - Nova Carter: 37 (18 warehouse, 19 home);
  - T1: 30 (15 + 15);
  - NERVA: recorded when its jobs finish.
- **What was done.**
  - The models are fitted on the completed lifetimes, by the same rule; no lifetime is selected or dropped.
  - The fitted files record their lifetime count (`configs/M5_*`).
  - Later stages use more VMs instead of more processes per VM, and progress heartbeats.
