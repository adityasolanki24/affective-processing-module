# Project state

*The single authoritative account of what APM is and does today. Updated 2026-10-19, after the APM v1.0 release
programme (M; branch `release/apm-v1.0`, not merged; report: `experiments/M_release/REPORT.md`). Where M differs from
programme L, M governs.*

Everything below is **simulation only**. No hardware has been tested, and no human has evaluated the robots'
expression.

## Current architecture (one sentence)

1. The robot senses encounters with its **IMU and a rendered depth eye**.
2. An **explicit, revisable memory** turns them into a belief about each person: a learned latent type, with
   change-aware forgetting.
3. A **decision-theoretic mapping** (no VOI by default) with an **enforced risk cap** turns beliefs into a bounded
   **expressive intent**.
4. An **embodiment adapter** realises the intent on the robot's own validated controller.
5. **Two independent safety layers** can override it: APM's risk supervisor and the robot's physical supervisor.

One **integrated runtime** runs all of it, in Isaac Sim (NVIDIA Isaac Sim 5.1, PhysX) and in MuJoCo. Details:
[ARCHITECTURE.md](ARCHITECTURE.md).

<!-- STATE-RESULTS: filled after the M6 analysis -->
