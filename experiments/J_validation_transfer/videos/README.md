# Paired NERVA experience videos (J6-X)

Made by `scripts/j_videos.py` from J6-X lifetimes; `manifest.json` has every clip's numbers.

**Pairing.**
- The same robot and the same seeds in both clips. Only B1's physical behaviour differs: pushy in `risky_B`, calm in
  `safe`.
- Each lifetime was re-simulated, and every encounter row matched the J6-X log exactly (`matches_J6X_log`).
- Rendering is visual only, and was verified not to change the simulation.

**Selection.** For each system, the lifetime whose late-B1 stopping-distance difference is the median across the 40
J6-X pairs: CL index 64, LEARNED_V index 65. These are typical pairs, not the most striking.

| file (left: risky_B, right: safe) | what it shows |
|---|---|
| `CL_64_first_pair.mp4` | **Experience.** First meeting with B1, belief 0.35 in both, identical intent. The pushy entity lands 2 pushes; the calm one none |
| `CL_64_last_pair.mp4` | **Subsequent expression with H's fixed L0.** Belief 0.49 vs 0.25: stops at 0.51 m vs 0.44 m, a small difference |
| `LEARNED_V_65_first_pair.mp4` | **Experience.** First meeting, belief 0.35 in both, both stop at about 0.40 m; 1 push vs none |
| `LEARNED_V_65_last_pair.mp4` | **Subsequent expression with the learned mapping.** Belief 0.42 vs 0.25: stops at **0.60 m vs 0.31 m**, and arrives in 4.2 s vs 8.7 s |

**Visual coding.**
- The grey column is the entity.
- An arm appears on each lunge, with length equal to the lunge's reach: red if it reached the robot (a push), grey if it
  fell short.
- Captions give the history, encounter, the entity's simulator type, the robot's belief, the intended proximity, the
  distance and contacts.
- The type label is for the viewer only. It is never an input to the robot, and is not an emotion.

The single-encounter `*_risky_B_*.mp4` and `*_safe_*.mp4` files are the panels of the pairs.
