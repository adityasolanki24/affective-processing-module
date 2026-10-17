# Literature and architecture review for programme G

Purpose: place every candidate mechanism in established work before it is implemented, so nothing standard is
presented as new, and so every candidate has a simpler baseline that could explain the same benefit. References are
primary papers (author, year, venue); implementations named are the authors' or widely used reference code.
Bibliographic details were written from the reviewer's knowledge and should be checked against the publisher before
the final report quotes them.

Columns used for each entry: **problem** it was designed for · **actually novel** in the literature · **relevance to
APM** · **cost** (compute/data) · **minimal defensible implementation here** · **simpler baseline** that could
explain the same benefit.

## 1. Risk-sensitive and constrained decision making

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| CVaR optimisation (Rockafellar & Uryasev 2000, *J. Risk*; Chow et al. 2015, NeurIPS "Risk-sensitive and robust decision-making: a CVaR optimization approach") | tail-risk objectives; novelty: convex formulation of CVaR, its MDP decomposition | high: collisions are rare tail events · cheap for a one-step rule | collision cost on an upper quantile of predicted P_col (ensemble spread) · a larger static λc (R1) |
| Constrained MDPs / Lagrangian methods (Altman 1999, *Constrained MDPs*; Tessler et al. 2019 ICLR RCPO; Ray et al. 2019 Safety Gym) | maximise return subject to a cost budget; novelty: learned Lagrange multipliers adapt the penalty | high: APM's modulated λc(t) is an *adaptive* penalty, which a Lagrange multiplier on a collision budget also is · cheap | λ ← λ + η(observed cost − budget) per episode · static tuned λ (R1) |
| Constrained Policy Optimization (Achiam et al. 2017, ICML) | trust-region policy updates with constraint satisfaction · novelty: per-update constraint guarantee | medium (only for learned policies, G4/G5) · RL-scale | Lagrangian PPO as the constrained baseline · PPO with a fixed cost penalty |
| Shielding / safety layers (Alshiekh et al. 2018, AAAI; Dalal et al. 2018 "Safe exploration in continuous action spaces") | keep a learner inside a safe set · novelty: formal/learned action correction | the testbed already has a fixed supervisor; it must stay identical for all systems · cheap | none new (supervisor fixed) · — |

## 2. Model-predictive control with learned consequence models

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| PETS (Chua et al. 2018, NeurIPS): probabilistic ensembles + trajectory sampling | data-efficient model-based RL · novelty: ensembles for epistemic + aleatoric uncertainty in MPC | high: B1 already is one-step MPC over 4 high-level actions with a 5-member ensemble · cheap for short horizons | A6: 2–3-step search over action sequences with the ensemble, pessimistic (mean + κ·sd) collision cost · B1's one-step argmax |
| Nagabandi et al. 2018 (ICRA), learned dynamics + MPC (+ model-free fine-tuning) | sample efficiency · novelty: MPC-initialised model-free learning | medium (G5 hybrid) · moderate | residual/fine-tune after MPC teacher · MPC alone |
| MuZero-style value-equivalent models (Schrittwieser et al. 2020, Nature) | planning with learned latent models in large domains | low: 4 actions, short horizons; massive compute · very high | not implemented · short-horizon search with the explicit predictor |

## 3. Recurrent policies and partial observability

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| DRQN (Hausknecht & Stone 2015, AAAI Fall Symp.) | POMDPs with deep RL · novelty: LSTM replaces frame stacking | high: identities, hazards and history are partly hidden · moderate | GRU actor-critic (A2; F's B4) · frame stacking / hand-made history features |
| R2D2 (Kapturowski et al. 2019, ICLR) | recurrent replay in distributed RL · novelty: stored-state + burn-in | low-medium · large compute | not needed at this scale · PPO with truncated BPTT |
| Recurrent PPO (Schulman et al. 2017 PPO; recurrent variants in standard libraries) | on-policy policy gradient with clipping | high (F's E4 trainer) · moderate | as in F (`apm_research.rl`) · BC from a strong teacher |

## 4. External memory and memory-augmented networks

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| Neural Turing Machine (Graves et al. 2014, arXiv) / DNC (Graves et al. 2016, Nature) | learning algorithms needing addressable memory · novelty: differentiable read/write | low-medium: APM's memory is keyed by identity; content addressing is unnecessary when keys are given · expensive, unstable training | per-identity GRU state (M6) · per-identity Beta counts (M0–M3) |
| MERLIN (Wayne et al. 2018, arXiv) / memory-based RL | long-horizon POMDPs with predictive memory | low at this scale · high | not implemented · explicit identity memory |
| Episodic control (Blundell et al. 2016, arXiv; Pritzel et al. 2017 NEC, ICML) | fast reuse of rare high-return experience · novelty: non-parametric value lookup | medium: rare collisions with specific people · cheap | nearest-neighbour outcome lookup keyed by identity/context · count-based identity memory |

## 5. Bayesian filtering, hidden-state models and change-point detection

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| Beta–Bernoulli / hierarchical Bayes (Gelman et al., *Bayesian Data Analysis*) | rates with partial pooling | high: per-identity collision probability with a type-level prior (M0 is already a fixed-strength version) · trivial | M3: hierarchical pooling with an estimated type prior · M0 |
| Exponential forgetting / discounted estimators (Garivier & Moulines 2011, ALT: discounted/sliding-window UCB) | non-stationary bandits · novelty: regret bounds for discounting and windows | high: E3's revision failure is exactly non-stationarity · trivial | M1 (discount γ per encounter), M2 (window W) · M0 |
| Bayesian online change-point detection (Adams & MacKay 2007, arXiv) | detect regime changes online · novelty: exact run-length posterior recursion | high: disposition switches are change points · cheap (truncated run length) | M4: BOCPD on per-identity Bernoulli outcomes with hazard h · M1 with a tuned discount |
| HMMs (Rabiner 1989, Proc. IEEE) / switching models | latent regime tracking · novelty: forward algorithm | high: a two-state (careful/careless) latent disposition with switching probability · trivial | M5: 2-state forward filter per identity · M1/M4 |
| Kalman filtering (Kalman 1960) | linear-Gaussian state tracking | medium: a continuous latent hazard per identity · trivial | random-walk on logit hazard (variant of M5) · M1 |

## 6. Online adaptation under selective feedback

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| Selective labels (Lakkaraju et al. 2017, KDD) / missing-not-at-random (Little & Rubin, *Statistical Analysis with Missing Data*) | outcomes observed only under some decisions | **central**: avoiding a person removes the evidence about them (F's E3 censoring) | exposure-gated evidence (count only encounters where a collision was physically possible) · counting all encounters (M0) |
| Partial monitoring / apple tasting (Helmbold, Littlestone & Long 2000, Inf. Comput.) | learning when feedback depends on the action taken · novelty: exploration requirements under one-sided feedback | high: "approach" reveals outcomes, "avoid" does not | bounded uncertainty-guided probing (safe SLOW approach with a probe budget) · probe-count-matched random probing |
| Thompson sampling / optimism (Thompson 1933; Russo et al. 2018 tutorial, FnT ML) | exploration–exploitation | medium: which uncertain identity is worth a safe probe · trivial | posterior-sample-gated probe · no probing |
| Off-policy evaluation (Precup et al. 2000; Dudík et al. 2011 doubly robust) | evaluate policies from logged data | low here (we can simulate) · — | not needed · closed-loop simulation |

## 7. Continual learning, replay and regularisation

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| EWC (Kirkpatrick et al. 2017, PNAS) | catastrophic forgetting · novelty: Fisher-weighted quadratic anchor | medium (G7) · cheap | EWC on the predictor during shadow updates · replay |
| Experience replay for continual learning (Rolnick et al. 2019, NeurIPS "CLEAR"; Chaudhry et al. 2019 tiny episodic memories) | forgetting · novelty: replay (+ behaviour cloning) is a strong baseline | high (G7) · cheap | reservoir replay + promotion gate (APM v0.1 already has replay + shadow learner) · frozen model + dynamic memory |
| Progressive nets (Rusu et al. 2016) / PackNet | no-forgetting by architecture growth | low (embedded constraints) · grows memory | not implemented · frozen model |

## 8. Latent world models and recurrent state-space models

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| World Models (Ha & Schmidhuber 2018, NeurIPS) | learn a compressed latent dynamics model, train controller in it | low: observations are already low-dimensional features · high | not implemented · explicit consequence predictor |
| PlaNet RSSM (Hafner et al. 2019, ICML) / Dreamer (Hafner et al. 2020, ICLR) | planning/learning in a learned latent space · novelty: deterministic + stochastic recurrent state | medium as an *upper bound*: G4's A6 is a much smaller relative · high GPU cost | short-horizon search with the explicit predictor (A6) · B1 |

## 9. Hierarchical and multi-timescale state

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| Clockwork RNN (Koutník et al. 2014, ICML) | multiple timescales in one RNN · novelty: modules updated at different clock rates | medium: fast "danger now" vs slow "this place/person is risky" | two leaky integrators / GRU with a slow leak (A5) · single GRU |
| Multiple-timescale RNN (Yamashita & Tani 2008, PLoS Comput Biol) | functional hierarchy from time constants | medium · cheap | fast + slow leaky units · EMA with two time constants (R5's φ already has 2 s and 30 s EMAs) |
| FeUdal networks (Vezhnevets et al. 2017, ICML) / options (Sutton, Precup & Singh 1999) | temporal abstraction | low: actions are already high level · high | not implemented · — |

## 10. Homeostatic RL and learned modulation

| method | problem · novelty | relevance · cost | minimal implementation · simpler baseline |
|---|---|---|---|
| Homeostatic RL (Keramati & Gutkin 2014, eLife) | reward as reduction of drive (deviation from setpoints) · novelty: links reward to internal-state regulation | medium: the honest formal frame for "internal state modulates cost" · cheap | cost weight as a function of an internal hazard state (G1 family) · static penalty |
| Neuromodulation-inspired meta-parameters (Doya 2002, Neural Networks "Metalearning and neuromodulation") | adaptive learning rate / discount / exploration / temperature | medium: PAD acts like an adaptive risk temperature · cheap | adaptive λc(t) = R2–R5 · static λc |
| Emotion-inspired RL (Moerland, Broekens & Jonker 2018, Machine Learning, survey) | review of "emotion" signals in RL | context: most proposals reduce to reward shaping, modulated exploration or risk sensitivity; claims of necessity are rarely tested against tuned baselines | used as the reason G1 exists · — |
| Appraisal-based computational emotion (Marsella & Gratch 2009 EMA, Cognitive Systems Research) | believable agent emotion | source of APM's appraisal variables; designed for believability, not control performance | R7 (appraisal without PAD) · R2 (raw threat EMA) |
| Residual policy learning (Silver et al. 2018 arXiv; Johannink et al. 2019 ICRA) | learn corrections around a hand-designed controller · novelty: start from a working controller | high: A4 is exactly this, with bounded utility corrections | bounded residual on action utilities · the controller alone |
| DAgger (Ross, Gordon & Bagnell 2011, AISTATS) | compounding error in imitation · novelty: on-policy relabelling by the expert | high for G5 (teacher = A0) · cheap in simulation | BC + DAgger iterations with the safety supervisor active · BC only |

## Implications for programme G

1. The useful part of B2 is plausibly an **adaptive collision-cost weight** — known in the literature as risk
   sensitivity / adaptive Lagrange multipliers / homeostatic modulation. G1 therefore tests PAD against filters and a
   Bayesian hazard estimate with equal tuning, not against an under-tuned B1.
2. E3's revision failure is the textbook non-stationary-estimation and **selective-labels** problem. Discounting,
   windows, change-point detection and exposure-gated evidence are the standard remedies (G2); a neural memory has to
   beat them, not cumulative counts.
3. A specialised recurrent regulator must beat both an ordinary GRU and a strong non-neural A0 (G4), and must not owe
   its result to a training method the GRU did not receive (G5).
4. Nothing in this list motivates an "emotional neural network" as a new invention. If a learned regulator wins, it
   will be described as what it is (e.g. a residual recurrent risk-weight controller).
