# Literature review: developmental expressive intelligence (programme H, 2026-10-12)

## Scope and sources

- **Primary sources:** peer-reviewed papers. Preprints are marked **[preprint]**.
- **How details were checked:**
  - Bibliographic details of the robotics/HRI items most relevant to H were checked by web search on 2026-10-12:
    Grandia et al.; Sripathy et al.; Suguitan et al.; Knight & Simmons; Saerbeck & Bartneck; Hiolle et al.;
    EmoLo; EMoG.
  - Everything else comes from the author's prior knowledge of the published literature. Details are kept to what
    is well established. Where a detail could not be confirmed, the entry says so.
- **No claim of novelty is made anywhere in programme H** until the competing approaches below have been
  evaluated.

### The seven questions asked of every approach

| code | question |
|---|---|
| **Dem** | What was actually demonstrated? |
| **Hum** | Did human observers distinguish the expressions? |
| **S/L** | Was the motion scripted (S) or learned (L)? |
| **Dev** | Did internal states develop from experience? |
| **Safe** | Were safety and task competence preserved? |
| **Gen** | Did the results generalise to new contexts or bodies? |
| **Base** | Which simpler baseline could explain the result? |

---

## 1. Expressive robot motion and affective HRI

| work | Dem | Hum | S/L | Dev | Safe | Gen | Base |
|---|---|---|---|---|---|---|---|
| Saerbeck & Bartneck 2010 (HRI), affect from robot motion | acceleration and path curvature of simple robots (Roomba, iCat) shift perceived affect | yes: acceleration predicts perceived arousal; valence partly from acceleration × curvature; embodiment had no effect | S (parametric) | no | n/a | two bodies | speed/acceleration alone; a LEGO-robot follow-up found few free-text remarks matching the questionnaire terms |
| Knight & Simmons 2014 (RO-MAN), Laban Effort features for x, y, θ | Effort features (Time, Weight, Space, Flow) computed from mobile-robot paths; a classifier matched intended "manner" 41.7% vs 41.2% for human labellers (MTurk) | partly: human agreement itself modest | S (expert-made paths) | no | n/a | low-DoF bodies | path speed and curvature statistics |
| Knight & Simmons 2016 (IROS), path shape ("swagger"); 2016 (ICRA), Laban head motions | path shape and orientation change perceived confidence and focus; head-motion Effort conveys state | yes (user studies) | S | no | n/a | Nao head, Keepon | orientation and speed |
| Takayama, Dooley & Ju 2011 (HRI), animation principles | forethought and reaction motions make robot intentions more readable | yes (video studies) | S | no | n/a | one PR2 | — |
| Dragan, Lee & Srinivasa 2013 (HRI), legibility vs predictability | formal legibility: motion that lets observers infer the goal early | yes | L (trajectory optimisation) | no | yes (task kept) | manipulation goals | — |
| Zhou & Dragan 2018 (IROS), cost functions for motion style | style as a learned cost on trajectories, from user labels | yes (user labels) | L | no | yes | — | per-style hand-tuned costs |
| Sripathy et al. 2022 (arXiv 2203.02091; MIT copy labelled IROS 2022), span functional expressive motion | one VAD-space cost model learned from user labels; generates task trajectories at any VAD target; Cassie biped and a vacuum robot | yes (user study) | L | no | task constraints kept | any VAD point; new emotion words via language | per-emotion costs (beaten in the paper) |
| Venture & Kulić 2019 (ACM THRI), survey of expressive motion generation and evaluation | evaluation practice: forced choice, ratings, Laban features | — | — | — | — | — | — |
| Bethel & Murphy 2008 (IEEE SMC-C), survey for appearance-constrained robots | body movement, posture, orientation, proximity and colour as affect channels where a robot has no face | — | — | — | — | — | — |
| Beck, Cañamero et al. 2012 (ACM TiiS), emotional body language on Nao | static key poses: head position strongly affects valence and arousal ratings | yes | S | no | n/a | Nao | posture alone |
| Suguitan, Gomez & Hoffman 2020 (HRI), MoveAE | classifying VAE edits a base affective movement toward a target class by latent arithmetic; modified movements roughly comparable to originals in recognisability (per secondary sources) | yes (user evaluation) | L (from human-designed/emotional data) | no | n/a | Blossom robot | the source animations |
| Desai et al. 2019 (CHI), Geppetto | crowdsourced semantic editing of expressive robot motion parameters | yes | S + learned perceptual model | no | n/a | — | — |
| Hoffman & Ju 2014 (JHRI), designing robots with movement in mind | design methodology: movement-first expressive robots | — | — | — | — | — | — |
| ELEGNT 2025 (arXiv 2501.12493) [preprint, title only verified] | expressive and functional movement for a non-anthropomorphic lamp robot | not checked | not checked | — | — | — | — |

**Implications for H:**
- Low-DoF robots can be read through speed, acceleration, path shape, orientation, head posture and proximity.
  These are exactly the channels NERVA's validated controller exposes.
- But observer agreement is often modest, and **speed/acceleration alone carries much of the effect**. This is why
  H1 requires distinctness with speed features removed, and matched-speed comparisons.

## 2. Emotion perception from biological and robot motion

- **Johansson 1973 (point-light displays):** human motion is perceived from sparse kinematics.
- **Montepare, Goldstein & Clausen 1987 (J Nonverbal Behav):** emotions identified from gait above chance.
- **Pollick et al. 2001 (Cognition):** perceived affect from arm movements is organised mainly by an arousal axis
  tied to velocity/kinematics.
- **Atkinson et al. 2004 (Perception):** emotion recognised from dynamic and static, point-light and full-light
  body displays. Dynamics help, exaggeration increases recognition.
- **Roether, Omlor, Christensen & Giese 2009 (J Vision):** critical gait features for emotion perception: speed,
  posture (head inclination, trunk flexion) and limb amplitude.
  - Sad gait: slower, head down, reduced amplitude.
  - Angry and happy gaits: faster, larger amplitudes.
  - **Many cues are speed-dependent.**
- **Dael, Mortillaro & Scherer 2012 (Emotion):** body action and posture coding (BAP). Bodily expression is
  multi-cue and only partly emotion-specific.
- **Wallbott 1998 (Eur J Soc Psych):** movement quality and activity, rather than specific postures, distinguish
  many emotions.
- **Kleinsmith & Bianchi-Berthouze 2013 (IEEE TAC):** survey of affective body expression perception and
  recognition. Observer agreement varies by emotion, context and culture.
- **Heider & Simmel 1944; Tremoulet & Feldman 2000 (Perception):** animacy and intention are attributed to simple
  moving shapes from speed and direction changes.
  - Relevant for low-DoF expression.
  - Also a warning: observers attribute mental states to almost anything that moves contingently.
- **Barrett et al. 2019 (Psych Sci Public Interest):** emotion categories are not reliably read from expressions
  across contexts and cultures.
  - **H2 therefore does not assume agreement on emotional meanings.**
  - It uses quality ratings (caution, confidence, curiosity) and free text alongside forced choice.

## 3. Movement qualities: timing, acceleration, smoothness, posture, space

- **Laban/Bartenieff Effort and Shape:**
  - Effort axes: Time (sudden–sustained), Weight (strong–light), Space (direct–indirect), Flow (bound–free).
  - **Chi, Costa, Zhao & Badler 2000 (SIGGRAPH), EMOTE:** parameterised Effort/Shape for animated arm and torso
    motion.
  - H's intent channels draw on these:
    - tempo / hesitation ≈ Time / Flow;
    - directness ≈ Space;
    - lean / head elevation ≈ Shape.
- **Smoothness:**
  - Flash & Hogan 1985: minimum-jerk model.
  - Hogan & Sternad 2009 (J Motor Behav): jerk measures depend on duration, amplitude and arrests.
  - Balasubramanian et al. 2015 (J NeuroEng Rehab): spectral arc length (SPARC) as a robust smoothness measure.
    **H uses SPARC and RMS jerk.**
- **Proxemics:**
  - Hall 1966.
  - Mumm & Mutlu 2011 (HRI): human–robot proxemics depend on gaze and likeability.
  - Kruse et al. 2013 (RAS): human-aware navigation survey.
  - Approach distance and orientation are therefore legitimate expressive and social channels for a mobile robot.

## 4. Expressive locomotion for bipeds and humanoids

| work | Dem | Hum | S/L | Dev | Safe | Gen | Base |
|---|---|---|---|---|---|---|---|
| Destephe et al. 2013–2015 (RO-MAN; Humanoids/IJHR), emotional gait on WABIAN-2R | happy/sad (and intensity) gait parameters from motion capture of actors; robot walked toward subjects (26 subjects for happy/sad) | yes | S (captured/parametric) | no | walking kept | one robot | speed and posture differences |
| Grandia et al. 2024 (RSS; arXiv 2501.05204), Disney bipedal robotic character | RL policies trained with imitation rewards track artist animation; an animation engine composes and blends sources and synchronises "show functions"; real-time operator performances | not reported as a perception study | L (imitation of animation) | no | robust execution is the point | one character | the animation itself; expression is authored |
| Cheng et al. 2024 (RSS), ExBody | humanoid upper body tracks human motion while legs stay robust | no | L | no | yes | multiple motions | — |
| **EmoLo, Kobayashi 2026 (EngrXiv) [preprint]** | **same robot as NERVA: Open Duck Mini V2.** One style-conditioned RL policy, 3 discrete styles (Happy/Neutral/Sad); a style objective modulates head pitch and motion activity through bounded rewards and a command-dependent gate; sim and hardware; evaluated by trial completion and head-pitch trajectories | **no human perception study found** | L (style reward) | no | trials completed; noise, delay, pushes in training | 3 fixed styles | head pitch + activity level; scripted head offsets on a robust walker |
| EMoG 2026 (arXiv 2609.14432) [preprint] | emotional-style code with continuous intensity → MLP generates periodic gait references → one RL tracking policy; dataset of professional performers; LLM parser from language to style | "perceptible expressive cues" (details not checked) | L (from performer data) | no | command tracking maintained | continuous intensities | the reference-generator itself |
| NERVA S1–S6, E1, E1′ (this project) | see `reports/H0_AUDIT.md` | none | L | no | E1′ kept the gate | — | — |

**Implications for H:**
- EmoLo shows that style-conditioned locomotion on this exact robot is possible with head-pitch and activity
  modulation.
  - Programme H does **not** claim novelty for expressive gait on Open Duck.
  - EmoLo's styles are fixed and hand-labelled, not developed from experience, and it reports no perception study.
  - Its head-pitch result matters: it motivates H0's head-envelope measurement.
- Retraining NERVA's locomotion à la EmoLo is outside H's budget and gate order.
  - A preregistered style-training follow-up is listed in the roadmap.

## 5. Motion-style transfer and movement-conditioned policies

- **Holden, Saito & Komura 2016 (SIGGRAPH):** deep framework for character motion synthesis and editing (style
  via Gram matrices).
- **Aberman et al. 2020 (SIGGRAPH):** unpaired motion style transfer from video to animation.
- **Starke et al. 2022 (SIGGRAPH), DeepPhase:** periodic autoencoders give phase manifolds for locomotion.
- **Margolis & Agrawal 2022 (CoRL), "Walk These Ways":** a quadruped policy conditioned on gait parameters
  (frequency, phase offsets, foot-swing height, body height, body pitch, stance width).
  - Shows that **many continuously controllable gait parameters are trainable when trained for from the start.**
  - Contrast NERVA's E1′, which added style after neutral training.
- **Peng et al. 2021 (SIGGRAPH), AMP;** **Peng et al. 2022 (SIGGRAPH), ASE;** **Escontrela et al. 2022 (IROS):**
  adversarial motion priors give style from reference data without hand-written style rewards.
  - They require reference motion datasets, which NERVA does not have beyond Placo gait references.

## 6. Imitation learning from reference motion

- **Peng et al. 2018 (SIGGRAPH), DeepMimic;** **Peng et al. 2020 (RSS), imitating animals:** imitation rewards
  plus RL reproduce reference skills robustly. This is the basis of Open Duck's and NERVA's training.
- **Supporting methods:**
  - BC: Pomerleau 1989;
  - DAgger: Ross, Gordon & Bagnell 2011 (AISTATS);
  - GAIL: Ho & Ermon 2016 (NeurIPS);
  - VAE + GAIL style embeddings: Wang et al. 2017 (NeurIPS).
- **For H:** imitation needs references for each style. NERVA's E1 showed that the reference generator itself
  limits which postures are feasible.

## 7. Recurrent latents and disentangled styles

- **Cho et al. 2014; Chung et al. 2014:** GRU.
- **Ha & Schmidhuber 2018; Hafner et al. 2020 (ICLR), Dreamer/RSSM:** recurrent world-model latents.
- **Higgins et al. 2017 (ICLR), β-VAE:** disentanglement pressure.
- **Locatello et al. 2019 (ICML):** **unsupervised disentanglement is impossible without inductive biases or
  supervision.**
  - Any learned "style latent" in H must be checked for correspondence to an observable quality.
  - Latent axes are not assumed meaningful.
- **APM v0.3 G3/G45:** recurrent latents did not beat stateless or A0 regulation; a specialised fast/slow regulator
  did not beat a GRU.

## 8. RL for expressive or socially aware robotics

- **Chen et al. 2017 (IROS):** socially aware motion planning with deep RL. Social norms as reward and
  constraints.
- **Qureshi et al. 2016 (Humanoids):** multimodal deep RL for greeting interactions, learned from real interaction
  rewards.
- **Christiano et al. 2017 (NeurIPS):** RL from human preferences. The principled route to an expressive objective
  is **human judgements**, which H cannot collect without approval.
- **Sripathy et al. 2022 and Zhou & Dragan 2018** (above): expressive costs learned from user labels.
- **NERVA S5/S6 and EmoLo's bounded rewards:** hand-written expressive rewards invite reward hacking. NERVA's
  policies collapsed to standing.
- **H rule:** no invented "emotion quality" reward. RL only with a valid task objective plus constraints.

## 9. Developmental robotics, intrinsic motivation, experience-dependent behaviour

- **Surveys and books:**
  - Lungarella, Metta, Pfeifer & Sandini 2003 (Connection Science);
  - Asada et al. 2009 (IEEE TAMD), cognitive developmental robotics;
  - Cangelosi & Schlesinger 2015 (MIT Press), Developmental Robotics.
- **Intrinsic motivation and curiosity:**
  - Oudeyer, Kaplan & Hafner 2007 (IEEE TEC);
  - Schmidhuber 2010 (IEEE TAMD);
  - Pathak et al. 2017 (ICML).
- **Hiolle, Cañamero, Davila Ross & Bard 2012 (ACM TiiS):** an arousal-based architecture on Sony AIBO.
  - Novelty raises arousal, which triggers calls for the caregiver.
  - Caregiver comfort lowers arousal and shapes exploration and learning outcomes.
  - "Needy" vs "independent" profiles were rated differently by adults.
  - **This is the closest prior demonstration of experience-dependent internal state changing expressive and
    exploratory behaviour.**
  - Its profiles were designed parameter sets, not developed dispositions.
  - Hiolle, Lewis & Cañamero 2014 (Frontiers in Neurorobotics) extended the model to adapt to caregiver
    responsiveness.
- **Lewis & Cañamero 2016 (Adaptive Behavior):** hedonic quality and homeostasis in a motivated robot.
- **Moerland, Broekens & Jonker 2018 (Machine Learning):** survey of emotion in RL agents.
  - Most "emotions" are functions of value or reward signals (e.g. TD error).
  - A simple baseline for H: expression driven by predicted outcome or surprise.
- **Mood/affective-memory models:** Churamani et al. 2020 [arXiv 2010.07221, preprint]; Mini social robot 2022
  (Int J Soc Robot).
  - Mood is modelled as a slow, decaying state shaped by affective memory.
  - Recognition was tested by online surveys in the Mini study.
  - **No multi-month study found in which a robot's individual differences emerged from its own experience**
    (search, 2026-10-12).

## 10. Personality-like consistency

- **Mischel & Shoda 1995 (Psych Review), CAPS:** personality as stable *if–then* situation–behaviour signatures,
  not context-free traits.
- **Fleeson 2001 (JPSP):** traits as density distributions of states.
- **Animal-personality methods:**
  - Gosling 2001 (Psych Bulletin);
  - Réale et al. 2007 (Biol Rev).
  - Personality is defined operationally as **repeatable individual differences across time and contexts**,
    measured by repeatability (ICC). **H6 adopts this definition.**
- **Robot personality:**
  - Lee, Peng, Jin & Yan 2006 (J Communication): people recognise designed robot personality;
  - Tapus, Ţăpuş & Matarić 2008 (Intell Serv Robotics): personality matching in assistive robots;
  - Robert et al. 2020 (Foundations & Trends in Robotics), review.
  - Almost all robot personalities are **designed presets**, the explicit non-goal of H6.

## 11. Multimodal and embodiment-independent behaviour representations

- **Kopp et al. 2006 (IVA), Behavior Markup Language (BML):**
  - embodiment-independent behaviour specification, realised by embodiment-specific realisers;
  - the conceptual ancestor of H's `ExpressionIntent → adapter` split.
- **Saerbeck & Bartneck 2010:** the same motion parameters gave similar affect attributions on two embodiments.
- **Bretan, Hoffman & Weinberg 2015 (IJHCS):** emotionally expressive dynamic physical behaviours (Shimi).
- **Learned cross-embodiment transfer exists for task control:**
  - Open X-Embodiment / RT-X 2023 (manipulation);
  - GNM, Shah et al. 2023 (ICRA), navigation.
- **No validated learned cross-embodiment *expression* transfer was found.**
- **H7 position:** a shared interface accepting two robots is not transfer. Claims are limited to measured
  interpretability, and that is untested without human data.

## 12. Computational emotion models

- **PAD:** Mehrabian & Russell 1974. **Circumplex:** Russell 1980.
- **Appraisal:**
  - OCC: Ortony, Clore & Collins 1988;
  - Scherer's component process model 2001/2009;
  - EMA: Marsella & Gratch 2009 (Cogn Syst Res).
- **Mood layers:**
  - ALMA: Gebhard 2005 (AAMAS), PAD mood driven by OCC emotions, with personality defaults;
  - WASABI: Becker-Asano & Wachsmuth 2010 (JAAMAS).
- **Robot affect spaces:**
  - Kismet: Breazeal 2003 (IJHCS), arousal–valence–stance → face/voice;
  - Kirby, Forlizzi & Simmons 2010 (RAS): emotions, moods and attitudes on a receptionist robot.
- **Reviews and alternative views:**
  - Marsella, Gratch & Petta 2010, review;
  - constructed emotion: Barrett 2017.
- **Assessment for H:**
  - PAD is a **descriptive** coordinate system used mostly to *drive* designed expressions. It has no demonstrated
    advantage as a learned state.
  - APM v0.3 showed PAD harmful for regulation; that does **not** decide its expressive value.
  - H3 therefore includes PAD (S4) against scalar, small continuous, GRU and fast/slow states on *expressive*
    criteria.

---

## Synthesis: what is known, what is not, and the strongest simple baselines

1. **Known.** Low-DoF and mobile robots can convey readable qualities through speed, acceleration, path shape,
   orientation, head posture and proximity, mostly with scripted motion. Learned style on legged robots works when
   trained from the start (Walk These Ways, Disney, EmoLo).
2. **Largely not shown.**
   - **Internal states developed from a robot's own experienced outcomes that then change its expression in a
     revisable, history-caused way, tested against fixed-rule and memory baselines.**
   - Hiolle et al. come closest, with arousal-driven exploration shaped by caregiver responses. Their profiles
     were designed, and no strong baseline comparison was made.
3. **Strongest simple baselines H must beat:**
   - (a) fixed scripted styles;
   - (b) a transparent lookup from current risk/uncertainty or memory statistics to a style (cf. Moerland et al.:
     most "emotions" are value/prediction signals);
   - (c) speed alone, as the trivial expressive channel.
4. **Supervision.** Without human perception data, "interpretable" cannot be optimised or claimed.
   - Self-supervised motion representations are not guaranteed to map to qualities (Locatello et al.).
   - H therefore separates claims A/B/C and treats automated observers as exploratory only.
