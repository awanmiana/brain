# Child Brain v1 — a developmental agent grown on the v5 affective engine

> I built this the way I would teach a child: give it a temperament, name its
> feelings, let it remember, let it ask questions, let it sleep, and let it
> learn to calm itself down. The result is not a language model and does not
> claim to be conscious. It is a small, deterministic agent that develops.

The v5 engine is the child's **innate temperament**. It is a frozen, folded
1,000,000-weight affective core with a Bayesian readout. `child_brain.py` grows
a developmental agent around it:

```
perceive -> feel -> remember -> predict -> decide -> learn -> sleep
```

Everything is pure Python 3 stdlib. There is no numpy, no network, no external
model, and no stochastic training loop: a given seed reproduces the same child.

---

## 1. What develops

| stage | age | what the child practices | supervision | sleep |
|---|---|---|---:|---:|
| newborn | 0–1 | reflexes only: the innate v5 temperament meets the world | 0% | — |
| infant | 1–2 | names for feelings; one simple scene; heavy scaffolding | 85% | 1 epoch |
| toddler | 2–4 | two scenes, intensifiers, first negation, first questions | 55% | 2 |
| child | 4–7 | mixed feelings, social conflict, less supervision | 40% | 2 |
| adolescent | 7–11 | ambiguity, abstraction, three scenes, metacognition | 30% | 3 |
| adult | 11–16 | novel frames, sparse labels, self-regulation | 22% | 3 |

The curriculum is generated from the same lexicon the v5 teacher uses, but the
caregiver is **not** the teacher. The caregiver has a private **family dialect**:
it leans toward some basic emotions and away from others, and it amplifies or
dampens the ten affect dimensions. The child never sees that dialect. It has to
discover it from labels, questions, and corrections — the way a child learns
the emotional vocabulary of a particular family.

---

## 2. The architecture

### 2.1 Innate core (frozen)

`AffectiveEngine` from `affective_engine_v5.py`:

* 1,000 genome parameters folded into a 1,000 × 1,000 orthogonal circulant,
* text → hashed lexical/character/category features → folded core → Bayesian
  readout,
* 30 fine-grained mental states + 10 affect dimensions,
* calibrated posterior uncertainty.

The child inherits this as a reflex. It is never fine-tuned.

### 2.2 Perception

For each experience the child computes:

* the v5 **innate affect** vector and fine-grained state distribution,
* an **eight-emotion prior** (joy, trust, fear, surprise, sadness, disgust,
  anger, anticipation) by projecting the 30 states onto Plutchik's primitives,
* a compact 48-dimensional sample of the folded core,
* 32 category saturations plus evidence and intensity,
* a memory key built from categories + innate affect.

The fine-grained teacher score is **not** given to the child. Only the child's
own v5 readout is visible to it.

### 2.3 Plastic cortex

A recursive-least-squares last layer learns a residual correction on top of the
innate prior:

```
A_new = γ A + x xᵀ
P_new = (1/γ) [P − P x xᵀ P / (γ + xᵀ P x)]
M_new = M + (P x / (γ + xᵀ P x)) (y − Mᵀ x)
```

* `x` has 121 features; `y` has 18 outputs (8 basic emotions + 10 affect dims).
* `γ < 1` is a **forgetting factor**: young brains stay plastic, older brains
  consolidate. The stage schedule raises `γ` with age.
* The update is a genuine online Bayesian-style update, not SGD with a learning
  rate. It is the same ridge/Kalman family as v5's closed-form head.

### 2.4 Episodic memory

An `EpisodicMemory` stores labelled and unlabelled experiences. At prediction
time the child retrieves the six most similar past episodes and adds:

* a similarity-weighted average of their basic-emotion labels,
* a similarity-weighted average of their affect vectors,
* mean similarity and familiarity.

Only labelled episodes vote. Memory is causal: an episode is stored after it is
used, so the child never remembers the future.

### 2.5 Metacognition

A tiny online logistic model learns `P(correct | confidence, entropy, evidence,
innate confidence, memory similarity, familiarity, epistemic spread, agreement
with reflex)`. The child:

* **answers** when `P(correct)` is above a calibrated threshold,
* **asks the caregiver** when predictive entropy / epistemic uncertainty is high
  and a question budget remains,
* **abstains** otherwise.

The threshold is recalibrated after every stage on a fresh labelled sample.

### 2.6 Curiosity

Uncertainty sampling: the child spends its limited question budget on the
experiences with the highest normalized entropy and epistemic spread. A random
question policy is run as an ablation.

### 2.7 Sleep

After each stage the child replays up to 320 stored episodes for 1–3 epochs.
Replay is what keeps earlier lessons alive when the forgetting factor is below
1. After stage 3 the child also performs mild **synaptic pruning**: the least
useful non-bias feature rows are zeroed.

### 2.8 Self-regulation

A linear Q-model chooses among six coping actions:

```
none, breathe, reframe, seek_support, rest, approach
```

Each action has a fixed effect on the ten affect dimensions. The child learns
which action is appropriate for which state from the caregiver's affect labels.
The reward is wellbeing minus distress. The model includes action × state
interaction features, so it can learn "breathe when threat is high" rather than
one fixed policy.

---

## 3. The full developmental run

Command:

```bash
python3 child_brain.py --develop --ablate
```

Measured on 2,120 lived experiences, 6 stages, held-out milestone sets, 240 s
of CPU. The paired two-seed ablation adds about 9 minutes on the same machine.

### 3.1 Stage results

| stage | basic acc | innate acc | textbook mapping | affect MSE | ECE | coverage | selective acc | questions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| newborn | 0.725 | 0.725 | 0.775 | 0.1539 | 0.248 | 1.00 | 0.725 | 0 |
| infant | **0.980** | 0.770 | 0.750 | 0.0000 | 0.023 | 1.00 | 0.980 | 9 |
| toddler | 0.908 | 0.742 | 0.708 | 0.0000 | 0.052 | 0.97 | 0.922 | 0 |
| child | 0.814 | 0.679 | 0.836 | 0.0000 | 0.068 | 0.71 | 0.960 | 2 |
| adolescent | 0.771 | 0.664 | 0.829 | 0.0000 | 0.061 | 0.77 | 0.861 | 7 |
| adult | **0.857** | 0.693 | 0.829 | 0.0000 | 0.079 | 0.81 | **0.956** | 9 |

* **Innate acc** is the child's v5 reflex on the same held-out items.
* **Textbook mapping** is the best a child could do with the generic Plutchik
  projection and no family dialect. The adult beats it by +2.9 points, so the
  child really did learn the family's private way of naming feelings.
* **Selective acc** is accuracy on the ~81% of items the child chose to answer.
  It answers the easy 81% at 95.6% accuracy and abstains on the rest.

### 3.2 Within-stage learning curves

The probe is a fixed 40-item held-out set, re-read before and during each stage.

| stage | seen 0 | seen 100 | seen 200 | seen 300 | seen 400 |
|---|---:|---:|---:|---:|---:|
| infant | 0.825 | 0.975 | 0.975 | 0.975 | — |
| child | 0.825 | 0.925 | 0.875 | 0.925 | 0.875 |
| adult | 0.875 | 0.925 | 0.900 | 0.925 | 0.925 |

Infant affect MSE falls from **0.160** to **0.00020**. The child starts with a
usable reflex and then learns the family's naming convention. The child-stage
curve is noisier because the stimuli are harder and the supervision is thinner;
the adult curve is a clean rise from 0.875 to 0.925.

### 3.3 Retention

Accuracy on earlier milestone sets after the adult stage:

| remembered stage | accuracy after adult | change vs. when learned |
|---|---:|---:|
| newborn | 0.975 | +0.250 |
| infant | 0.990 | +0.010 |
| toddler | 0.933 | +0.025 |
| child | 0.814 | 0.000 |
| adolescent | 0.814 | +0.043 |

The child remembers 99% of its infant lessons and 93% of its toddler lessons
after four later stages. That is the point of sleep replay.

### 3.4 Ablations

Paired, two-seed, three-stage runs. Values below are averaged over the two
seeds; the full JSON is in `child_brain_report.json`.

| condition | final acc | infant retention | questions | labels used | acc / question |
|---|---:|---:|---:|---:|---:|
| full | **0.829** | **0.983** | 10.0 | 378.5 | **0.0829** |
| no memory | 0.822 | 0.963 | 9.0 | 377.5 | 0.0913 |
| no sleep | 0.822 | 0.967 | 10.5 | 377.5 | 0.0782 |
| random curiosity | 0.811 | 0.963 | 61.5 | 436.0 | 0.0132 |

Reading the ablation carefully:

* **Memory** adds +0.7 points of final accuracy and +2.1 points of infant
  retention. It also improves affect MSE.
* **Curiosity** is the clearest result: uncertainty sampling reaches +1.8
  points over random while asking **6.2× fewer questions** (10.0 vs 61.5).
  Accuracy per question is **6.3× higher** (0.083 vs 0.013).
* **Sleep** is the subtle one. On final accuracy it is marginally above no-sleep
  (0.829 vs 0.822, within the noise of a two-seed, three-stage run), and its
  retention is higher (0.983 vs 0.967). Its job is consolidation, not raw
  acquisition. The full developmental run above is the stronger evidence: 99%
  retention of infant lessons and 93% of toddler lessons after four later
  stages.
* The `acc / question` column is only comparable between conditions with the
  same supervision schedule (full vs random curiosity); no-memory and no-sleep
  ask fewer questions, so their ratio is not a like-for-like comparison.
* Small-sample ablations are noisy by nature. The report keeps per-condition
  seconds and supports multiple seeds so any claim can be re-tested.

### 3.5 Milestones

The run closes with ten milestones scored automatically from the report:

| milestone | result | value |
|---|---|---:|
| recognises basic emotions | pass | 0.857 |
| improves on innate temperament | pass | +0.132 |
| learns the family dialect | pass | +0.029 vs textbook |
| knows when it does not know | pass | +0.114 at 50% coverage |
| is calibrated | pass | ECE 0.079 |
| remembers earlier lessons | pass | 1.01× infant accuracy |
| asks useful questions | pass | 6.29× accuracy per question vs random |
| memory supports recall | pass | +0.021 retention |
| sleep consolidates | pass | +0.017 retention |
| regulates affect | pass | chosen 0.076 vs random −0.013 |

### 3.6 Self-regulation

On 90 held-out adult experiences:

| policy | mean reward |
|---|---:|
| learned action | **0.0764** |
| random action | −0.0131 |
| oracle best action | 0.0807 |

The learned policy is within 5% of the oracle and learns state-dependent
choices (mostly `seek_support`, sometimes `reframe` or `approach`), rather than
one fixed action.

---

## 4. What this has in common with how I work

This is a small, honest analogue of an agent loop, not a copy of a language
model:

| agent faculty | child-brain implementation |
|---|---|
| innate prior / temperament | frozen v5 folded Bayesian core |
| perception | text → appraisal → affect + basic-emotion prior |
| working memory | current feature vector + retrieved episodes |
| long-term memory | episodic store with similarity recall |
| learning from feedback | online RLS cortex + caregiver corrections |
| active learning | uncertainty sampling with a question budget |
| metacognition | learned `P(correct)` + calibrated abstention |
| consolidation | sleep replay + synaptic pruning |
| executive function | learned coping-action Q-values |
| expression | v5 state/behaviour/thought vocabulary |
| development | six stages, changing supervision and plasticity |

What it does **not** have: language generation, planning over long horizons,
tool use, self-model in the strong sense, or consciousness. It is a child-sized
cognitive-affective loop, and it is honest about that.

---

## 5. Reproducibility

* All curriculum, supervision, question, memory, and sleep decisions are driven
  by `SplitMix64` from the seed printed in the report.
* The child's text hashes use SHA-256, not Python's salted `hash()`.
* The v5 innate core is deterministic (same genome SHA-256 every run).
* The checkpoint stores the cortex (`M`, `P`), episodic memory, metacognition,
  calibration temperature, abstention threshold, and regulator.
* `--develop` is a fresh run. `--demo` and `--teach` load the saved child.

The developmental run is not bit-identical across machines because Python's
floating-point summation order is stable within an interpreter but the timing
and CPU environment are not part of the model. Given the same interpreter and
seed, the run is reproducible.

## 6. Files

```
child_brain.py                  the developmental agent (stdlib only)
child_brain_checkpoint.json     learned cortex, memory, metacognition, regulator
child_brain_report.json         stage results, ablations, milestones, demo
affective_engine_v5.py          the frozen innate core
affective_v5_readout.json       cached v5 posterior
```

## 7. Commands

```bash
# full developmental run (about 5 minutes)
python3 child_brain.py --develop

# full run + ablations (about 13 minutes)
python3 child_brain.py --develop --ablate

# load the trained child and inspect new texts
python3 child_brain.py --demo "The GPS dies in the dark and I am completely alone."
python3 child_brain.py --demo "text one" --demo "text two"

# teach the child directly (repeatable; repetition has more effect)
python3 child_brain.py --teach "A rockfall blocks the only path out." fear
python3 child_brain.py --teach "text" joy --teach "text" trust

# print the saved developmental record
python3 child_brain.py --report
```

A single correction is a nudge, not a flip: the innate prior is strong, and
the plastic layer moves a little with each labelled experience. This is
deliberate. If you want the child to learn a new convention quickly, repeat the
lesson or run a short stage of the curriculum.

The demo prints the child's basic-emotion prediction, confidence, whether it
would answer or ask/abstain, its ten affect dimensions, the nearest-memory
similarity, the v5 expressive state/behaviour/thought, and the coping action it
would choose with the predicted post-regulation affect.
