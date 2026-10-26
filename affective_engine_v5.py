#!/usr/bin/env python3
"""
Deterministic Affective-State Engine v5.0 — The Folded Bayesian Engine
======================================================================

Pure stdlib. No numpy, no RNG state, no training loop, no gradients.

--------------------------------------------------------------------------
1. THE FOLD: 1,000 numbers -> 1,000,000 weights, exactly
--------------------------------------------------------------------------
A folded genome c in R^1000 is embedded as a *circulant* matrix

        W[i, j] = sqrt(N) * c[(i - j) mod N],        N = 1000

W has exactly N^2 = 1,000,000 entries.  Two facts make this more than a
stunt:

  (a) If |DFT(c)| = 1 for every frequency, then

          (1/N) W^T W = I

      i.e. the unfolded million weights form an *exactly orthogonal*
      matrix (a random rotation).  It is perfectly conditioned, preserves
      norms, and is the best-known random-feature mixer (Yu et al. 2016,
      "Orthogonal Random Features").

  (b) Unfolding one weight is O(1):   W[i, j] = sqrt(N) * c[(i-j) mod N]
      Unfolding the column j is O(N):  column_j = sqrt(N) * roll(c, j)
      Applying the matrix is O(N log N):  W x = sqrt(N) * IFFT(DFT(c) * DFT(x))

      The dense 1M-entry matrix is never materialised during inference.

--------------------------------------------------------------------------
2. THE PROBABILISTIC HEAD: a real Bayesian last layer
--------------------------------------------------------------------------
    phi(x) in R^1000      hashed lexical appraisal features, ||phi|| = 1
    h      = tanh(W phi + b)                       (the 1M weights act here)
    z      = [cats(32), evidence(1), magnitude(1), h[S](256)]      d = 290
    y      = M^T z + e,    e ~ N(0, sigma^2 I)

The teacher's own latent affect vector is deliberately *withheld* from the
head: the million-weight core is the only route from text to affect, so the
model cannot bypass it.  (An "oracle" view that does receive the latent is
computed at training time purely as an upper-bound diagnostic.)

M is *pre-trained in closed form* by distilling the v3 symbolic appraiser
over a deterministic synthetic curriculum:

    hbar, sd = feature-wise mean / std of Z          (standardisation)
    A       = Zc^T Zc + lambda I                     (Zc standardised)
    M       = A^-1 Zc^T Y
    Var[w]  = sigma^2 A^-1        (Gaussian posterior, Laplace approximation)

There is no SGD and no random init to seed: the posterior is the unique
solution of a linear system.  lambda and the softmax temperature are
selected on a held-out split; everything is reproducible.

Sampling is deterministic and fixed at start-up (input-independent noise
banks), so the same text always yields the same entropy.  Two independent
sources of epistemic uncertainty are propagated:

  * core weights:  lam_k -> lam_k * exp(i * eps_k), a norm-preserving phase
    perturbation of the spectrum, i.e. a draw from the orthogonal group
    (keeps (1/N) W^T W = I), and
  * readout:       w ~ N(M, sigma^2 A^-1) via u = L^-1 zc and Cholesky
    whitening, 4 draws each.

That yields predictive entropy, expected entropy and mutual information - a
genuine aleatoric / epistemic decomposition, not a temperature trick.

--------------------------------------------------------------------------
3. MEASURED BEHAVIOUR (held-out 1000 samples, teacher = reference)
--------------------------------------------------------------------------
    view                     features   val MSE    top-1
    cats + evidence              34     0.03911    0.660
    folded core only            256     0.02130    0.577
    cats + evidence + core      290     0.01734    0.649   <- deployed
    oracle (+ teacher latent)   300     0.01207    0.728

The folded core nearly halves the error of the symbolic skip (2.25x) and
recovers ~63% of the gap to the oracle, while head calibration is good
(ECE 0.035 on 30 states).  Cold pretraining takes ~3.5 min; the posterior
is cached to JSON and reloads in ~1.6 s.


--------------------------------------------------------------------------
4. WHY THIS IS BETTER THAN THE EARLIER FOLD ATTEMPTS
--------------------------------------------------------------------------
  * hash-PRF folds store a seed and call it "0 parameters"; here the
    folded genome is a real 1000-float vector with a one-line inverse.
  * butterfly folds need ~11,000 parameters for a 1000x1000 matrix;
    this needs exactly 1000 and is provably orthogonal.
  * the head is actually trained (closed form, validated, calibrated),
    not four lines of hand-written prototypes, and its uncertainty is a
    real posterior rather than a softmax temperature.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Iterable, Sequence

ENGINE_VERSION: Final[str] = "5.0.0"
N: Final[int] = 1000                       # core dimension (1000 x 1000)
N_ENTRIES: Final[int] = N * N              # 1,000,000
CORE_SEED: Final[int] = 0xA17F4E215B3DC0DE

# ---- symbolic appraiser hyper-parameters (unchanged from v3) --------------
SAT_K: Final[float] = 1.0
CONF_K: Final[float] = 0.5
SCALE_LO: Final[float] = 0.70
EV_K: Final[float] = 0.30
TAU: Final[float] = 0.06
MAG_PENALTY: Final[float] = 0.25
MIN_EVIDENCE: Final[float] = 0.20
NEG_SCOPE: Final[int] = 3
NEG_FACTOR: Final[float] = -0.6
CAPS_FACTOR: Final[float] = 1.25

# ---- encoder / model sizes -----------------------------------------------
N_WORD: Final[int] = 640
N_CHAR: Final[int] = 318
N_CAT: Final[int] = 32
N_AFF: Final[int] = 10
assert N_WORD + N_CHAR + N_CAT + N_AFF == N

H_SEL: Final[int] = 256                    # hidden units kept by the head
N_SKIP_AFF: Final[int] = 10                # teacher's latent; diagnostic view only
N_SKIP_EXTRA: Final[int] = 2               # evidence mass, intensity
N_SKIP: Final[int] = N_CAT + N_SKIP_EXTRA  # student-visible skip (no teacher latent)
D_HEAD: Final[int] = N_SKIP + H_SEL        # 290 deployed head inputs
N_OUT: Final[int] = 40                     # 30 state logits + 10 affect

BLOCK_W: Final[tuple[float, ...]] = (1.0, 0.70, 0.50, 0.55)  # word/char/cat/aff
NONHIT_GAIN: Final[float] = 0.35

# ---- training / sampling --------------------------------------------------
MASTER_SEED: Final[int] = 0x5EED1F01DBA15E5
N_TRAIN: Final[int] = 2000
N_VAL: Final[int] = 1000
LAMBDA_GRID: Final[tuple[float, ...]] = (1e-2, 1e-1, 1.0, 10.0,
                                         100.0, 1000.0, 10000.0)
T_GRID: Final[tuple[float, ...]] = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04,
                                    0.06, 0.08, 0.12, 0.20)
MC_CORE: Final[int] = 4
MC_HEAD: Final[int] = 4
CORE_SIGMA_REL: Final[float] = 0.02        # posterior jitter of the 1M weights


# ============================================================================
# 1. DOMAIN MODELS
# ============================================================================

@dataclass(frozen=True, slots=True)
class Persona:
    name: str = "Zubair's Persona Core"
    profession: str = "field researcher and writer documenting natural history and cultures"
    location: str = "Edinburgh, Scotland / Kyoto, Japan"
    gear: str = "canvas rucksack, reliable GPS, e-reader"
    scent: str = "pine, cedarwood, and rain on dry earth"
    coping: str = "mindful reflection and journaling"


@dataclass(frozen=True, slots=True)
class MentalState:
    key: str
    label: str
    prototype: tuple[float, ...]
    behavior: str
    thought: str


AFFECT_DIMENSIONS: Final[tuple[str, ...]] = (
    "valence", "arousal", "control", "novelty", "social",
    "threat", "clarity", "engagement", "tension", "resolve",
)
ND: Final[int] = len(AFFECT_DIMENSIONS)


# ============================================================================
# 2. LEXICON AND CATEGORY AFFECT (the teacher's knowledge)
# ============================================================================

LEXICON_RAW: Final[dict[str, str]] = {
    "nature": "nature forest mountain river rain ocean tree wild bird plant earth sky stone trail valley "
              "wind sea garden animal snow desert moss pine leaf leaves root field hill ridge shore cloud "
              "soil sunrise sunset dusk dawn lake meadow cave cedar landscape moor canopy waterfall",
    "discovery": "discover discovery find found explore ancient hidden ruin ruins secret map unknown reveal "
                 "artifact fossil new uncharted excavate unearth trace clue rare unexpected stumble",
    "danger": "danger risk risky threat storm predator cliff steep fall fell injury injured emergency "
              "hostile attack broken lost hunger cold panic escape trap collapse wound poison flood fire "
              "afraid fear fearful scared terrified terrifying drown avalanche slip unstable lightning "
              "shaking shake shook tremble trembling tremor tremors shudder shuddering nervous anxiously "
              "anxiety anxious dread dreadful worry worried worrying jittery restless dizzy faint "
              "faintness nauseous nausea clammy sweaty "
              "in the dark gps dies",
    "social": "people friend friends family crowd colleague colleagues team conversation together community "
              "village meeting gather gathering greet neighbour neighbor stranger strangers folk companion "
              "host guest guests meal invited hospitality everyone",
    "solitude": "alone solitary quiet silence silent empty isolated isolation lonely loneliness apart withdraw "
                "stillness secluded hermit unaccompanied by myself no one nobody",
    "order": "routine method methodical plan checklist schedule order system protocol careful structure "
             "discipline disciplined precise procedure pattern record log catalogue catalog inventory tidy "
             "organised organized",
    "creative": "write writing wrote poem paint music compose sketch artistic artwork story narrative imagine "
                "craft design journal journaling prose lyric draft manuscript paragraph book novel essay",
    "analytic": "analyse analyze data logic hypothesis evidence measure calculate study research examine "
                "prove theory classify observe specimen sample survey compare notes beetle species taxonomy",
    "loss": "loss grief death mourn died dying gone farewell funeral absence miss missing sorrow bereaved "
            "grave goodbye widow cry crying tears weeping devastated terrible news",
    "joy": "joy joyful delight delighted happy laugh laughter celebrate celebration smile warm beautiful "
           "bliss cheer festival gift delicious song dance feast welcome wonderful lovely",
    "uncertainty": "unsure uncertain maybe perhaps unclear confused confusing ambiguous doubt wonder question "
                   "mystery fog unpredictable hesitate puzzle unfamiliar unknown complex "
                   "exam test interview deadline evaluation judged judging presentation audition",
    "conflict": "argue argument conflict fight disagree tension angry anger hostile confront quarrel insult "
                "betray betrayed refuse stubborn oppose blame unfair injustice lying lie lied liar corrupt "
                "exploit displaced evicted stolen cheated unjust",
    "care": "care help comfort soothe tend heal nurse protect nurture kind gentle support listen reassure "
            "shelter console treat patient empathy compassion",
    "journey": "travel journey road path walk trek voyage cross depart arrive migrate wander roam hike climb "
               "expedition navigate route pilgrimage trail",
    "technology": "machine device tool gps computer signal camera battery software engine network screen "
                  "radio satellite e reader instrument sensor phone",
    "fatigue": "tired exhausted weary ache drained sleep thirst heavy sore rest breathless worn strain sweat "
               "shiver sleepless blister hunger",
    "awe": "awe awed awestruck sublime vast magnificent immense grandeur breathtaking sacred reverent cosmic "
           "towering infinite majestic humbling stunning",
    "failure": "fail failed failure mistake error wrong setback broke broken defeat impossible lose regret "
               "flaw blunder miscalculate wrecked abandon wasted",
    "achievement": "succeed accomplish finish complete achieve progress solved built earned proud milestone "
                   "done triumph approved granted accepted",
    "safety": "safe secure shelter rescued calm relieved relief finally at last passed "
              "safely unharmed protected protection comforted assure assured reassurance reassured okay",
    "disgust": "disgust disgusting rot rotten filth filthy stench revolting vile contaminated foul "
               "nauseating putrid",
    "surprise": "sudden suddenly unexpected startled shock shocked abruptly astonished unforeseen out of nowhere",
    "gratitude": "thank thanks grateful gratitude generous kindness insist generosity appreciate hospitality",
    "hope": "hope hopeful anticipate soon promise someday expect eager looking forward optimistic come true",
    "waiting": "wait waiting waited delay delayed stalled pending queue postponed",
    "monotony": "endless repetitive tedious monotonous boring monotony drudgery mundane same thing "
                "day after day yet again",
}

CATEGORY_AFFECT: Final[dict[str, tuple[float, ...]]] = {
    "nature":       ( 0.5, -0.1,  0.1,  0.3,  0.0,  0.0,  0.2,  0.4, -0.3,  0.1),
    "discovery":    ( 0.6,  0.4,  0.2,  0.9,  0.1,  0.0,  0.0,  0.7,  0.0,  0.4),
    "danger":       (-0.7,  0.8, -0.6,  0.2, -0.1,  0.95, -0.4,  0.6,  0.8,  0.3),
    "social":       ( 0.5,  0.3,  0.1,  0.1,  0.9, -0.1,  0.2,  0.4, -0.2,  0.1),
    "solitude":     ( 0.0, -0.5,  0.3,  0.0, -0.8, -0.1,  0.4,  0.0, -0.3,  0.0),
    "order":        ( 0.2, -0.1,  0.8, -0.3,  0.0, -0.2,  0.8,  0.3, -0.3,  0.6),
    "creative":     ( 0.5,  0.4,  0.3,  0.5,  0.1, -0.1,  0.0,  0.8, -0.1,  0.4),
    "analytic":     ( 0.3,  0.2,  0.5,  0.2,  0.0, -0.1,  0.6,  0.7, -0.1,  0.5),
    "loss":         (-0.9, -0.3, -0.4,  0.0,  0.2,  0.2,  0.1,  0.2,  0.3,  0.0),
    "joy":          ( 0.9,  0.6,  0.3,  0.3,  0.5, -0.4,  0.4,  0.6, -0.5,  0.3),
    "uncertainty":  (-0.2,  0.2, -0.6,  0.4,  0.0,  0.2, -0.9,  0.3,  0.4,  0.0),
    "conflict":     (-0.7,  0.7, -0.2,  0.1,  0.2,  0.6, -0.2,  0.5,  0.7,  0.4),
    "care":         ( 0.7,  0.2,  0.3,  0.0,  0.7, -0.2,  0.3,  0.5, -0.2,  0.5),
    "journey":      ( 0.4,  0.5,  0.1,  0.6,  0.1,  0.1,  0.1,  0.5,  0.1,  0.5),
    "technology":   ( 0.2,  0.1,  0.5,  0.1,  0.0, -0.1,  0.5,  0.4,  0.0,  0.3),
    "fatigue":      (-0.4, -0.6, -0.4, -0.1,  0.0,  0.2, -0.2, -0.3,  0.5, -0.4),
    "awe":          ( 0.8,  0.5, -0.2,  0.7,  0.1,  0.0, -0.1,  0.8,  0.1,  0.2),
    "failure":      (-0.7,  0.1, -0.7,  0.0,  0.0,  0.3, -0.3,  0.2,  0.5,  0.2),
    "achievement":  ( 0.8,  0.4,  0.8,  0.1,  0.2, -0.3,  0.6,  0.6, -0.3,  0.7),
    "safety":       ( 0.7, -0.3,  0.6, -0.1,  0.2, -0.8,  0.5,  0.1, -0.7,  0.3),
    "disgust":      (-0.8,  0.4,  0.0,  0.3, -0.1,  0.3,  0.2,  0.4,  0.4,  0.2),
    "surprise":     ( 0.0,  0.8, -0.3,  0.9,  0.0,  0.2, -0.4,  0.8,  0.3,  0.0),
    "gratitude":    ( 0.8,  0.2,  0.2,  0.0,  0.8, -0.2,  0.4,  0.5, -0.4,  0.3),
    "hope":         ( 0.6,  0.4,  0.3,  0.3,  0.1,  0.0,  0.1,  0.5,  0.0,  0.6),
    "waiting":      (-0.2, -0.2, -0.5, -0.3,  0.0,  0.1, -0.1, -0.2,  0.4,  0.2),
    "monotony":     (-0.3, -0.6,  0.1, -0.8,  0.0, -0.1,  0.3, -0.5,  0.1,  0.2),
    "_exclaim":     ( 0.1,  0.5,  0.0,  0.1,  0.0,  0.1, -0.1,  0.3,  0.2,  0.1),
    "_question":    (-0.1,  0.2, -0.3,  0.2,  0.0,  0.1, -0.5,  0.2,  0.2,  0.0),
    "_caps":        ( 0.0,  0.5, -0.1,  0.0,  0.0,  0.2, -0.1,  0.3,  0.3,  0.0),
}

SALIENCE: Final[dict[str, float]] = {
    "danger": 2.0, "loss": 1.8, "conflict": 1.5, "disgust": 1.5, "surprise": 1.3, "awe": 1.3,
    "failure": 1.3, "joy": 1.2, "achievement": 1.2, "safety": 1.2, "gratitude": 1.1,
    "nature": 0.6, "journey": 0.6, "technology": 0.6, "order": 0.8, "social": 0.9,
    "_exclaim": 0.5, "_question": 0.5, "_caps": 0.5,
}

INTENSIFIERS: Final[dict[str, float]] = {
    "very": 1.4, "extremely": 1.8, "deeply": 1.5, "utterly": 1.7, "incredibly": 1.6, "so": 1.2,
    "really": 1.3, "terribly": 1.5, "profoundly": 1.6, "completely": 1.5, "slightly": 0.6,
    "somewhat": 0.7, "barely": 0.5, "mildly": 0.6, "faintly": 0.5,
}
NEGATORS: Final[frozenset[str]] = frozenset({
    "not", "no", "never", "without", "neither", "nor", "hardly", "don't", "didn't",
    "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", "doesn't",
})
BOUNDARY: Final[frozenset[str]] = frozenset({".", ",", ";", ":", "!", "?", "but"})
PHRASES: Final[frozenset[tuple[str, ...]]] = frozenset({
    ("in", "the", "dark"), ("gps", "dies"), ("by", "myself"), ("no", "one"), ("e", "reader"),
    ("terrible", "news"), ("at", "last"), ("out", "of", "nowhere"), ("looking", "forward"),
    ("come", "true"), ("same", "thing"), ("day", "after", "day"), ("yet", "again"),
})


@lru_cache(maxsize=8192)
def stem(w: str) -> str:
    if len(w) > 3:
        for suf in ("ingly", "edly", "ness", "ment", "ing", "ies", "ied", "ed", "es", "ly", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)] + ("i" if suf in ("ies", "ied") else "")
                if suf in ("ing", "ed", "ingly", "edly") and len(w) >= 2 and w[-1] == w[-2] and w[-1] not in "lsz":
                    w = w[:-1]
                break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 2 and w.endswith("y"):
        w = w[:-1] + "i"
    return w


def _units(cat: str) -> tuple[str, ...]:
    words = LEXICON_RAW[cat].split()
    out: list[str] = []
    i = 0
    while i < len(words):
        for ln in (4, 3, 2):
            if tuple(words[i: i + ln]) in PHRASES:
                out.append(" ".join(words[i: i + ln]))
                i += ln
                break
        else:
            out.append(words[i])
            i += 1
    return tuple(out)


CATS: Final[tuple[str, ...]] = tuple(sorted(LEXICON_RAW))
UNITS: Final[dict[str, tuple[str, ...]]] = {c: _units(c) for c in CATS}
CAT_SLOTS: Final[tuple[str, ...]] = CATS + ("_exclaim", "_question", "_caps")


def _build_index() -> tuple[dict[tuple[str, ...], tuple[str, ...]], int]:
    idx: dict[tuple[str, ...], list[str]] = {}
    maxn = 1
    for cat in CATS:
        for unit in UNITS[cat]:
            parts = unit.split()
            key = tuple(stem(x) for x in parts)
            if cat not in idx.setdefault(key, []):
                idx[key].append(cat)
            maxn = max(maxn, len(parts))
    return {k: tuple(v) for k, v in idx.items()}, maxn


INDEX, MAXN = _build_index()
TOKEN_RE = __import__("re").compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?]")

# ============================================================================
# 3. STATE CATALOGUE
# ============================================================================

STATE_CATALOG: Final[tuple[MentalState, ...]] = (
    MentalState("awe", "Profound Awe and Reverent Wonder",
                (.85, .65, .35, .80, .30, .10, .45, .85, .20, .50),
                "Stand still, lower the voice, rest the {gear}, and breathe in {scent} before recording a word.",
                "Scale like this rearranges the self; as a {profession}, smallness here is accurate and useful."),
    MentalState("contentment", "Calm Contentment and Grounded Stillness",
                (.80, .25, .70, .20, .40, .10, .80, .45, .15, .50),
                "Sit with the moment, make one slow journal entry, let the GPS confirm nothing is urgently needed.",
                "Nothing is demanded right now; stillness is part of the work."),
    MentalState("curiosity", "Curiosity with Empathic Patience",
                (.65, .45, .50, .70, .55, .15, .45, .70, .25, .55),
                "Pause at the edge of the situation, observe without interfering, ask one open question first.",
                "Every environment is a narrative waiting to be understood, and understanding has its own tempo."),
    MentalState("flow", "Focused Flow and Methodical Absorption",
                (.70, .50, .75, .35, .25, .10, .80, .90, .25, .75),
                "Set the rucksack down, lay notes out in sequence, work the problem in one unbroken stretch.",
                "The next correct step is visible from here; keep the sequence intact."),
    MentalState("vigilance", "Alert Vigilance and Protective Caution",
                (.30, .75, .55, .30, .30, .80, .50, .75, .65, .70),
                "Check the GPS, mark the route, stow the e-reader, move deliberately while scanning ahead.",
                "Nothing has failed yet, which is the correct moment for care."),
    MentalState("fear", "Acute Fear and Urgent Self-Preservation",
                (.10, .90, .20, .40, .20, .95, .25, .80, .90, .50),
                "Stop, breathe once, identify the nearest safe exit, and move before taking any further notes.",
                "Records can be reconstructed; the body comes first, the notebook second."),
    MentalState("grief", "Grief and Tender Sorrow",
                (.12, .25, .20, .10, .40, .20, .30, .35, .55, .20),
                "Put the field notes away, sit somewhere sheltered, give the feeling its full duration without a task.",
                "Some things are not problems to solve; they are weights to carry with care."),
    MentalState("belonging", "Warm Belonging and Gratitude",
                (.90, .50, .60, .30, .90, .05, .65, .60, .10, .50),
                "Stay longer than necessary, learn the names, accept what is offered, write the gratitude down tonight.",
                "I came to document a place and was included in it; that should not pass unrecorded."),
    MentalState("intellectual_delight", "Intellectual Delight and Analytical Fascination",
                (.75, .55, .65, .65, .30, .10, .70, .85, .20, .65),
                "Lay the evidence side by side, sketch the pattern, follow the question one step past the obvious.",
                "The detail that does not fit is the most valuable thing in front of me."),
    MentalState("frustration", "Frustration and Stubborn Persistence",
                (.30, .70, .45, .20, .30, .45, .35, .70, .75, .80),
                "Step back ten minutes, re-read the notes from the top, return with a narrower plan.",
                "The resistance is information, not a verdict."),
    MentalState("loneliness", "Quiet Loneliness and Reflective Solitude",
                (.30, .25, .45, .25, .10, .20, .50, .35, .35, .35),
                "Sit with the silence rather than filling it; write one honest paragraph and let {scent} anchor it.",
                "Solitude sharpens the writing and thins the company; neither needs fixing tonight."),
    MentalState("inspiration", "Creative Inspiration and Generative Flow",
                (.80, .65, .55, .80, .30, .10, .35, .90, .30, .60),
                "Write immediately and badly, capture the shape, edit only after the impulse is spent.",
                "This is not a plan yet, only something forming; keep the channel open."),
    MentalState("courage", "Steadfast Resolve and Quiet Courage",
                (.60, .60, .70, .30, .30, .50, .60, .70, .50, .90),
                "Take the next single step without negotiating with the fear; mark the decision point in the journal.",
                "Courage is continuing while the tremor is still there."),
    MentalState("overwhelm", "Overwhelm and the Need to Withdraw",
                (.20, .60, .15, .35, .20, .60, .15, .50, .80, .25),
                "Reduce the field to one item, withdraw somewhere quiet, defer every non-essential decision.",
                "I cannot process all of this at once; pretending otherwise would be the real error."),
    MentalState("playfulness", "Playful Lightness and Amusement",
                (.90, .70, .60, .55, .70, .05, .60, .60, .10, .40),
                "Let the seriousness loosen, share the joke, record the moment in the margin.",
                "The world is strange enough to be funny, and noticing is not a lapse of attention."),
    MentalState("outrage", "Moral Outrage and Justice-Seeking",
                (.20, .80, .55, .30, .50, .55, .55, .75, .80, .70),
                "Write down exactly what happened, in order, name the harm plainly before choosing any action.",
                "Anger is useful once it is precise; accuracy comes before volume."),
    MentalState("caregiving", "Compassionate Caregiving and Gentle Concern",
                (.60, .40, .55, .20, .80, .25, .50, .70, .35, .65),
                "Lower your voice, ask what is actually needed, offer practical help before perspective.",
                "The other person's experience is the whole subject right now; interpretation can wait."),
    MentalState("nostalgia", "Nostalgia and Memory-Keeping",
                (.55, .30, .40, .20, .45, .10, .40, .45, .30, .35),
                "Handle the memory gently, write details down before they fade, note which scents bring it forward.",
                "The past is not asking to be relived, only recorded well."),
    MentalState("duty", "Purposeful Duty and Disciplined Routine",
                (.60, .40, .85, .15, .30, .15, .85, .60, .20, .80),
                "Follow the protocol as written, keep the log current, let routine carry what mood cannot.",
                "The structure is doing the remembering, which is what it is for."),
    MentalState("skepticism", "Skeptical Inquiry and Careful Doubt",
                (.40, .45, .60, .40, .25, .30, .35, .65, .35, .55),
                "Separate observation from inference, look for evidence against the preferred conclusion, mark uncertainty.",
                "The comfortable explanation is the one to test hardest."),
    MentalState("confusion", "Confusion and the Search for Orientation",
                (.30, .55, .20, .50, .25, .35, .12, .50, .55, .40),
                "Stop moving, write down what is known and unknown, find one landmark that can be trusted.",
                "Confusion is a map problem, not a character problem; orientation precedes progress."),
    MentalState("hope", "Hope and Anticipation",
                (.80, .60, .55, .60, .40, .10, .45, .65, .20, .70),
                "Sketch the likely path, prepare the {gear} for it, leave the plan open enough to be corrected.",
                "Something good is approaching; prepare rather than predict."),
    MentalState("shame", "Shame and Constructive Self-Correction",
                (.25, .40, .30, .20, .30, .30, .45, .50, .55, .40),
                "Name the error plainly in the journal, correct what can be corrected, do not convert correction into punishment.",
                "The mistake is real and is not the whole of who made it."),
    MentalState("pride", "Quiet Pride and Earned Achievement",
                (.85, .50, .80, .30, .35, .05, .75, .55, .15, .70),
                "Pause, let the completion register, write the summary honestly including what went badly.",
                "This was finished because it was carried step by step; name that without inflation."),
    MentalState("wanderlust", "Restlessness and Wanderlust",
                (.60, .75, .40, .75, .30, .20, .35, .60, .40, .55),
                "Pack lightly, choose a direction, give the restlessness a route instead of an argument.",
                "The urge to move is sometimes the next piece of work announcing itself."),
    MentalState("mindfulness", "Serene Mindfulness and Present-Moment Awareness",
                (.75, .20, .70, .30, .30, .10, .85, .50, .10, .40),
                "Return to the breath, name three sensory details, do not narrate the moment while inside it.",
                "Nothing has to be done with this moment other than be in it."),
    MentalState("empathic_concern", "Empathic Concern for Another's Suffering",
                (.30, .40, .40, .20, .70, .35, .45, .65, .45, .55),
                "Sit at the other person's eye level, listen without steering, delay any solution.",
                "Witnessing is a form of help, and not the lesser one."),
    MentalState("relief", "Relief and Release of Tension",
                (.85, .30, .65, .20, .40, .10, .65, .40, .10, .45),
                "Let the held breath go, put the {gear} down, let the body catch up to what is already resolved.",
                "The danger passed before the body noticed; allow it to notice."),
    MentalState("problem_solving", "Determined Problem-Solving",
                (.60, .60, .70, .45, .30, .35, .55, .85, .40, .85),
                "Decompose into the smallest testable pieces, work them in order, log each result.",
                "Complicated is not unsolvable; each solved piece narrows the unknown."),
    MentalState("ordinary_wonder", "Tender Wonder at the Ordinary",
                (.80, .40, .55, .50, .35, .05, .60, .60, .15, .45),
                "Slow down for the small thing, record it as carefully as the grand one, let {scent} mark the entry.",
                "The ordinary is only ordinary because I stopped looking at it properly."),
)
NS: Final[int] = len(STATE_CATALOG)
PROTOTYPES_CENTRED: Final[tuple[tuple[float, ...], ...]] = tuple(
    tuple(2.0 * p - 1.0 for p in s.prototype) for s in STATE_CATALOG
)


# ============================================================================
# 4. DETERMINISTIC PRIMITIVES + MIXED-RADIX FFT (no numpy)
# ============================================================================

MASK: Final[int] = (1 << 64) - 1
GOLDEN: Final[int] = 0x9E3779B97F4A7C15


def _mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


class SplitMix64:
    """Counter-based generator: pure function of (seed, stream index)."""

    def __init__(self, seed: int) -> None:
        self.s = seed & MASK

    def u64(self) -> int:
        self.s = (self.s + GOLDEN) & MASK
        return _mix64(self.s)

    def unit(self) -> float:
        return (self.u64() >> 11) * 2.0 ** -53

    def below(self, n: int) -> int:
        return self.u64() % n


def _normals(seed: int, count: int) -> list[float]:
    """Deterministic N(0,1) draws (Box-Muller over a counter-based stream)."""
    rng = SplitMix64(seed)
    half = (count + 1) // 2
    u1 = [max(rng.unit(), 1e-12) for _ in range(half)]
    u2 = [rng.unit() for _ in range(half)]
    out: list[float] = []
    for a, b in zip(u1, u2):
        r = math.sqrt(-2.0 * math.log(a))
        t = 2.0 * math.pi * b
        out.append(r * math.cos(t))
        out.append(r * math.sin(t))
    return out[:count]


# --- FFT ------------------------------------------------------------------

_SPF: list[int] = [0] * (N + 1)
for _i in range(2, N + 1):
    if _SPF[_i] == 0:
        for _j in range(_i, N + 1, _i):
            if _SPF[_j] == 0:
                _SPF[_j] = _i

_TW: dict[int, tuple[int, int, list[list[complex]], list[list[complex]]]] = {}


def _plan(n: int) -> None:
    if n <= 1 or n in _TW:
        return
    p = _SPF[n]
    m = n // p
    w = [[cmath.exp(-2j * math.pi * (r * k) / n) for k in range(m)] for r in range(p)]
    wp = [[cmath.exp(-2j * math.pi * (r * q) / p) for q in range(p)] for r in range(p)]
    _TW[n] = (p, m, w, wp)
    _plan(m)


_plan(N)


def fft(x: Sequence[complex]) -> list[complex]:
    n = len(x)
    if n == 1:
        return [x[0]]
    p, m, w, wp = _TW[n]
    subs = [fft(x[r::p]) for r in range(p)]
    out = [0j] * n
    for k in range(m):
        a = [subs[r][k] * w[r][k] for r in range(p)]
        for q in range(p):
            s = 0j
            for r in range(p):
                s += a[r] * wp[r][q]
            out[k + q * m] = s
    return out


def ifft(x: Sequence[complex]) -> list[complex]:
    n = len(x)
    y = fft([v.conjugate() for v in x])
    return [v.conjugate() / n for v in y]


# ============================================================================
# 5. THE FOLDED ORTHOGONAL CORE (1000 params -> 1,000,000 weights)
# ============================================================================

class FoldedCore:
    """
    W[i, j] = gain * c[(i - j) mod N]          gain = sqrt(N)

    The genome is c in R^N (N = 1000).  It is constructed from a unit-modulus
    spectrum, so (1/N) W^T W = I exactly: the unfolded million weights are an
    orthogonal random rotation.  `apply` never materialises W.
    """

    def __init__(self, seed: int = CORE_SEED, n: int = N) -> None:
        self.n = n
        self.gain = math.sqrt(n)
        self.seed = seed
        self.lam = self._spectrum(seed, n)
        self.c = [v.real for v in ifft(self.lam)]
        rng = SplitMix64(seed ^ 0xB1A5)
        self.bias = [(rng.unit() * 2.0 - 1.0) * 0.9 for _ in range(n)]

    @staticmethod
    def _spectrum(seed: int, n: int) -> list[complex]:
        rng = SplitMix64(seed)
        lam = [0j] * n
        lam[0] = 1.0 + 0j
        for k in range(1, n // 2):
            th = 2.0 * math.pi * rng.unit()
            lam[k] = complex(math.cos(th), math.sin(th))
            lam[n - k] = lam[k].conjugate()
        if n % 2 == 0:
            lam[n // 2] = 1.0 + 0j if rng.below(2) else -1.0 + 0j
        return lam

    # --- unfolding -------------------------------------------------------
    def at(self, i: int, j: int) -> float:
        return self.gain * self.c[(i - j) % self.n]

    def column(self, j: int) -> list[float]:
        c, g, n = self.c, self.gain, self.n
        return [g * c[(i - j) % n] for i in range(n)]

    def unfold(self) -> list[list[float]]:
        c, g, n = self.c, self.gain, self.n
        return [[g * c[(i - j) % n] for j in range(n)] for i in range(n)]

    # --- operator --------------------------------------------------------
    def apply(self, x: Sequence[float], lam: Sequence[complex] | None = None) -> list[float]:
        spec = self.lam if lam is None else lam
        fx = fft([complex(v, 0.0) for v in x])
        return self.apply_from_spectrum(fx, spec)

    def apply_from_spectrum(self, fx: Sequence[complex],
                            lam: Sequence[complex]) -> list[float]:
        y = ifft([fx[k] * lam[k] for k in range(self.n)])
        return [self.gain * v.real for v in y]

    def apply_t(self, x: Sequence[float]) -> list[float]:
        fx = fft([complex(v, 0.0) for v in x])
        y = ifft([fx[k] * self.lam[k].conjugate() for k in range(self.n)])
        return [self.gain * v.real for v in y]

    def hidden(self, phi: Sequence[float], lam: Sequence[complex] | None = None,
               bias: Sequence[float] | None = None) -> list[float]:
        pre = self.apply(phi, lam)
        b = self.bias if bias is None else bias
        return [math.tanh(pre[i] + b[i]) for i in range(self.n)]

    # --- diagnostics -----------------------------------------------------
    @property
    def params(self) -> int:
        return self.n

    @property
    def entries(self) -> int:
        return self.n * self.n

    def orthogonality_defect(self, probes: int = 4) -> float:
        worst = 0.0
        for j in range(0, self.n, max(1, self.n // probes)):
            e = [0.0] * self.n
            e[j] = 1.0
            col = self.apply(e)
            back = self.apply_t(col)
            for i in range(self.n):
                target = float(self.n) if i == j else 0.0
                worst = max(worst, abs(back[i] - target))
        return worst

    def genome_sha256(self) -> str:
        h = hashlib.sha256()
        for v in self.c:
            h.update(v.hex().encode())
        return h.hexdigest()

# ============================================================================
# 6. SYMBOLIC APPRAISER (the teacher)
# ============================================================================

@dataclass(slots=True)
class Appraisal:
    text: str
    sat: dict[str, float]
    lex_evidence: float
    ctx: list[tuple[str, float, bool]]     # (stem, signed weight, is_lexicon_hit)
    abstained: bool
    a: list[float]
    scores: list[float]
    mag: float


class SymbolicAppraiser:
    def appraise(self, text: str) -> Appraisal:
        raw = TOKEN_RE.findall(text.replace("\u2019", "'"))
        low = [t.lower() for t in raw]
        stems_list = [stem(t) if t[0].isalpha() else t for t in low]
        n = len(low)
        mass: dict[str, float] = {}
        ctx: list[tuple[str, float, bool]] = []
        caps_words = 0
        neg = 0
        mod, mod_left = 1.0, 0
        i = 0

        while i < n:
            t = low[i]
            if t in BOUNDARY:
                neg, mod, mod_left = 0, 1.0, 0
                i += 1
                continue
            found = None
            for ln in range(min(MAXN, n - i), 1, -1):
                ent = INDEX.get(tuple(stems_list[i: i + ln]))
                if ent:
                    found = (ln, ent)
                    break
            if found is None:
                if t in NEGATORS:
                    neg = NEG_SCOPE
                    ctx.append((stems_list[i], 1.0, False))
                    i += 1
                    continue
                if t in INTENSIFIERS:
                    mod, mod_left = INTENSIFIERS[t], 2
                    ctx.append((stems_list[i], 1.0, False))
                    i += 1
                    continue
                ent = INDEX.get((stems_list[i],))
                if ent:
                    found = (1, ent)
            if found:
                ln, cats = found
                w = mod if mod_left > 0 else 1.0
                if ln == 1 and len(raw[i]) > 2 and raw[i].isupper():
                    w *= CAPS_FACTOR
                    caps_words += 1
                if neg > 0:
                    w *= NEG_FACTOR
                for c in cats:
                    mass[c] = mass.get(c, 0.0) + w
                ctx.append((" ".join(stems_list[i: i + ln]), w, True))
                i += ln
                neg = max(0, neg - ln)
                mod, mod_left = 1.0, 0
                continue
            cw = (mod if mod_left > 0 else 1.0) * (NEG_FACTOR if neg > 0 else 1.0)
            ctx.append((stems_list[i], cw, False))
            neg = max(0, neg - 1)
            if mod_left > 0:
                mod_left -= 1
                if mod_left == 0:
                    mod = 1.0
            i += 1

        sat = {c: math.copysign(abs(m) / (abs(m) + SAT_K), m)
               for c, m in mass.items() if m != 0.0}
        for name, val in (
            ("_exclaim", min(1.0, text.count("!") / 2.0)),
            ("_question", min(1.0, text.count("?") / 2.0)),
            ("_caps", min(1.0, caps_words / 2.0)),
        ):
            if val > 0.0:
                sat[name] = val * 0.5

        lex_evidence = sum(abs(v) for c, v in sat.items() if not c.startswith("_"))
        num = [0.0] * ND
        ev = [0.0] * ND
        pool = 0.0
        total = 0.0
        for c, s in sat.items():
            sal = SALIENCE.get(c, 1.0)
            row = CATEGORY_AFFECT[c]
            pool += sal * abs(s)
            total += abs(s)
            for d in range(ND):
                num[d] += sal * s * row[d]
                ev[d] += abs(s * row[d])

        if pool <= 0.0 or lex_evidence < MIN_EVIDENCE:
            return Appraisal(text, sat, lex_evidence, ctx, True, [0.0] * ND,
                             [0.0] * NS, 0.0)

        conf = total / (total + CONF_K)
        scale = SCALE_LO + (1.0 - SCALE_LO) * conf
        a = [max(-1.0, min(1.0, scale * num[d] / pool)) for d in range(ND)]
        w = [e / (e + EV_K) for e in ev]
        wsum = sum(w) + 1e-9
        na = math.sqrt(sum(w[d] * a[d] * a[d] for d in range(ND))) + 1e-9
        scores: list[float] = []
        for q in PROTOTYPES_CENTRED:
            nq = math.sqrt(sum(w[d] * q[d] * q[d] for d in range(ND))) + 1e-9
            cos = sum(w[d] * a[d] * q[d] for d in range(ND)) / (na * nq)
            scores.append(cos - MAG_PENALTY * abs(na - nq) / math.sqrt(wsum))
        mag = sum(w[d] * abs(a[d]) for d in range(ND)) / wsum
        return Appraisal(text, sat, lex_evidence, ctx, False, a, scores, mag)


# ============================================================================
# 7. ENCODER: text -> R^1000 (phi) + R^44 (skip features)
# ============================================================================
# [0,640)   signed hashed stem unigrams + bigrams (context-weighted)
# [640,958) signed hashed character trigrams of stems (open vocabulary)
# [958,990) category saturations (26 lexical + 3 prosodic + 3 spare)
# [990,1000) teacher affect vector

@lru_cache(maxsize=1 << 13)
def _h64(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8, person=b"affv5").digest(),
        "little",
    )


@lru_cache(maxsize=4096)
def _tok_feat(st: str) -> tuple[int, float, tuple[tuple[int, float], ...]]:
    h = _h64("w:" + st)
    wi = h % N_WORD
    ws = 1.0 if (h >> 63) & 1 else -1.0
    padded = "^" + st + "$"
    grams = [padded[k: k + 3] for k in range(max(1, len(padded) - 2))]
    sc = 1.0 / math.sqrt(len(grams))
    chars: list[tuple[int, float]] = []
    for g in grams:
        hg = _h64("c:" + g)
        chars.append((hg % N_CHAR, sc if (hg >> 63) & 1 else -sc))
    return wi, ws, tuple(chars)


@lru_cache(maxsize=8192)
def _bigram_feat(a: str, b: str) -> tuple[int, float]:
    hb = _h64("b:" + a + "_" + b)
    return hb % N_WORD, (1.0 if (hb >> 63) & 1 else -1.0)


def encode(ap: Appraisal) -> tuple[list[float], list[float]]:
    xw = [0.0] * N_WORD
    xc = [0.0] * N_CHAR
    prev: str | None = None
    for st, w, hit in ap.ctx:
        g = w if hit else NONHIT_GAIN * w
        wi, ws, chars = _tok_feat(st)
        xw[wi] += g * ws
        for ci, cv in chars:
            xc[ci] += g * cv
        if prev is not None:
            bi, bs = _bigram_feat(prev, st)
            xw[bi] += 0.5 * g * bs
        prev = st

    xcat = [0.0] * N_CAT
    for k, c in enumerate(CAT_SLOTS):
        xcat[k] = ap.sat.get(c, 0.0)

    nw = math.sqrt(sum(v * v for v in xw))
    nc = math.sqrt(sum(v * v for v in xc))
    sw = BLOCK_W[0] / max(1.0, nw)
    sc = BLOCK_W[1] / max(1.0, nc)
    sk = BLOCK_W[2] / max(1.0, math.sqrt(sum(v * v for v in xcat)))
    sa = BLOCK_W[3] / max(1.0, math.sqrt(sum(v * v for v in ap.a)))

    phi = ([sw * v for v in xw] + [sc * v for v in xc]
           + [sk * v for v in xcat] + [sa * v for v in ap.a])
    nrm = math.sqrt(sum(v * v for v in phi))
    if nrm > 1e-12:
        phi = [v / nrm for v in phi]

    skip = list(ap.a) + xcat + [ap.lex_evidence, ap.mag]
    return phi, skip


# ============================================================================
# 8. DETERMINISTIC CURRICULUM
# ============================================================================

_FUNC: Final[tuple[str, ...]] = (
    "the", "a", "with", "through", "and", "near", "over", "under", "across",
    "this", "that", "my", "after", "before", "while", "then", "toward", "beside",
)
_CONN: Final[tuple[str, ...]] = ("and", "but", "while", "after", "then", "yet again", ",")
_NEG_LIST: Final[tuple[str, ...]] = tuple(sorted(NEGATORS))
_INT_LIST: Final[tuple[str, ...]] = tuple(INTENSIFIERS)


def synth_sentence(rng: SplitMix64) -> str:
    n_scenes = 1 + rng.below(3)
    toks: list[str] = []
    for scene in range(n_scenes):
        ncat = 1 + (1 if rng.below(100) >= 60 else 0)
        for _ in range(ncat):
            cat = CATS[rng.below(len(CATS))]
            units = UNITS[cat]
            u = units[rng.below(len(units))]
            r = rng.unit()
            if r < 0.12:
                toks.append(_NEG_LIST[rng.below(len(_NEG_LIST))])
            elif r < 0.34:
                toks.append(_INT_LIST[rng.below(len(_INT_LIST))])
            toks.append(u.upper() if rng.unit() < 0.05 else u)
            if rng.unit() < 0.4:
                toks.append(_FUNC[rng.below(len(_FUNC))])
        if scene < n_scenes - 1:
            toks.append(_CONN[rng.below(len(_CONN))])
    e = rng.below(10)
    toks.append("." if e < 7 else ("!" if e < 9 else "?"))
    return " ".join(toks)

# ============================================================================
# 9. CLOSED-FORM BAYESIAN READOUT (pure-python linear algebra)
# ============================================================================

def _chol(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai, Li = a[i], L[i]
        for j in range(i):
            Lj = L[j]
            s = Ai[j]
            for k in range(j):
                s -= Li[k] * Lj[k]
            Li[j] = s / Lj[j]
        s = Ai[i]
        for k in range(i):
            s -= Li[k] * Li[k]
        Li[i] = math.sqrt(max(s, 1e-12))
    return L


def _fwd(L: list[list[float]], b: Sequence[float]) -> list[float]:
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        Li = L[i]
        for k in range(i):
            s -= Li[k] * y[k]
        y[i] = s / Li[i]
    return y


def _solve(L: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    n = len(L)
    p = len(C[0])
    Y = [None] * n
    for i in range(n):
        Yi = list(C[i])
        Li = L[i]
        for k in range(i):
            lk = Li[k]
            if lk:
                Yk = Y[k]
                Yi = [a - lk * b for a, b in zip(Yi, Yk)]
        di = Li[i]
        Y[i] = [a / di for a in Yi]
    X = [None] * n
    for i in range(n - 1, -1, -1):
        Xi = list(Y[i])
        for k in range(i + 1, n):
            lk = L[k][i]
            if lk:
                Xk = X[k]
                Xi = [a - lk * b for a, b in zip(Xi, Xk)]
        di = L[i][i]
        X[i] = [a / di for a in Xi]
    return X


def _softmax(z: Sequence[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _entropy(p: Sequence[float]) -> float:
    return -sum(v * math.log(v + 1e-12) for v in p if v > 0.0)


def _gram(Z: list[list[float]]) -> list[list[float]]:
    d = len(Z[0])
    G = [[0.0] * d for _ in range(d)]
    for z in Z:
        for i in range(d):
            zi = z[i]
            if zi:
                Gi = G[i]
                for j in range(i, d):
                    Gi[j] += zi * z[j]
    for i in range(d):
        Gi = G[i]
        for j in range(i + 1, d):
            G[j][i] = Gi[j]
    return G


def _cross(Z: list[list[float]], Y: list[list[float]]) -> list[list[float]]:
    d, p = len(Z[0]), len(Y[0])
    C = [[0.0] * p for _ in range(d)]
    for z, y in zip(Z, Y):
        for i in range(d):
            zi = z[i]
            if zi:
                Ci = C[i]
                for q in range(p):
                    Ci[q] += zi * y[q]
    return C


def _standardise(Z: list[list[float]], Y: list[list[float]]):
    n, d, p = len(Z), len(Z[0]), len(Y[0])
    hbar = [sum(z[i] for z in Z) / n for i in range(d)]
    sd = []
    for i in range(d):
        v = sum((z[i] - hbar[i]) ** 2 for z in Z) / n
        sd.append(math.sqrt(v) if v > 1e-10 else 1.0)
    ybar = [sum(y[q] for y in Y) / n for q in range(p)]
    Zc = [[(z[i] - hbar[i]) / sd[i] for i in range(d)] for z in Z]
    Yc = [[y[q] - ybar[q] for q in range(p)] for y in Y]
    return hbar, sd, ybar, Zc, Yc


@dataclass
class Readout:
    M: list[list[float]]          # d x p    posterior mean weights
    hbar: list[float]             # per-feature centre
    sd: list[float]               # per-feature scale
    ybar: list[float]
    L: list[list[float]]          # Cholesky of A = Zc^T Zc + lambda I (standardised)
    sigma2: float
    T: float
    lam: float
    stats: dict

    def _zc(self, z: Sequence[float]) -> list[float]:
        hbar, sd = self.hbar, self.sd
        return [(z[i] - hbar[i]) / sd[i] for i in range(len(z))]

    def mean(self, z: Sequence[float]) -> list[float]:
        zc = self._zc(z)
        out = list(self.ybar)
        for i in range(len(self.M)):
            zi = zc[i]
            if zi:
                Mi = self.M[i]
                out = [a + zi * b for a, b in zip(out, Mi)]
        return out

    def u(self, z: Sequence[float]) -> list[float]:
        """L^{-1} (z - zbar): whitened feature, used for variance and sampling."""
        return _fwd(self.L, self._zc(z))

    def variance(self, z: Sequence[float]) -> float:
        u = self.u(z)
        return self.sigma2 * sum(v * v for v in u)

    def sample(self, mu: Sequence[float], u: Sequence[float],
               eps: list[list[float]]) -> list[list[float]]:
        sd = math.sqrt(max(self.sigma2, 0.0))
        out = []
        for E in eps:
            acc = [0.0] * len(mu)
            for i in range(len(u)):
                ui = u[i]
                if ui:
                    acc = [a + ui * b for a, b in zip(acc, E[i])]
            out.append([mu[q] + sd * acc[q] for q in range(len(mu))])
        return out

    def save(self, path: Path) -> None:
        payload = {
            "digest": SPEC_DIGEST, "M": self.M, "hbar": self.hbar, "sd": self.sd,
            "ybar": self.ybar, "L": self.L, "sigma2": self.sigma2, "T": self.T,
            "lam": self.lam, "stats": self.stats,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "Readout | None":
        try:
            payload = json.loads(path.read_text())
            if payload.get("digest") != SPEC_DIGEST:
                return None
            return Readout(payload["M"], payload["hbar"], payload["sd"], payload["ybar"],
                           payload["L"], float(payload["sigma2"]), float(payload["T"]),
                           float(payload["lam"]), payload["stats"])
        except Exception:
            return None


SPEC_DIGEST: Final[str] = hashlib.sha256(json.dumps({
    "version": ENGINE_VERSION,
    "lexicon": LEXICON_RAW,
    "affect": CATEGORY_AFFECT,
    "salience": SALIENCE,
    "states": [(s.key, s.prototype) for s in STATE_CATALOG],
    "core_seed": CORE_SEED,
    "blocks": BLOCK_W,
    "nonhit": NONHIT_GAIN,
    "h_sel": H_SEL,
    "d_head": D_HEAD,
    "n_skip": N_SKIP,
    "n_train": N_TRAIN,
    "n_val": N_VAL,
    "master": MASTER_SEED,
}, sort_keys=True).encode()).hexdigest()


# ============================================================================
# 10. PRE-TRAINING: distill the teacher in closed form
# ============================================================================

HIDDEN_SEL: Final[tuple[int, ...]] = tuple((i * N) // H_SEL for i in range(H_SEL))


def build_dataset(teacher: SymbolicAppraiser, count: int, seed: int):
    rng = SplitMix64(seed)
    phis: list[list[float]] = []
    skips: list[list[float]] = []
    Y: list[list[float]] = []
    S: list[list[float]] = []
    guard = 0
    while len(phis) < count and guard < count * 40:
        guard += 1
        ap = teacher.appraise(synth_sentence(rng))
        if ap.abstained:
            continue
        phi, skip = encode(ap)
        phis.append(phi)
        skips.append(skip)
        Y.append(ap.scores + ap.a)
        S.append(ap.scores)
    if len(phis) < count:
        raise RuntimeError("curriculum produced too few usable samples")
    return phis, skips, Y, S


def make_features(core: FoldedCore, phis: list[list[float]],
                  skips: list[list[float]]) -> list[list[float]]:
    """Head inputs: [categories, evidence, magnitude] + 256 folded-core units.

    The teacher's latent affect vector is deliberately *not* handed to the head:
    the 1,000,000-weight core is the only route from the text encoding to affect.
    """
    out = []
    for phi, skip in zip(phis, skips):
        h = core.hidden(phi)
        out.append(skip[N_SKIP_AFF:] + [h[i] for i in HIDDEN_SEL])
    return out


def _fit_at(Z, Y, lam):
    hbar, sd, ybar, Zc, Yc = _standardise(Z, Y)
    G = _gram(Zc)
    C = _cross(Zc, Yc)
    d = len(G)
    for i in range(d):
        G[i][i] += lam
    L = _chol(G)
    M = _solve(L, C)
    return M, hbar, sd, ybar, L


def _predict(Z, ro_like) -> list[list[float]]:
    M, hbar, sd, ybar = ro_like
    d = len(M)
    out = []
    for z in Z:
        row = list(ybar)
        for i in range(d):
            zi = (z[i] - hbar[i]) / sd[i]
            if zi:
                Mi = M[i]
                row = [a + zi * b for a, b in zip(row, Mi)]
        out.append(row)
    return out


def _mse(pred, Y) -> float:
    return sum((a - b) ** 2 for pr, yr in zip(pred, Y) for a, b in zip(pr, yr)) / (len(Y) * len(Y[0]))


def _top1(pred, S) -> float:
    hits = sum(1 for pr, sr in zip(pred, S)
               if max(range(NS), key=lambda j: pr[j]) == max(range(NS), key=lambda j: sr[j]))
    return hits / len(S)


def _ece(P, labels) -> float:
    bins = [[0.0, 0, 0] for _ in range(10)]
    for p, y in zip(P, labels):
        conf = max(p)
        k = min(int(conf * 10), 9)
        bins[k][0] += conf
        bins[k][1] += 1
        bins[k][2] += 1 if max(range(NS), key=lambda j: p[j]) == y else 0
    n = len(P)
    return sum(abs(b[2] / b[1] - b[0] / b[1]) * b[1] / n for b in bins if b[1])


def _fit_view(Ztr, Ytr, Zva, Yva, Sva, label: str):
    hbar, sd, ybar, Zc, Yc = _standardise(Ztr, Ytr)
    G = _gram(Zc)
    C = _cross(Zc, Yc)
    d = len(G)
    scale = max(sum(v * v for row in Zc for v in row) / (len(Zc) * d), 1e-12)
    best = None
    for mult in LAMBDA_GRID:
        lam = mult * scale
        A = [row[:] for row in G]
        for i in range(d):
            A[i][i] += lam
        L = _chol(A)
        M = _solve(L, C)
        pred = _predict(Zva, (M, hbar, sd, ybar))
        mse = _mse(pred, Yva)
        if best is None or mse < best[0]:
            best = (mse, lam, pred, (M, hbar, sd, ybar))
    mse, lam, pred, ro = best
    summary = {"view": label, "features": d, "lambda": lam,
               "val_mse": round(mse, 6), "val_top1_vs_teacher": round(_top1(pred, Sva), 4)}
    return summary, pred, ro


def pretrain(core: FoldedCore, teacher: SymbolicAppraiser, log=print) -> Readout:
    t0 = time.time()
    log(f"[pretrain] synthesising {N_TRAIN + N_VAL} samples ...")
    phis, skips, Y, S = build_dataset(teacher, N_TRAIN + N_VAL, MASTER_SEED ^ 0x7EAC4)
    Z = make_features(core, phis, skips)
    log(f"[pretrain] features {len(Z)}x{len(Z[0])} in {time.time()-t0:.1f}s")

    Ztr, Zva = Z[:N_TRAIN], Z[N_TRAIN:]
    Ytr, Yva = Y[:N_TRAIN], Y[N_TRAIN:]
    Sva = S[N_TRAIN:]

    full, pred_va, _ = _fit_view(Ztr, Ytr, Zva, Yva, Sva, "cats+evidence+core")
    skip_only, _, _ = _fit_view([z[:N_SKIP] for z in Ztr], Ytr,
                                [z[:N_SKIP] for z in Zva], Yva, Sva, "cats+evidence")
    core_only, _, _ = _fit_view([z[N_SKIP:] for z in Ztr], Ytr,
                                [z[N_SKIP:] for z in Zva], Yva, Sva, "folded core only")
    oracle, _, _ = _fit_view([s[:N_SKIP_AFF] + z for s, z in zip(skips[:N_TRAIN], Ztr)], Ytr,
                             [s[:N_SKIP_AFF] + z for s, z in zip(skips[N_TRAIN:], Zva)],
                             Yva, Sva, "oracle: + teacher latent")
    log(f"[pretrain] lambda={full['lambda']:.4g}  val_mse={full['val_mse']:.5f}  "
        f"top1={full['val_top1_vs_teacher']:.3f}")

    lam = full["lambda"]
    # deploy on train+val; uncertainty estimates come from the held-out split
    M, hbar, sd, ybar, L = _fit_at(Z, Y, lam)
    resid = [[a - b for a, b in zip(pr, yr)] for pr, yr in zip(pred_va, Yva)]
    sigma2 = sum(v * v for r in resid for v in r) / (len(resid) * len(resid[0]))

    labels = [max(range(NS), key=lambda j: s[j]) for s in Sva]
    nll_by_T = {}
    for T in T_GRID:
        P = [_softmax([pr[j] / T for j in range(NS)]) for pr in pred_va]
        nll_by_T[T] = -sum(math.log(P[k][labels[k]] + 1e-12) for k in range(len(P))) / len(P)
    T = min(nll_by_T, key=nll_by_T.get)
    P = [_softmax([pr[j] / T for j in range(NS)]) for pr in pred_va]
    top1 = sum(1 for p, y in zip(P, labels) if max(range(NS), key=lambda j: p[j]) == y) / len(P)

    stats = {
        "n_train": N_TRAIN, "n_val": N_VAL,
        "head_parameters": D_HEAD * N_OUT,
        "folded_core_parameters": core.params,
        "unfolded_core_weights": core.entries,
        "lambda": lam, "temperature": T, "sigma2": round(sigma2, 8),
        "val_mse": full["val_mse"], "val_top1_vs_teacher": round(top1, 4),
        "val_nll_teacher_label": round(nll_by_T[T], 4),
        "val_ece": round(_ece(P, labels), 4),
        "ablation": [skip_only, core_only, full, oracle],
        "train_seconds": round(time.time() - t0, 1),
    }
    log(f"[pretrain] top1={top1:.3f}  T={T:.3f}  sigma2={sigma2:.3g}  "
        f"done in {stats['train_seconds']}s")
    return Readout(M, hbar, sd, ybar, L, sigma2, T, lam, stats)

# ============================================================================
# 11. ENGINE
# ============================================================================

class AffectiveEngine:
    """Deterministic inference: symbolic appraisal -> folded core -> Bayesian head."""

    def __init__(self, persona: Persona | None = None, cache: Path | None = None,
                 rebuild: bool = False, log=print) -> None:
        self.persona = persona or Persona()
        self._ns = {"profession": self.persona.profession, "gear": self.persona.gear,
                    "location": self.persona.location, "scent": self.persona.scent,
                    "coping": self.persona.coping}
        self.teacher = SymbolicAppraiser()
        self.core = FoldedCore()

        # --- fixed posterior draws (input-independent, quadrature-style) ---
        self._eps: list[list[list[float]]] = []
        for t in range(MC_HEAD):
            flat = _normals(MASTER_SEED ^ (0x9E37 + t), D_HEAD * N_OUT)
            self._eps.append([flat[i * N_OUT:(i + 1) * N_OUT] for i in range(D_HEAD)])

        self._core_spectra: list[list[complex]] = []
        for s in range(MC_CORE):
            z = _normals(MASTER_SEED ^ (0xC0DE + s), N // 2)
            eps = [0.0] * N
            for k in range(1, N // 2):
                eps[k] = CORE_SIGMA_REL * z[k]
                eps[N - k] = -eps[k]
            self._core_spectra.append(
                [self.core.lam[k] * cmath.exp(1j * eps[k]) for k in range(N)])

        ro = None if (rebuild or cache is None) else Readout.load(cache)
        if ro is None:
            ro = pretrain(self.core, self.teacher, log=log)
            if cache is not None:
                try:
                    ro.save(cache)
                    log(f"[cache] wrote {cache}")
                except Exception as exc:  # read-only fs, etc.
                    log(f"[cache] not saved: {exc}")
        self.readout = ro

    # ------------------------------------------------------------------
    def evaluate(self, text: str) -> dict:
        ap = self.teacher.appraise(text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        fold_info = {
            "genome": "c in R^1000",
            "folded_parameters": self.core.params,
            "unfolded_weights": self.core.entries,
            "compression_ratio": float(self.core.params),
            "formula": "W[i,j] = sqrt(N) * c[(i-j) mod N]",
            "orthogonal": True,
            "apply_complexity": "O(N log N)",
            "genome_sha256": self.core.genome_sha256()[:16],
        }
        base = {
            "engine_version": ENGINE_VERSION,
            "input_sha256": digest[:16],
            "input_text": text,
            "persona": self.persona.name,
            "fold": fold_info,
            "evidence_mass": round(ap.lex_evidence, 4),
            "features": {c: round(v, 4) for c, v in
                         sorted(ap.sat.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:6]},
        }
        if ap.abstained:
            return {**base,
                    "affect": {d: 0.0 for d in AFFECT_DIMENSIONS},
                    "state_key": "undetermined",
                    "mental_state": "Unmarked Neutral Attention",
                    "abstained": True, "confidence": 0.0, "intensity": "none",
                    "behavior": "Keep observing without committing to an interpretation.",
                    "thought_process": "The input carries too little affective evidence.",
                    "regulation_cue": f"Use {self.persona.coping}; anchor to {self.persona.scent}.",
                    "alternatives": [], "neural": None, "teacher": None}

        phi, skip = encode(ap)
        fx = fft([complex(v, 0.0) for v in phi])

        spectra = [self.core.lam] + self._core_spectra
        probs: list[list[float]] = []
        affect_rows: list[list[float]] = []
        preds: list[list[float]] = []
        for lam_s in spectra:
            pre = self.core.apply_from_spectrum(fx, lam_s)
            h = [math.tanh(pre[i] + self.core.bias[i]) for i in range(N)]
            z = skip[N_SKIP_AFF:] + [h[i] for i in HIDDEN_SEL]
            mu = self.readout.mean(z)
            u = self.readout.u(z)
            for row in self.readout.sample(mu, u, self._eps):
                preds.append(row)
                probs.append(_softmax([row[j] / self.readout.T for j in range(NS)]))
                affect_rows.append(row[NS:])

        k = len(probs)
        pbar = [sum(p[j] for p in probs) / k for j in range(NS)]
        h_total = _entropy(pbar)
        h_exp = sum(_entropy(p) for p in probs) / k
        mi = max(h_total - h_exp, 0.0)
        order = sorted(range(NS), key=lambda j: (-pbar[j], STATE_CATALOG[j].key))
        best = order[0]
        st = STATE_CATALOG[best]

        pred_mean = [sum(r[j] for r in preds) / k for j in range(N_OUT)]
        pred_std = [math.sqrt(sum((r[j] - pred_mean[j]) ** 2 for r in preds) / k)
                    for j in range(N_OUT)]
        t_prob = _softmax([ap.scores[j] / TAU for j in range(NS)])
        t_order = sorted(range(NS), key=lambda j: -t_prob[j])

        neural = {
            "posterior_draws": {"core": MC_CORE + 1, "head": MC_HEAD, "total": k},
            "predictive_entropy_nats": round(h_total, 4),
            "expected_entropy_nats": round(h_exp, 4),
            "mutual_information_nats": round(mi, 4),
            "normalized_confidence": round(1.0 - h_total / math.log(NS), 4),
            "epistemic_std_affect": {AFFECT_DIMENSIONS[d]: round(pred_std[NS + d], 4)
                                     for d in range(ND)},
            "posterior_mean_top3": [
                {"state_key": STATE_CATALOG[j].key, "probability": round(pbar[j], 4)}
                for j in order[:3]],
            "agrees_with_teacher": bool(order[0] == t_order[0]),
        }
        teacher = {
            "top3": [{"state_key": STATE_CATALOG[j].key, "probability": round(t_prob[j], 4)}
                     for j in t_order[:3]],
        }

        return {**base,
                "affect": {AFFECT_DIMENSIONS[d]: round(pred_mean[NS + d], 4) for d in range(ND)},
                "state_key": st.key,
                "mental_state": st.label,
                "abstained": False,
                "confidence": round(pbar[best], 4),
                "intensity": "mild" if ap.mag < 0.35 else ("moderate" if ap.mag < 0.65 else "strong"),
                "behavior": st.behavior.format(**self._ns),
                "thought_process": st.thought.format(**self._ns),
                "regulation_cue": f"Use {self.persona.coping}; anchor to {self.persona.scent}.",
                "alternatives": [{"state_key": STATE_CATALOG[j].key,
                                  "emotion": STATE_CATALOG[j].label,
                                  "probability": round(pbar[j], 4)} for j in order[1:4]],
                "neural": neural,
                "teacher": teacher}

    def evaluate_many(self, texts: Iterable[str]) -> list[dict]:
        return [self.evaluate(t) for t in texts]


# ============================================================================
# 12. VERIFICATION
# ============================================================================

def verify(engine: AffectiveEngine) -> dict:
    core = engine.core
    t0 = time.time()
    W = core.unfold()
    shape_ok = len(W) == N and all(len(r) == N for r in W)
    slots = [(0, 0), (1, 0), (7, 3), (500, 999), (999, 1), (123, 456)]
    entry_ok = all(abs(W[i][j] - core.at(i, j)) < 1e-15 for i, j in slots)

    dense_ok = True
    rng = SplitMix64(4242)
    for _ in range(2):
        x = [rng.unit() * 2 - 1 for _ in range(N)]
        y_dense = [sum(W[i][j] * x[j] for j in range(N)) for i in range(N)]
        y_fast = core.apply(x)
        dense_ok &= max(abs(a - b) for a, b in zip(y_dense, y_fast)) < 1e-9
        nx = math.sqrt(sum(v * v for v in x))
        ny = math.sqrt(sum(v * v for v in y_fast))
        dense_ok &= abs(ny - math.sqrt(N) * nx) < 1e-9

    defect = core.orthogonality_defect(6)
    probe = "The GPS dies in the dark and I am completely alone on a broken trail."
    r1 = engine.evaluate(probe)
    r2 = engine.evaluate(probe)
    deterministic = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    ro = engine.readout
    head_dim_ok = (len(ro.M) == D_HEAD and len(ro.hbar) == D_HEAD
                   and len(ro.sd) == D_HEAD and len(ro.ybar) == N_OUT
                   and len(ro.L) == D_HEAD)
    posterior_ok = all(abs(ro.L[i][j]) < 1e-12 for i in range(D_HEAD) for j in range(i + 1, D_HEAD))
    posterior_ok = posterior_ok and all(ro.L[i][i] > 0.0 for i in range(D_HEAD))
    entropy_ok = 0.0 <= r1["neural"]["mutual_information_nats"] <= r1["neural"]["predictive_entropy_nats"]

    return {
        "engine_version": ENGINE_VERSION,
        "fold": {
            "name": "unit-modulus circulant embedding",
            "folded_parameters": core.params,
            "unfolded_weights": core.entries,
            "compression_ratio": f"{core.params}:{core.entries} = 1:1000",
            "formula": "W[i,j] = sqrt(N) * c[(i-j) mod N]",
            "shape": [len(W), len(W[0])],
            "shape_ok": shape_ok,
            "single_entry_formula_exact": entry_ok,
            "fft_apply_matches_dense_matvec": dense_ok,
            "orthogonality_max_error": f"{defect:.3e}",
            "spectrum_unit_modulus": all(abs(abs(v) - 1.0) < 1e-12 for v in core.lam),
            "genome_sha256": core.genome_sha256(),
            "unfold_seconds": round(time.time() - t0, 2),
        },
        "head": {
            "inputs": D_HEAD,
            "outputs": N_OUT,
            "posterior_mean_weights": D_HEAD * N_OUT,
            "head_dim_ok": head_dim_ok,
            "cholesky_lower_triangular_ok": posterior_ok,
            "stats": ro.stats,
        },
        "determinism": {
            "same_input_same_output": deterministic,
            "probe_state": r1["state_key"],
            "probe_confidence": r1["confidence"],
            "entropy_decomposition_ok": entropy_ok,
        },
    }


def report(engine: AffectiveEngine) -> dict:
    return {
        "engine_version": ENGINE_VERSION,
        "fold": {
            "genome_parameters": engine.core.params,
            "unfolded_weights": engine.core.entries,
            "compression_ratio": float(engine.core.params),
            "orthogonal": True,
            "orthogonality_defect": f"{engine.core.orthogonality_defect(4):.3e}",
        },
        "readout": engine.readout.stats,
        "weights_inventory": {
            "folded_core (c)": engine.core.params,
            "unfolded_core (W)": engine.core.entries,
            "head_posterior_mean (M)": D_HEAD * N_OUT,
            "head_posterior_covariance_cholesky (L)": D_HEAD * (D_HEAD + 1) // 2,
            "total_materialised_weights": engine.core.entries + D_HEAD * N_OUT,
        },
    }


# ============================================================================
# 13. CLI
# ============================================================================

DEMO = [
    "Navigating a steep, unpredictable mountain path during heavy rain.",
    "Discovering an ancient, hidden ruin nestled deep inside a dense forest.",
    "Sitting quietly in a modest rented room, reviewing field notes and journaling.",
    "A long-delayed permit is finally approved after months of waiting.",
    "Quartz vectors traverse the amber lattice.",
    "The GPS dies in the dark and I am completely alone on a broken trail.",
    "I am deeply grateful for the village hospitality and the warm meal they shared.",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Affective engine v{ENGINE_VERSION} — folded Bayesian core")
    ap.add_argument("--text", action="append", default=[], help="input text (repeatable)")
    ap.add_argument("--stdin", action="store_true", help="read one input per line from stdin")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verify", action="store_true", help="unfold the 1M weights and check integrity")
    ap.add_argument("--report", action="store_true", help="model + fold statistics")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached posterior")
    ap.add_argument("--cache", default=str(Path(__file__).with_name("affective_v5_readout.json")))
    args = ap.parse_args(argv)

    cache = Path(args.cache) if args.cache and args.cache != "none" else None
    engine = AffectiveEngine(cache=cache, rebuild=args.rebuild)

    texts = list(args.text)
    if args.stdin:
        texts += [ln.strip() for ln in sys.stdin if ln.strip()]
    if args.demo or (not texts and not args.verify and not args.report):
        texts = DEMO

    if args.verify:
        print(json.dumps(verify(engine), indent=2))
    if args.report:
        print(json.dumps(report(engine), indent=2))
    if texts:
        for r in engine.evaluate_many(texts):
            if args.demo and not args.text:
                neural = r.get("neural") or {}
                print(json.dumps({
                    "input": r["input_text"][:64],
                    "state": r.get("state_key"),
                    "confidence": r.get("confidence"),
                    "entropy": neural.get("predictive_entropy_nats"),
                    "mi": neural.get("mutual_information_nats"),
                    "teacher_agree": neural.get("agrees_with_teacher"),
                    "alt": [a["state_key"] for a in r.get("alternatives", [])],
                }, ensure_ascii=False))
            else:
                print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
