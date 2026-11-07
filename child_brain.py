#!/usr/bin/env python3
"""
Child Brain v1.0 — a developmental agent built on the v5 affective engine
=========================================================================

The v5 engine is the child's *innate temperament*: a frozen, folded 1M-weight
affective core with a Bayesian readout.  This file grows a child around it.

The developmental loop is deliberately agent-like:

    perceive -> feel -> remember -> predict -> decide -> learn -> sleep

  * perceive   text -> v5 appraisal -> innate affect + basic-emotion prior
  * feel       the v5 core supplies a temperament that is never retrained
  * remember   an episodic memory of labelled and unlabelled experiences
  * predict    a plastic cortical readout (recursive least squares) that
               corrects the innate prior toward the caregiver's dialect
  * decide     answer / ask for help / abstain, from a learned metacognitive
               estimate of whether the current prediction is correct
  * learn      online from caregiver labels, questions, and corrections
  * sleep      replay old episodes, recalibrate, and optionally prune

The caregiver is simulated by a deterministic "family policy": it names the
child's feeling with one of eight basic emotions and reports a slightly
idiosyncratic affect vector.  The child must discover that dialect from
experience, just as a child learns the emotional vocabulary of a particular
family.

Everything is pure stdlib and deterministic for a given seed.  No numpy, no
network, no external model.

Usage
-----
    python3 child_brain.py --develop          # full developmental curriculum
    python3 child_brain.py --develop --ablate # + memory/curiosity/sleep ablations
    python3 child_brain.py --demo "..."       # use the saved child checkpoint
    python3 child_brain.py --report           # print the saved report

Artifacts
---------
    child_brain_checkpoint.json   learned cortex, memory, metacognition
    child_brain_report.json       developmental record and evaluations
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Sequence

from affective_engine_v5 import (
    AFFECT_DIMENSIONS,
    CATS,
    HIDDEN_SEL,
    N,
    N_SKIP_AFF,
    ND,
    NS,
    STATE_CATALOG,
    TAU,
    UNITS,
    AffectiveEngine,
    SplitMix64,
    encode,
    _FUNC,
    _CONN,
    _NEG_LIST,
    _INT_LIST,
)

CHILD_VERSION: Final[str] = "1.0.0"
CHILD_SEED: Final[int] = 0xC41D5EED2026
CACHE_PATH: Final[Path] = Path(__file__).with_name("affective_v5_readout.json")
CHECKPOINT_PATH: Final[Path] = Path(__file__).with_name("child_brain_checkpoint.json")
REPORT_PATH: Final[Path] = Path(__file__).with_name("child_brain_report.json")

# ---------------------------------------------------------------------------
# 1. Basic emotions: the vocabulary a caregiver actually uses
# ---------------------------------------------------------------------------
# Plutchik's eight primary emotions.  Every one of the v5 fine-grained states
# is projected onto this smaller vocabulary.  The child must learn the
# projection, because the caregiver never hands it over.

BASIC_EMOTIONS: Final[tuple[str, ...]] = (
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
)
N_BASIC: Final[int] = len(BASIC_EMOTIONS)

STATE_TO_BASIC: Final[dict[str, dict[str, float]]] = {
    "awe":                  {"surprise": 0.65, "joy": 0.20, "trust": 0.15},
    "contentment":          {"joy": 0.60, "trust": 0.40},
    "curiosity":            {"anticipation": 0.55, "surprise": 0.30, "joy": 0.15},
    "flow":                 {"joy": 0.50, "anticipation": 0.30, "trust": 0.20},
    "vigilance":            {"fear": 0.70, "anticipation": 0.20, "anger": 0.10},
    "fear":                 {"fear": 1.00},
    "grief":                {"sadness": 0.90, "trust": 0.10},
    "belonging":            {"trust": 0.50, "joy": 0.50},
    "intellectual_delight": {"joy": 0.50, "surprise": 0.30, "anticipation": 0.20},
    "frustration":          {"anger": 0.70, "sadness": 0.20, "anticipation": 0.10},
    "loneliness":           {"sadness": 0.80, "fear": 0.20},
    "inspiration":          {"joy": 0.50, "surprise": 0.30, "anticipation": 0.20},
    "courage":              {"trust": 0.40, "anticipation": 0.30, "anger": 0.20, "joy": 0.10},
    "overwhelm":            {"fear": 0.60, "sadness": 0.30, "anger": 0.10},
    "playfulness":          {"joy": 0.90, "surprise": 0.10},
    "outrage":              {"anger": 0.80, "disgust": 0.20},
    "caregiving":           {"trust": 0.60, "joy": 0.30, "sadness": 0.10},
    "nostalgia":            {"sadness": 0.50, "joy": 0.30, "trust": 0.20},
    "duty":                 {"trust": 0.50, "anticipation": 0.40, "joy": 0.10},
    "skepticism":           {"disgust": 0.40, "anticipation": 0.40, "fear": 0.20},
    "confusion":            {"fear": 0.40, "surprise": 0.40, "sadness": 0.20},
    "hope":                 {"anticipation": 0.70, "joy": 0.30},
    "shame":                {"sadness": 0.50, "disgust": 0.50},
    "pride":                {"joy": 0.80, "trust": 0.20},
    "wanderlust":           {"anticipation": 0.60, "joy": 0.30, "surprise": 0.10},
    "mindfulness":          {"trust": 0.50, "joy": 0.30, "surprise": 0.20},
    "empathic_concern":     {"trust": 0.50, "sadness": 0.40, "joy": 0.10},
    "relief":               {"joy": 0.60, "trust": 0.40},
    "problem_solving":      {"anticipation": 0.60, "trust": 0.30, "joy": 0.10},
    "ordinary_wonder":      {"surprise": 0.50, "joy": 0.40, "trust": 0.10},
}

_STATE_INDEX: Final[dict[str, int]] = {s.key: i for i, s in enumerate(STATE_CATALOG)}
BASIC_MATRIX: Final[list[list[float]]] = [[0.0] * NS for _ in range(N_BASIC)]
for _state, _weights in STATE_TO_BASIC.items():
    _j = _STATE_INDEX[_state]
    for _emotion, _w in _weights.items():
        BASIC_MATRIX[BASIC_EMOTIONS.index(_emotion)][_j] = _w

# The child's cortex sees a compact, spread-out sample of the folded core.
CHILD_HIDDEN: Final[tuple[int, ...]] = tuple(HIDDEN_SEL[::5][:48])
N_HIDDEN: Final[int] = len(CHILD_HIDDEN)

# Feature layout (kept explicit so checkpoints are self-describing):
#   bias(1) basic_prior(8) innate_affect(10) categories(32) evidence+magnitude(2)
#   hidden(48) memory_basic(8) memory_affect(10) memory_sim+count(2)
FEATURE_DIM: Final[int] = 1 + N_BASIC + ND + 32 + 2 + N_HIDDEN + N_BASIC + ND + 2
N_OUTPUT: Final[int] = N_BASIC + ND
MEMORY_KEY_DIM: Final[int] = 34 + N_BASIC + ND


# ---------------------------------------------------------------------------
# 2. Small numerical helpers
# ---------------------------------------------------------------------------

def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Sequence[float]) -> list[float]:
    n = norm(a)
    return [v / n for v in a] if n > 1e-12 else list(a)


def softmax(z: Sequence[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def entropy(p: Sequence[float]) -> float:
    return -sum(v * math.log(v + 1e-12) for v in p if v > 0.0)


def sigmoid(z: float) -> float:
    if z >= 0.0:
        q = math.exp(-z)
        return 1.0 / (1.0 + q)
    q = math.exp(z)
    return q / (1.0 + q)


def matvec(m: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in m]


def argmax(x: Sequence[float]) -> int:
    return max(range(len(x)), key=lambda i: x[i])


def onehot(i: int, n: int, smooth: float = 0.0) -> list[float]:
    base = smooth / n
    out = [base] * n
    out[i] += 1.0 - smooth
    return out


def shuffle(a: list, rng: SplitMix64) -> None:
    for i in range(len(a) - 1, 0, -1):
        j = rng.below(i + 1)
        a[i], a[j] = a[j], a[i]


def text_hash(text: str) -> str:
    """Process-stable short hash (Python's hash() is salted per process)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_NEG_SET: Final[frozenset[str]] = frozenset(_NEG_LIST)
_INT_SET: Final[frozenset[str]] = frozenset(_INT_LIST)


def difficulty(p: "Perception") -> float:
    """A caregiver's rough ordering: short, single-scene, unmodified first."""
    toks = p.text.lower().split()
    neg = sum(1 for t in toks if t in _NEG_SET)
    inten = sum(1 for t in toks if t in _INT_SET)
    return (len(toks) + 2.5 * neg + 1.5 * inten
            + 0.8 * min(3.0, p.evidence) + 0.5 * abs(p.magnitude))


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.5
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def ece(probs: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 10) -> float:
    acc = [[0.0, 0, 0] for _ in range(bins)]
    for p, y in zip(probs, labels):
        conf = max(p)
        b = min(int(conf * bins), bins - 1)
        acc[b][0] += conf
        acc[b][1] += 1
        acc[b][2] += 1 if argmax(p) == y else 0
    n = len(probs)
    return sum(abs(b[2] / b[1] - b[0] / b[1]) * b[1] / n for b in acc if b[1])


# ---------------------------------------------------------------------------
# 3. Developmental stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stage:
    idx: int
    name: str
    age: str
    description: str
    n_train: int
    n_test: int
    supervision: float
    curiosity: float
    sleep_epochs: int
    gamma: float
    max_scenes: int
    max_cats: int
    neg_p: float
    int_p: float
    cats: tuple[str, ...]
    learn: bool = True
    memory: bool = True
    novelty: float = 0.0


BASIC_CATS: Final[tuple[str, ...]] = (
    "joy", "danger", "loss", "social", "discovery", "nature", "order", "care",
)
MID_CATS: Final[tuple[str, ...]] = BASIC_CATS + (
    "uncertainty", "conflict", "journey", "fatigue", "gratitude", "hope", "waiting", "technology",
)
ALL_CATS: Final[tuple[str, ...]] = tuple(CATS)

STAGES: Final[tuple[Stage, ...]] = (
    Stage(0, "newborn", "0-1", "Reflexes only: the innate v5 temperament meets the world.",
          120, 80, 0.00, 0.00, 0, 0.990, 1, 1, 0.00, 0.00, BASIC_CATS,
          learn=False, memory=False),
    Stage(1, "infant", "1-2", "Names for feelings: simple scenes, heavy caregiver scaffolding.",
          300, 100, 0.85, 0.05, 1, 0.990, 1, 1, 0.00, 0.00, BASIC_CATS),
    Stage(2, "toddler", "2-4", "Two-scene stories, intensifiers, first negation, first questions.",
          400, 120, 0.55, 0.12, 2, 0.993, 2, 1, 0.12, 0.22, MID_CATS),
    Stage(3, "child", "4-7", "Mixed feelings, social conflict, less supervision, more memory.",
          450, 140, 0.40, 0.15, 2, 0.995, 2, 2, 0.22, 0.32, ALL_CATS),
    Stage(4, "adolescent", "7-11", "Ambiguity, abstraction, three scenes, metacognitive practice.",
          450, 140, 0.30, 0.15, 3, 0.997, 3, 2, 0.32, 0.42, ALL_CATS, novelty=0.3),
    Stage(5, "adult", "11-16", "Transfer: novel frames, sparse labels, self-regulation.",
          400, 140, 0.22, 0.10, 3, 0.998, 3, 3, 0.38, 0.48, ALL_CATS, novelty=0.6),
)


# ---------------------------------------------------------------------------
# 4. Curriculum generation
# ---------------------------------------------------------------------------

def synth_stage_sentence(rng: SplitMix64, stage: Stage) -> str:
    """Generate one age-appropriate scene description."""
    n_scenes = 1 + rng.below(stage.max_scenes)
    toks: list[str] = []
    for scene in range(n_scenes):
        n_cat = 1 + rng.below(stage.max_cats)
        used: set[str] = set()
        for _ in range(n_cat):
            cat = stage.cats[rng.below(len(stage.cats))]
            # avoid repeating exactly the same category inside one scene
            if cat in used and len(stage.cats) > 1:
                cat = stage.cats[rng.below(len(stage.cats))]
            used.add(cat)
            units = UNITS[cat]
            u = units[rng.below(len(units))]
            if rng.unit() < stage.neg_p:
                toks.append(_NEG_LIST[rng.below(len(_NEG_LIST))])
            if rng.unit() < stage.int_p:
                toks.append(_INT_LIST[rng.below(len(_INT_LIST))])
            toks.append(u.upper() if rng.unit() < 0.04 else u)
            if rng.unit() < 0.45:
                toks.append(_FUNC[rng.below(len(_FUNC))])
        if scene < n_scenes - 1:
            toks.append(_CONN[rng.below(len(_CONN))])
    if stage.novelty > 0.0 and rng.unit() < stage.novelty:
        frame = ("somehow", "as if", "the meaning of", "a strange sense that", "almost")
        toks.insert(0, frame[rng.below(len(frame))])
    e = rng.below(10)
    toks.append("." if e < 7 else ("!" if e < 9 else "?"))
    return " ".join(toks)


# ---------------------------------------------------------------------------
# 5. Perception: text -> innate affect + basic-emotion prior
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Perception:
    text: str
    basic_prior: list[float]
    innate_affect: list[float]
    skip: list[float]
    hidden: list[float]
    key: list[float]
    teacher_states: list[float]
    teacher_affect: list[float]
    evidence: float
    magnitude: float


def perceive(engine: AffectiveEngine, text: str) -> Perception | None:
    ap = engine.teacher.appraise(text)
    if ap.abstained:
        return None
    phi, skip = encode(ap)
    h = engine.core.hidden(phi)
    z = skip[N_SKIP_AFF:] + [h[i] for i in HIDDEN_SEL]
    mu = engine.readout.mean(z)
    p_states = softmax([mu[j] / engine.readout.T for j in range(NS)])
    basic_prior = matvec(BASIC_MATRIX, p_states)
    innate_affect = mu[NS:]
    hidden = [h[i] for i in CHILD_HIDDEN]
    teacher_states = softmax([s / TAU for s in ap.scores])
    # The memory key must not include the teacher's latent affect (skip[:10]).
    # Only the child-visible categories, its innate prior, and its own affect
    # are allowed into memory.
    key = unit(list(skip[N_SKIP_AFF:]) + basic_prior + innate_affect)
    return Perception(
        text=text,
        basic_prior=basic_prior,
        innate_affect=innate_affect,
        skip=list(skip),
        hidden=hidden,
        key=key,
        teacher_states=teacher_states,
        teacher_affect=list(ap.a),
        evidence=ap.lex_evidence,
        magnitude=ap.mag,
    )


def generate_perceptions(engine: AffectiveEngine, stage: Stage, count: int,
                         rng: SplitMix64) -> list[Perception]:
    out: list[Perception] = []
    guard = 0
    while len(out) < count and guard < count * 30:
        guard += 1
        p = perceive(engine, synth_stage_sentence(rng, stage))
        if p is not None:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 6. The caregiver and its hidden family dialect
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Label:
    basic: int
    affect: list[float]
    probs: list[float]
    teacher_basic: int
    teacher_basic_probs: list[float]


@dataclass(frozen=True, slots=True)
class Caregiver:
    bias: list[float]
    gain: list[float]
    offset: list[float]

    @staticmethod
    def from_seed(seed: int) -> "Caregiver":
        rng = SplitMix64(seed)
        # A family does not use the textbook prototype.  It leans toward some
        # feelings and away from others, and it amplifies threat/social warmth.
        bias = [(rng.unit() * 2.0 - 1.0) * 0.55 for _ in range(N_BASIC)]
        gain = [1.0 + (rng.unit() * 2.0 - 1.0) * 0.22 for _ in range(ND)]
        offset = [(rng.unit() * 2.0 - 1.0) * 0.10 for _ in range(ND)]
        return Caregiver(bias, gain, offset)

    def label(self, p: Perception) -> Label:
        p_basic = matvec(BASIC_MATRIX, p.teacher_states)
        logits = [math.log(p_basic[j] + 1e-9) + self.bias[j] for j in range(N_BASIC)]
        probs = softmax(logits)
        basic = argmax(probs)
        affect = [clip(self.gain[d] * p.teacher_affect[d] + self.offset[d]) for d in range(ND)]
        teacher_basic = argmax(p_basic)
        return Label(basic, affect, probs, teacher_basic, p_basic)

    def approval(self, answered: bool, asked: bool, correct: bool) -> float:
        if answered:
            return 1.0 if correct else -1.0
        if asked:
            return 0.4
        return 0.2 if not correct else -0.2


# ---------------------------------------------------------------------------
# 7. Episodic memory
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Episode:
    key: list[float]
    x: list[float]
    basic: int | None
    affect: list[float] | None
    stage: int
    text_hash: str
    t: int


class EpisodicMemory:
    """Similarity-based episodic store.  Only labelled episodes vote."""

    def __init__(self, maxlen: int = 1200, k: int = 6) -> None:
        self.episodes: list[Episode] = []
        self.maxlen = maxlen
        self.k = k
        self.t = 0

    def add(self, key: list[float], x: list[float], basic: int | None,
            affect: list[float] | None, stage: int, text_hash: str) -> None:
        self.t += 1
        ep = Episode(list(key), list(x), basic, None if affect is None else list(affect),
                     stage, text_hash, self.t)
        if len(self.episodes) >= self.maxlen:
            # Keep a spread of ages: drop the oldest unlabelled episode first,
            # then the oldest episode.  This prevents the infant stage from
            # disappearing while still bounding retrieval cost.
            drop = None
            for i, old in enumerate(self.episodes):
                if old.basic is None:
                    drop = i
                    break
            if drop is None:
                drop = 0
            self.episodes.pop(drop)
        self.episodes.append(ep)

    def retrieve(self, key: list[float]) -> tuple[list[float], list[float], float, float]:
        if not self.episodes:
            return [0.0] * N_BASIC, [0.0] * ND, 0.0, 0.0
        scored = [(dot(key, ep.key), ep) for ep in self.episodes]
        scored.sort(key=lambda t: -t[0])
        top = scored[: self.k]
        pos = [(s, ep) for s, ep in top if ep.basic is not None and s > 0.02]
        if not pos:
            return [0.0] * N_BASIC, [0.0] * ND, 0.0, 0.0
        weights = [max(0.0, s) ** 2 for s, _ in pos]
        sw = sum(weights) + 1e-12
        mem_basic = [0.0] * N_BASIC
        mem_affect = [0.0] * ND
        sim = 0.0
        for w, (s, ep) in zip(weights, pos):
            sim += w * s
            mem_basic[ep.basic] += w  # type: ignore[index]
            for d in range(ND):
                mem_affect[d] += w * ep.affect[d]  # type: ignore[index]
        mem_basic = [v / sw for v in mem_basic]
        mem_affect = [v / sw for v in mem_affect]
        return mem_basic, mem_affect, sim / sw, min(1.0, len(pos) / 5.0)

    def labelled(self) -> list[Episode]:
        return [e for e in self.episodes if e.basic is not None]

    def to_json(self) -> dict:
        return {
            "maxlen": self.maxlen, "k": self.k, "t": self.t,
            "episodes": [
                {"key": e.key, "x": e.x, "basic": e.basic, "affect": e.affect,
                 "stage": e.stage, "text_hash": e.text_hash, "t": e.t}
                for e in self.episodes
            ],
        }

    @staticmethod
    def from_json(payload: dict) -> "EpisodicMemory":
        mem = EpisodicMemory(int(payload.get("maxlen", 1200)), int(payload.get("k", 6)))
        mem.t = int(payload.get("t", 0))
        for e in payload.get("episodes", []):
            mem.episodes.append(Episode(e["key"], e["x"], e.get("basic"), e.get("affect"),
                                        int(e.get("stage", 0)), e.get("text_hash", ""),
                                        int(e.get("t", 0))))
        return mem


# ---------------------------------------------------------------------------
# 8. Plastic cortex: recursive least squares
# ---------------------------------------------------------------------------

class PlasticCortex:
    """
    Online ridge regression / Kalman-style last layer.

    A_new = gamma * A + x x^T
    P_new = A_new^{-1} = (1/gamma) [P - P x x^T P / (gamma + x^T P x)]
    M_new = M + (P x / (gamma + x^T P x)) (y - M^T x)

    gamma < 1 is a forgetting factor: the young brain stays plastic, and sleep
    replay restores what would otherwise fade.
    """

    def __init__(self, d: int = FEATURE_DIM, p: int = N_OUTPUT,
                 lam: float = 1.0, gamma: float = 1.0) -> None:
        self.d = d
        self.p = p
        self.lam = lam
        self.gamma = gamma
        self.P = [[0.0] * d for _ in range(d)]
        for i in range(d):
            self.P[i][i] = 1.0 / lam
        self.M = [[0.0] * p for _ in range(d)]
        self.n = 0
        self.sigma2 = 0.02

    def predict(self, x: Sequence[float]) -> list[float]:
        out = [0.0] * self.p
        for i, xi in enumerate(x):
            if xi:
                Mi = self.M[i]
                for j in range(self.p):
                    out[j] += xi * Mi[j]
        return out

    def update(self, x: Sequence[float], y: Sequence[float]) -> list[float]:
        d, p, g = self.d, self.p, self.gamma
        P = self.P
        Px = [sum(P[i][j] * x[j] for j in range(d)) for i in range(d)]
        denom = g + sum(x[i] * Px[i] for i in range(d))
        if denom < 1e-12:
            denom = 1e-12
        k = [v / denom for v in Px]
        pred = self.predict(x)
        err = [y[j] - pred[j] for j in range(p)]
        for i in range(d):
            ki = k[i]
            if ki == 0.0:
                continue
            Mi = self.M[i]
            Pi = P[i]
            for j in range(p):
                Mi[j] += ki * err[j]
            for j in range(d):
                Pi[j] = (Pi[j] - ki * Px[j]) / g
        self.n += 1
        # running residual variance for predictive error bars
        rv = sum(e * e for e in err) / p
        self.sigma2 = 0.995 * self.sigma2 + 0.005 * rv
        if self.n % 128 == 0:
            self._symmetrise()
        return err

    def _symmetrise(self) -> None:
        P = self.P
        for i in range(self.d):
            for j in range(i + 1, self.d):
                s = 0.5 * (P[i][j] + P[j][i])
                P[i][j] = P[j][i] = s

    def variance(self, x: Sequence[float]) -> float:
        P = self.P
        d = self.d
        Px = [sum(P[i][j] * x[j] for j in range(d)) for i in range(d)]
        return max(0.0, self.sigma2 * (1.0 + sum(x[i] * Px[i] for i in range(d))))

    def to_json(self) -> dict:
        return {"d": self.d, "p": self.p, "lam": self.lam, "gamma": self.gamma,
                "n": self.n, "sigma2": self.sigma2, "M": self.M, "P": self.P}

    @staticmethod
    def from_json(payload: dict) -> "PlasticCortex":
        c = PlasticCortex(int(payload["d"]), int(payload["p"]),
                          float(payload.get("lam", 1.0)), float(payload.get("gamma", 1.0)))
        c.n = int(payload.get("n", 0))
        c.sigma2 = float(payload.get("sigma2", 0.02))
        c.M = payload["M"]
        c.P = payload["P"]
        return c


# ---------------------------------------------------------------------------
# 9. Metacognition: "do I know that I know?"
# ---------------------------------------------------------------------------

class Metacognition:
    """Online logistic model of whether the current answer will be correct."""

    def __init__(self, dim: int = 9, lr: float = 0.08, l2: float = 1e-4) -> None:
        self.w = [0.0] * dim
        self.lr = lr
        self.l2 = l2
        self.n = 0

    def p_correct(self, meta: Sequence[float]) -> float:
        return sigmoid(dot(self.w, meta))

    def update(self, meta: Sequence[float], correct: bool) -> None:
        y = 1.0 if correct else 0.0
        p = self.p_correct(meta)
        for i in range(len(self.w)):
            self.w[i] += self.lr * ((y - p) * meta[i] - self.l2 * self.w[i])
        self.n += 1

    def to_json(self) -> dict:
        return {"w": self.w, "lr": self.lr, "l2": self.l2, "n": self.n}

    @staticmethod
    def from_json(payload: dict) -> "Metacognition":
        m = Metacognition(len(payload["w"]), float(payload.get("lr", 0.08)),
                          float(payload.get("l2", 1e-4)))
        m.w = list(payload["w"])
        m.n = int(payload.get("n", 0))
        return m


# ---------------------------------------------------------------------------
# 10. Prediction record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Prediction:
    raw_logits: list[float]
    probs: list[float]
    basic: int
    affect: list[float]
    confidence: float
    entropy: float
    epistemic: float
    mem_basic: list[float]
    mem_affect: list[float]
    mem_sim: float
    mem_count: float
    meta: list[float]
    p_correct: float


# ---------------------------------------------------------------------------
# 11. The child
# ---------------------------------------------------------------------------

class ChildBrain:
    def __init__(self, engine: AffectiveEngine, caregiver: Caregiver,
                 seed: int = CHILD_SEED, memory: bool = True) -> None:
        self.engine = engine
        self.caregiver = caregiver
        self.rng = SplitMix64(seed)
        self.cortex = PlasticCortex()
        self.memory = EpisodicMemory() if memory else None
        self.metacog = Metacognition()
        self.T_cal = 1.0
        self.prior_weight = 0.60  # innate reflex prior on the basic-emotion logits
        self.answer_threshold = 0.5
        self.stage = 0
        self.history: list[dict] = []
        self.approval_sum = 0.0
        self.approval_n = 0

    # -- features ---------------------------------------------------------
    def _features(self, p: Perception,
                  mem: tuple[list[float], list[float], float, float]) -> list[float]:
        mem_basic, mem_affect, mem_sim, mem_count = mem
        x = [1.0]
        x += p.basic_prior
        x += p.innate_affect
        x += p.skip[:32]
        x += [min(1.0, p.evidence / 3.0), p.magnitude]
        x += p.hidden
        x += mem_basic
        x += mem_affect
        x += [mem_sim, mem_count]
        return x

    def _memory_features(self, p: Perception) -> tuple[list[float], list[float], float, float]:
        if self.memory is None:
            return [0.0] * N_BASIC, [0.0] * ND, 0.0, 0.0
        return self.memory.retrieve(p.key)

    # -- inference --------------------------------------------------------
    def predict(self, p: Perception, use_memory: bool = True) -> Prediction:
        if use_memory and self.memory is not None:
            mem = self.memory.retrieve(p.key)
        else:
            mem = ([0.0] * N_BASIC, [0.0] * ND, 0.0, 0.0)
        x = self._features(p, mem)
        mu = self.cortex.predict(x)
        # The newborn starts with the v5 reflex, not a blank cortex.  The
        # plastic layer learns a residual correction on top of that prior.
        raw = [mu[j] + self.prior_weight * math.log(p.basic_prior[j] + 1e-9)
               for j in range(N_BASIC)]
        probs = softmax([v / max(self.T_cal, 1e-6) for v in raw])
        basic = argmax(probs)
        affect = [clip(v) for v in mu[N_BASIC:]]
        conf = max(probs)
        ent = entropy(probs)
        epi = math.sqrt(max(0.0, self.cortex.variance(x)))
        meta = [
            1.0,
            conf,
            ent / math.log(N_BASIC),
            min(1.0, p.evidence / 3.0),
            max(p.basic_prior),
            mem[2],
            mem[3],
            min(1.0, epi / 0.5),
            1.0 if argmax(p.basic_prior) == basic else 0.0,
        ]
        return Prediction(raw, probs, basic, affect, conf, ent, epi,
                          mem[0], mem[1], mem[2], mem[3], meta,
                          self.metacog.p_correct(meta))

    # -- learning ---------------------------------------------------------
    def learn(self, p: Perception, label: Label, pred: Prediction | None = None,
              weight: float = 1.0) -> None:
        if pred is None:
            pred = self.predict(p)
        mem = (pred.mem_basic, pred.mem_affect, pred.mem_sim, pred.mem_count)
        x = self._features(p, mem)
        y = onehot(label.basic, N_BASIC, smooth=0.05) + label.affect
        if weight != 1.0:
            x = [v * math.sqrt(weight) for v in x]
            y = [v * math.sqrt(weight) for v in y]
        self.cortex.update(x, y)
        if self.memory is not None:
            self.memory.add(p.key, x, label.basic, label.affect, self.stage,
                            text_hash(p.text))

    def observe(self, p: Perception) -> None:
        """An unlabelled experience still becomes part of episodic memory."""
        if self.memory is not None:
            x = self._features(p, ([0.0] * N_BASIC, [0.0] * ND, 0.0, 0.0))
            self.memory.add(p.key, x, None, None, self.stage, text_hash(p.text))

    # -- consolidation ----------------------------------------------------
    def sleep(self, epochs: int = 2, cap: int = 320) -> dict:
        if self.memory is None or epochs <= 0:
            return {"epochs": 0, "replayed": 0}
        episodes = self.memory.labelled()
        if not episodes:
            return {"epochs": 0, "replayed": 0}
        shuffle(episodes, self.rng)
        episodes = episodes[:cap]
        replayed = 0
        for _ in range(epochs):
            for ep in episodes:
                y = onehot(ep.basic, N_BASIC, smooth=0.05) + ep.affect  # type: ignore[arg-type]
                self.cortex.update(ep.x, y)
                replayed += 1
        return {"epochs": epochs, "replayed": replayed, "pool": len(episodes)}

    def prune(self, fraction: float = 0.02) -> int:
        """Synaptic pruning: zero the least useful non-bias feature rows."""
        if fraction <= 0.0:
            return 0
        importance = []
        for i in range(1, self.cortex.d):
            importance.append((sum(abs(v) for v in self.cortex.M[i]), i))
        importance.sort()
        k = max(0, int(fraction * len(importance)))
        pruned = 0
        for _, i in importance[:k]:
            for j in range(self.cortex.d):
                self.cortex.P[i][j] = 0.0
                self.cortex.P[j][i] = 0.0
            self.cortex.M[i] = [0.0] * self.cortex.p
            pruned += 1
        return pruned

    # -- calibration ------------------------------------------------------
    def calibrate(self, perceptions: Sequence[Perception],
                  labels: Sequence[Label]) -> dict:
        raw = []
        truth = []
        pcorrect = []
        for p, lab in zip(perceptions, labels):
            pred = self.predict(p)
            raw.append(pred.raw_logits)
            truth.append(lab.basic)
            pcorrect.append(pred.p_correct)

        best_T, best_nll = 1.0, float("inf")
        for T in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.70, 2.00, 2.50,
                  3.00, 3.50, 4.00):
            nll = 0.0
            for logits, y in zip(raw, truth):
                p = softmax([v / T for v in logits])
                nll -= math.log(p[y] + 1e-12)
            nll /= len(raw)
            if nll < best_nll:
                best_T, best_nll = T, nll
        self.T_cal = best_T
        # abstain on the least confident 20% of the calibration set
        self.answer_threshold = clip(quantile(pcorrect, 0.20), 0.30, 0.85)

        before = ece([softmax(r) for r in raw], truth)
        after = ece([softmax([v / best_T for v in r]) for r in raw], truth)
        return {"T": round(best_T, 4), "nll": round(best_nll, 5),
                "ece_before": round(before, 4), "ece_after": round(after, 4),
                "answer_threshold": round(self.answer_threshold, 4)}

    # -- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "version": CHILD_VERSION,
            "stage": self.stage,
            "T_cal": self.T_cal,
            "answer_threshold": self.answer_threshold,
            "cortex": self.cortex.to_json(),
            "memory": None if self.memory is None else self.memory.to_json(),
            "metacog": self.metacog.to_json(),
            "regulator": (None if getattr(self, "regulator", None) is None
                          else self.regulator.to_json()),
            "history": self.history,
        }

    @staticmethod
    def from_json(engine: AffectiveEngine, caregiver: Caregiver,
                  payload: dict) -> "ChildBrain":
        child = ChildBrain(engine, caregiver,
                           memory=payload.get("memory") is not None)
        child.stage = int(payload.get("stage", 0))
        child.T_cal = float(payload.get("T_cal", 1.0))
        child.answer_threshold = float(payload.get("answer_threshold", 0.5))
        child.cortex = PlasticCortex.from_json(payload["cortex"])
        if payload.get("memory") is not None:
            child.memory = EpisodicMemory.from_json(payload["memory"])
        child.metacog = Metacognition.from_json(payload["metacog"])
        if payload.get("regulator") is not None:
            child.regulator = Regulator.from_json(payload["regulator"])  # type: ignore[attr-defined]
        child.history = payload.get("history", [])
        return child


# ---------------------------------------------------------------------------
# 12. Evaluation
# ---------------------------------------------------------------------------

def evaluate_child(child: ChildBrain, perceptions: Sequence[Perception],
                   caregiver: Caregiver) -> dict:
    if not perceptions:
        return {"n": 0}
    labels = [caregiver.label(p) for p in perceptions]
    preds = [child.predict(p) for p in perceptions]

    correct = [1 if pr.basic == lab.basic else 0 for pr, lab in zip(preds, labels)]
    acc = sum(correct) / len(correct)
    mse = sum((pr.affect[d] - lab.affect[d]) ** 2
              for pr, lab in zip(preds, labels) for d in range(ND)) / (len(preds) * ND)
    nll = -sum(math.log(pr.probs[lab.basic] + 1e-12)
               for pr, lab in zip(preds, labels)) / len(preds)
    conf = sum(pr.confidence for pr in preds) / len(preds)
    epi = sum(pr.epistemic for pr in preds) / len(preds)
    cal = ece([pr.probs for pr in preds], [lab.basic for lab in labels])

    answered = [i for i, pr in enumerate(preds) if pr.p_correct >= child.answer_threshold]
    coverage = len(answered) / len(preds)
    sel_acc = (sum(correct[i] for i in answered) / len(answered)) if answered else 0.0

    def acc_at(frac: float) -> float:
        k = max(1, int(round(frac * len(preds))))
        order = sorted(range(len(preds)), key=lambda i: -preds[i].p_correct)[:k]
        return sum(correct[i] for i in order) / len(order)

    innate_acc = sum(1 if argmax(p.basic_prior) == lab.basic else 0
                     for p, lab in zip(perceptions, labels)) / len(perceptions)
    innate_mse = sum((p.innate_affect[d] - lab.affect[d]) ** 2
                     for p, lab in zip(perceptions, labels) for d in range(ND)) / (len(perceptions) * ND)
    teacher_acc = sum(1 if lab.teacher_basic == lab.basic else 0 for lab in labels) / len(labels)
    mem_used = sum(1 for pr in preds if pr.mem_count > 0.0) / len(preds)

    return {
        "n": len(preds),
        "basic_acc": round(acc, 4),
        "affect_mse": round(mse, 5),
        "nll": round(nll, 4),
        "ece": round(cal, 4),
        "mean_confidence": round(conf, 4),
        "mean_epistemic": round(epi, 4),
        "coverage": round(coverage, 4),
        "selective_acc": round(sel_acc, 4),
        "acc_at_50": round(acc_at(0.50), 4),
        "acc_at_70": round(acc_at(0.70), 4),
        "acc_at_90": round(acc_at(0.90), 4),
        "innate_basic_acc": round(innate_acc, 4),
        "innate_affect_mse": round(innate_mse, 5),
        "teacher_mapping_acc": round(teacher_acc, 4),
        "memory_used": round(mem_used, 4),
    }


def _online_curve(child: ChildBrain, perceptions: Sequence[Perception],
                  labels: Sequence[Label], chunk: int = 50) -> list[dict]:
    """Accuracy on the curriculum as it unfolds (reporting only)."""
    curve = []
    correct = 0
    for i, (p, lab) in enumerate(zip(perceptions, labels), 1):
        pred = child.predict(p)
        correct += 1 if pred.basic == lab.basic else 0
        if i % chunk == 0 or i == len(perceptions):
            curve.append({"seen": i, "acc": round(correct / i, 4)})
    return curve


def _probe(child: ChildBrain, perceptions: Sequence[Perception],
           labels: Sequence[Label]) -> tuple[float, float]:
    hits = 0
    mse = 0.0
    for p, lab in zip(perceptions, labels):
        pr = child.predict(p)
        hits += 1 if pr.basic == lab.basic else 0
        mse += sum((pr.affect[d] - lab.affect[d]) ** 2 for d in range(ND)) / ND
    n = len(perceptions)
    return round(hits / n, 4), round(mse / n, 5)


# ---------------------------------------------------------------------------
# 13. Development
# ---------------------------------------------------------------------------

def develop(engine: AffectiveEngine, caregiver: Caregiver,
            stages: Sequence[Stage] = STAGES, seed: int = CHILD_SEED,
            policy: str = "uncertainty", sleep_enabled: bool = True,
            memory_enabled: bool = True, log=print) -> tuple[ChildBrain, dict]:
    child = ChildBrain(engine, caregiver, seed=seed, memory=memory_enabled)
    rng = SplitMix64(seed ^ 0xDE7E10)
    t0 = time.time()

    # Fixed milestone sets, one per stage, generated before any learning.
    milestone_rng = SplitMix64(seed ^ 0x1A2B3C)
    milestones: dict[int, tuple[list[Perception], list[Label]]] = {}
    for st in stages:
        ps = generate_perceptions(engine, st, st.n_test, milestone_rng)
        milestones[st.idx] = (ps, [caregiver.label(p) for p in ps])

    retention: dict[str, dict[int, dict]] = {}
    stage_reports: list[dict] = []

    for st in stages:
        child.stage = st.idx
        child.cortex.gamma = st.gamma
        stage_t0 = time.time()
        train_rng = SplitMix64(seed ^ (0x51A6E + st.idx * 7919))
        train = generate_perceptions(engine, st, st.n_train, train_rng)
        train_labels = [caregiver.label(p) for p in train]
        # Caregiver pedagogy: simple scenes first, then longer and more
        # negated / intensified ones.  A stable sort preserves ties.
        order = sorted(range(len(train)), key=lambda i: difficulty(train[i]))
        train = [train[i] for i in order]
        train_labels = [train_labels[i] for i in order]

        # ---- newborn: reflexes only ------------------------------------
        if not st.learn:
            saved_memory = child.memory
            if not st.memory:
                child.memory = None
            metrics = evaluate_child(child, milestones[st.idx][0], caregiver)
            child.memory = saved_memory
            stage_reports.append({
                "stage": st.idx, "name": st.name, "age": st.age,
                "description": st.description, "learned": False,
                "experiences": len(train), "asked": 0, "supervised": 0,
                "corrected": 0, "answered": 0,
                "approval": None, "calibration": None,
                "sleep": {"epochs": 0, "replayed": 0}, "pruned_features": 0,
                "metrics": metrics, "learning_curve": [], "probe_curve": [],
                "seconds": round(time.time() - stage_t0, 2),
            })
            log(f"[{st.idx} {st.name:10s}] reflexes only  acc={metrics['basic_acc']:.3f}  "
                f"innate={metrics['innate_basic_acc']:.3f}  mse={metrics['affect_mse']:.4f}")
            continue

        # ---- online childhood ------------------------------------------
        budget = int(st.n_train * st.curiosity)
        asked = 0
        supervised = 0
        corrected = 0
        answered = 0
        approval_sum = 0.0
        feedback_n = 0
        curve: list[dict] = []
        online_correct = 0

        # Fixed probe: the same held-out experiences are re-read at intervals,
        # which makes the developmental curve visible rather than confounded
        # by the changing difficulty of the curriculum.
        probe_ps = milestones[st.idx][0][:40]
        probe_labels = [caregiver.label(p) for p in probe_ps]
        probe_curve: list[dict] = []
        p_acc, p_mse = _probe(child, probe_ps, probe_labels)
        probe_curve.append({"seen": 0, "acc": p_acc, "affect_mse": p_mse})

        for i, (p, lab) in enumerate(zip(train, train_labels), 1):
            pred = child.predict(p)
            correct = pred.basic == lab.basic
            online_correct += 1 if correct else 0
            if i % 50 == 0 or i == len(train):
                curve.append({"seen": i, "acc": round(online_correct / i, 4)})
                if i % 100 == 0 or i == len(train):
                    p_acc, p_mse = _probe(child, probe_ps, probe_labels)
                    probe_curve.append({"seen": i, "acc": p_acc, "affect_mse": p_mse})
            is_supervised = rng.unit() < st.supervision
            ask = False
            if not is_supervised and budget > 0:
                if policy == "uncertainty":
                    score = 0.6 * (pred.entropy / math.log(N_BASIC)) + 0.4 * min(1.0, pred.epistemic / 0.5)
                    ask = score > 0.55
                elif policy == "random":
                    ask = rng.unit() < 0.30
                if ask:
                    budget -= 1
                    asked += 1
            answered_here = pred.p_correct >= child.answer_threshold
            if answered_here:
                answered += 1

            reveal = is_supervised or ask
            # A confident mistake is corrected by the caregiver.
            if not reveal and answered_here and not correct and rng.unit() < 0.80:
                reveal = True
                corrected += 1
            if is_supervised:
                supervised += 1

            if reveal:
                child.learn(p, lab, pred)
                child.metacog.update(pred.meta, correct)
                approval = caregiver.approval(answered_here, ask, correct)
                approval_sum += approval
                feedback_n += 1
            else:
                child.observe(p)
                # Only answered decisions receive explicit approval.
                if answered_here:
                    approval = caregiver.approval(True, False, correct)
                    approval_sum += approval
                    feedback_n += 1
                    child.metacog.update(pred.meta, correct)

        # ---- sleep ------------------------------------------------------
        sleep_stats = {"epochs": 0, "replayed": 0}
        pruned = 0
        if sleep_enabled:
            sleep_stats = child.sleep(st.sleep_epochs)
            if st.idx >= 3:
                pruned = child.prune(0.015)
        # ---- calibration on a fresh, labelled sample --------------------
        cal_rng = SplitMix64(seed ^ (0xCA11B + st.idx * 104729))
        cal = generate_perceptions(engine, st, 240, cal_rng)
        cal_labels = [caregiver.label(p) for p in cal]
        calibration = child.calibrate(cal, cal_labels)

        # ---- evaluation -------------------------------------------------
        metrics = evaluate_child(child, milestones[st.idx][0], caregiver)

        # retention on every earlier milestone set
        retention[str(st.idx)] = {}
        for prev in stages:
            if prev.idx >= st.idx:
                continue
            retention[str(st.idx)][prev.idx] = evaluate_child(
                child, milestones[prev.idx][0], caregiver)

        report = {
            "stage": st.idx, "name": st.name, "age": st.age,
            "description": st.description, "learned": True,
            "experiences": len(train), "supervised": supervised,
            "asked": asked, "corrected": corrected, "answered": answered,
            "approval": round(approval_sum / feedback_n, 4) if feedback_n else None,
            "sleep": sleep_stats, "pruned_features": pruned,
            "calibration": calibration, "metrics": metrics,
            "learning_curve": curve, "probe_curve": probe_curve,
            "seconds": round(time.time() - stage_t0, 2),
        }
        stage_reports.append(report)
        log(f"[{st.idx} {st.name:10s}] acc={metrics['basic_acc']:.3f}  "
            f"innate={metrics['innate_basic_acc']:.3f}  teacher={metrics['teacher_mapping_acc']:.3f}  "
            f"mse={metrics['affect_mse']:.4f}  ece={metrics['ece']:.3f}  "
            f"sel@{metrics['coverage']:.2f}={metrics['selective_acc']:.3f}  "
            f"asked={asked}  sleep={sleep_stats['replayed']}")

    total_seconds = round(time.time() - t0, 2)
    report = {
        "version": CHILD_VERSION,
        "seed": seed,
        "caregiver_dialect": {
            "bias": [round(v, 5) for v in caregiver.bias],
            "gain": [round(v, 5) for v in caregiver.gain],
            "offset": [round(v, 5) for v in caregiver.offset],
        },
        "stages": stage_reports,
        "retention": retention,
        "total_seconds": total_seconds,
        "policy": policy,
        "sleep_enabled": sleep_enabled,
        "memory_enabled": memory_enabled,
    }
    child.history = stage_reports
    return child, report


# ---------------------------------------------------------------------------
# 14. Self-regulation: choosing a coping action
# ---------------------------------------------------------------------------

COPING_ACTIONS: Final[tuple[str, ...]] = ("none", "breathe", "reframe", "seek_support", "rest", "approach")

# Effect on the ten affect dimensions:
# valence, arousal, control, novelty, social, threat, clarity, engagement, tension, resolve
COPING_DELTA: Final[dict[str, tuple[float, ...]]] = {
    "none":         ( 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00),
    "breathe":      ( 0.06, -0.28,  0.16, -0.04,  0.00, -0.30,  0.12, -0.06, -0.24,  0.05),
    "reframe":      ( 0.10, -0.10,  0.24, -0.02,  0.00, -0.18,  0.26,  0.06, -0.16,  0.14),
    "seek_support": ( 0.16, -0.06,  0.12,  0.00,  0.32, -0.14,  0.14,  0.08, -0.14,  0.10),
    "rest":         ( 0.08, -0.32,  0.08, -0.06,  0.00, -0.20,  0.10, -0.18, -0.26, -0.04),
    "approach":     ( 0.04,  0.12,  0.26,  0.06,  0.00, -0.06,  0.12,  0.24,  0.06,  0.26),
}


class Regulator:
    """Linear Q-value model over [affect, basic emotion, action one-hot]."""

    def __init__(self) -> None:
        n_act = len(COPING_ACTIONS)
        # Interact each action with the state, so the child can learn
        # "breathe when threat is high" rather than one fixed policy.
        d = 1 + n_act + n_act * (ND + N_BASIC)
        self.cortex = PlasticCortex(d=d, p=1, lam=1.0, gamma=1.0)

    @staticmethod
    def features(affect: Sequence[float], basic: Sequence[float], action: int) -> list[float]:
        n_act = len(COPING_ACTIONS)
        x = [1.0] + [0.0] * n_act
        x[1 + action] = 1.0
        for a in range(n_act):
            if a == action:
                x += list(affect) + list(basic)
            else:
                x += [0.0] * (ND + N_BASIC)
        return x

    @staticmethod
    def reward(post: Sequence[float]) -> float:
        valence, arousal, control, _, social, threat, clarity, engagement, tension, _ = post
        distress = max(0.0, -valence) + 0.6 * max(0.0, threat) + 0.3 * max(0.0, tension)
        wellbeing = (0.30 * max(0.0, valence) + 0.20 * control + 0.15 * clarity
                     + 0.10 * max(0.0, engagement) + 0.10 * max(0.0, social))
        return wellbeing - distress - 0.10 * max(0.0, arousal - 0.50)

    def train(self, examples: Sequence[tuple[list[float], list[float]]], epochs: int = 2) -> int:
        n = 0
        for _ in range(epochs):
            for affect, basic in examples:
                for a in range(len(COPING_ACTIONS)):
                    post = [clip(affect[d] + COPING_DELTA[COPING_ACTIONS[a]][d])
                            for d in range(ND)]
                    x = self.features(affect, basic, a)
                    self.cortex.update(x, [self.reward(post)])
                    n += 1
        return n

    def best_action(self, affect: Sequence[float], basic: Sequence[float]) -> tuple[int, list[float]]:
        qs = [self.cortex.predict(self.features(affect, basic, a))[0]
              for a in range(len(COPING_ACTIONS))]
        a = argmax(qs)
        post = [clip(affect[d] + COPING_DELTA[COPING_ACTIONS[a]][d]) for d in range(ND)]
        return a, qs

    def to_json(self) -> dict:
        return {"cortex": self.cortex.to_json()}

    @staticmethod
    def from_json(payload: dict) -> "Regulator":
        r = Regulator()
        r.cortex = PlasticCortex.from_json(payload["cortex"])
        return r


def train_regulator(child: ChildBrain, perceptions: Sequence[Perception],
                    caregiver: Caregiver) -> tuple[Regulator, dict]:
    reg = Regulator()
    split = max(1, int(len(perceptions) * 0.75))
    train_set = perceptions[:split]
    examples = []
    for p in train_set:
        lab = caregiver.label(p)
        examples.append((lab.affect, onehot(lab.basic, N_BASIC)))
    reg.train(examples, epochs=4)

    # evaluate on held-out experiences
    test = perceptions[split:]
    chosen_reward = 0.0
    random_reward = 0.0
    best_fixed = {a: 0.0 for a in range(len(COPING_ACTIONS))}
    oracle_reward = 0.0
    rng = SplitMix64(CHILD_SEED ^ 0x5E1F)
    for p in test:
        lab = caregiver.label(p)
        a, _ = reg.best_action(lab.affect, onehot(lab.basic, N_BASIC))
        post = [clip(lab.affect[d] + COPING_DELTA[COPING_ACTIONS[a]][d]) for d in range(ND)]
        chosen_reward += Regulator.reward(post)
        ra = rng.below(len(COPING_ACTIONS))
        rpost = [clip(lab.affect[d] + COPING_DELTA[COPING_ACTIONS[ra]][d]) for d in range(ND)]
        random_reward += Regulator.reward(rpost)
        for aa in range(len(COPING_ACTIONS)):
            apost = [clip(lab.affect[d] + COPING_DELTA[COPING_ACTIONS[aa]][d]) for d in range(ND)]
            best_fixed[aa] += Regulator.reward(apost)
        oracle_reward += max(Regulator.reward([clip(lab.affect[d] + COPING_DELTA[COPING_ACTIONS[aa]][d])
                                                for d in range(ND)])
                             for aa in range(len(COPING_ACTIONS)))
    n = len(test)
    metrics = {
        "n": n,
        "chosen_reward": round(chosen_reward / n, 5),
        "random_reward": round(random_reward / n, 5),
        "oracle_reward": round(oracle_reward / n, 5),
        "best_fixed": {COPING_ACTIONS[a]: round(v / n, 5) for a, v in best_fixed.items()},
    }
    return reg, metrics


# ---------------------------------------------------------------------------
# 15. Ablations
# ---------------------------------------------------------------------------

def run_ablation(engine: AffectiveEngine, caregiver: Caregiver,
                 seeds: Sequence[int] = (CHILD_SEED ^ 0xA11, CHILD_SEED ^ 0xB22),
                 log=print) -> dict:
    """Paired, multi-seed developmental runs under four conditions.

    The curriculum is deliberately short (three stages, larger samples than a
    smoke test).  Each condition sees exactly the same seeds and stimuli.
    """
    short = tuple(
        Stage(s.idx, s.name, s.age, s.description,
              max(180, s.n_train // 2), max(120, s.n_test),
              s.supervision, s.curiosity, s.sleep_epochs, s.gamma,
              s.max_scenes, s.max_cats, s.neg_p, s.int_p, s.cats,
              s.learn, s.memory, s.novelty)
        for s in STAGES[1:4]  # infant, toddler, child
    )
    conditions = {
        "full": dict(policy="uncertainty", sleep_enabled=True, memory_enabled=True),
        "no_memory": dict(policy="uncertainty", sleep_enabled=True, memory_enabled=False),
        "no_sleep": dict(policy="uncertainty", sleep_enabled=False, memory_enabled=True),
        "random_curiosity": dict(policy="random", sleep_enabled=True, memory_enabled=True),
    }
    out: dict[str, dict] = {}
    for name, cfg in conditions.items():
        log(f"[ablation] {name} ...")
        rows = []
        for seed in seeds:
            _, report = develop(engine, caregiver, stages=short, seed=seed,
                                log=lambda *a: None, **cfg)
            last = report["stages"][-1]["metrics"]
            retention = report["retention"].get(str(short[-1].idx), {})
            first = short[0].idx
            retained = retention.get(str(first), retention.get(first, {}))
            asked = sum(s.get("asked", 0) for s in report["stages"])
            labels = sum(s.get("supervised", 0) + s.get("asked", 0) + s.get("corrected", 0)
                         for s in report["stages"])
            rows.append({
                "final_acc": last["basic_acc"],
                "final_affect_mse": last["affect_mse"],
                "final_ece": last["ece"],
                "retention_infant_acc": retained.get("basic_acc", 0.0),
                "asked": asked,
                "labels": labels,
                "seconds": report["total_seconds"],
            })
        n = len(rows)
        out[name] = {
            "final_acc": round(sum(r["final_acc"] for r in rows) / n, 4),
            "final_affect_mse": round(sum(r["final_affect_mse"] for r in rows) / n, 6),
            "final_ece": round(sum(r["final_ece"] for r in rows) / n, 4),
            "retention_infant_acc": round(sum(r["retention_infant_acc"] for r in rows) / n, 4),
            "asked": round(sum(r["asked"] for r in rows) / n, 2),
            "labels": round(sum(r["labels"] for r in rows) / n, 2),
            "acc_per_label": round(sum(r["final_acc"] for r in rows) / max(1.0, sum(r["labels"] for r in rows)), 6),
            "acc_per_question": round(sum(r["final_acc"] for r in rows) / max(1.0, sum(r["asked"] for r in rows)), 6),
            "seeds": n,
            "seconds": round(sum(r["seconds"] for r in rows), 2),
        }
        log(f"[ablation] {name:16s} final={out[name]['final_acc']:.3f}  "
            f"retention={out[name]['retention_infant_acc']:.3f}  "
            f"asked={out[name]['asked']:.1f}  labels={out[name]['labels']:.1f}")
    return out


# ---------------------------------------------------------------------------
# 16. Milestones
# ---------------------------------------------------------------------------

def milestones(report: dict, ablations: dict | None, regulation: dict | None) -> list[dict]:
    stages = [s for s in report["stages"] if s.get("learned")]
    if not stages:
        return []
    final = stages[-1]["metrics"]
    newborn = next((s for s in report["stages"] if not s.get("learned")), stages[0])
    baseline = newborn["metrics"]["basic_acc"]
    out = [
        {"name": "recognises basic emotions", "ok": final["basic_acc"] > 0.40,
         "value": final["basic_acc"], "target": "> 0.40"},
        {"name": "improves on innate temperament", "ok": final["basic_acc"] > baseline + 0.03,
         "value": round(final["basic_acc"] - baseline, 4), "target": "> +0.03 over newborn reflex"},
        {"name": "learns the family dialect", "ok": final["basic_acc"] >= final["teacher_mapping_acc"] - 0.02,
         "value": round(final["basic_acc"] - final["teacher_mapping_acc"], 4), "target": ">= -0.02 vs textbook mapping"},
        {"name": "knows when it does not know", "ok": final["acc_at_50"] > final["basic_acc"] + 0.08,
         "value": round(final["acc_at_50"] - final["basic_acc"], 4), "target": "> +0.08 at 50% coverage"},
        {"name": "is calibrated", "ok": final["ece"] < 0.10,
         "value": final["ece"], "target": "< 0.10"},
        {"name": "remembers earlier lessons", "ok": _retention_ok(report, stages),
         "value": _retention_value(report, stages), "target": "> 0.70 of first-stage accuracy"},
        {"name": "asks useful questions",
         "ok": (_ablation_ratio(ablations, "random_curiosity", "acc_per_question") or 0.0) > 1.0,
         "value": _ablation_ratio(ablations, "random_curiosity", "acc_per_question"),
         "target": "> 1x accuracy per question vs random"},
        {"name": "memory supports recall", "ok": _ablation_ok(ablations, "no_memory", "retention_infant_acc"),
         "value": _ablation_value(ablations, "no_memory", "retention_infant_acc"),
         "target": "memory > no-memory retention"},
        {"name": "sleep consolidates", "ok": _ablation_ok(ablations, "no_sleep", "retention_infant_acc"),
         "value": _ablation_value(ablations, "no_sleep", "retention_infant_acc"), "target": "sleep > no-sleep retention"},
        {"name": "regulates affect", "ok": regulation is not None and regulation["chosen_reward"] > regulation["random_reward"],
         "value": None if regulation is None else round(regulation["chosen_reward"] - regulation["random_reward"], 5),
         "target": "chosen > random reward"},
    ]
    return out


def _retention_value(report: dict, stages: list[dict]) -> float | None:
    first = stages[0]["stage"]
    last = stages[-1]["stage"]
    row = report["retention"].get(str(last), {})
    r = row.get(str(first), row.get(first))
    if not r:
        return None
    base_row = next((s for s in report["stages"] if s.get("stage") == first), None)
    if base_row is None:
        return None
    base = base_row["metrics"]["basic_acc"]
    return round(r["basic_acc"] / base, 4) if base else None


def _retention_ok(report: dict, stages: list[dict]) -> bool:
    v = _retention_value(report, stages)
    return v is not None and v > 0.70


def _ablation_value(ablations: dict | None, condition: str, key: str):
    if not ablations or condition not in ablations:
        return None
    base = ablations.get("full", {}).get(key)
    other = ablations[condition].get(key)
    return None if base is None or other is None else round(base - other, 4)


def _ablation_ok(ablations: dict | None, condition: str, key: str) -> bool:
    v = _ablation_value(ablations, condition, key)
    return v is not None and v > 0.0


def _ablation_ratio(ablations: dict | None, condition: str, key: str):
    if not ablations:
        return None
    base = ablations.get("full", {}).get(key)
    other = ablations.get(condition, {}).get(key)
    if not base or not other:
        return None
    return round(base / other, 2)


# ---------------------------------------------------------------------------
# 17. Demo and CLI
# ---------------------------------------------------------------------------

DEMO_TEXTS: Final[tuple[str, ...]] = (
    "The GPS dies in the dark and I am completely alone on a broken trail.",
    "I am deeply grateful for the village hospitality and the warm meal they shared.",
    "A long-delayed permit is finally approved after months of waiting.",
    "Discovering an ancient, hidden ruin nestled deep inside a dense forest.",
    "Sitting quietly in a modest rented room, reviewing field notes and journaling.",
    "A rockfall blocks the only path out and the light is failing.",
    "Quartz vectors traverse the amber lattice.",
)


def verify_child(child: ChildBrain) -> dict:
    """Structural checks on a saved child."""
    mem_ok = True
    if child.memory is not None:
        mem_ok = all(len(e.x) == FEATURE_DIM and len(e.key) == MEMORY_KEY_DIM
                     for e in child.memory.episodes)
    reg_ok = True
    reg = getattr(child, "regulator", None)
    if reg is not None:
        expected = 1 + len(COPING_ACTIONS) + len(COPING_ACTIONS) * (ND + N_BASIC)
        reg_ok = reg.cortex.d == expected and reg.cortex.p == 1
    probe = "The GPS dies in the dark and I am completely alone on a broken trail."
    p = perceive(child.engine, probe)
    deterministic = p is not None
    if p is not None:
        a = child.predict(p)
        b = child.predict(p)
        deterministic = (a.basic == b.basic and a.probs == b.probs
                         and a.affect == b.affect)
    return {
        "version": CHILD_VERSION,
        "cortex_dim": child.cortex.d,
        "cortex_outputs": child.cortex.p,
        "feature_dim_expected": FEATURE_DIM,
        "outputs_expected": N_OUTPUT,
        "cortex_dim_ok": child.cortex.d == FEATURE_DIM and child.cortex.p == N_OUTPUT,
        "metacognition_dim": len(child.metacog.w),
        "metacognition_dim_ok": len(child.metacog.w) == 9,
        "memory_episodes": 0 if child.memory is None else len(child.memory.episodes),
        "memory_dim_ok": mem_ok,
        "regulator_ok": reg_ok,
        "deterministic_prediction": deterministic,
        "stage": child.stage,
        "temperature": child.T_cal,
        "answer_threshold": child.answer_threshold,
    }


def demo(child: ChildBrain, texts: Sequence[str]) -> list[dict]:
    out = []
    for text in texts:
        p = perceive(child.engine, text)
        if p is None:
            out.append({"text": text, "state": "undetermined", "abstained": True})
            continue
        pred = child.predict(p)
        lab = child.caregiver.label(p)
        answered = pred.p_correct >= child.answer_threshold
        action, qs = child.regulator.best_action(pred.affect, onehot(pred.basic, N_BASIC)) \
            if getattr(child, "regulator", None) is not None else (-1, [])
        post = None
        if action >= 0:
            post = {AFFECT_DIMENSIONS[d]: round(clip(pred.affect[d] + COPING_DELTA[COPING_ACTIONS[action]][d]), 4)
                    for d in range(ND)}
        # The v5 core supplies the child's expressive vocabulary: the felt
        # state, the behaviour it affords, and the private thought.
        try:
            full = child.engine.evaluate(text)
            expression = {
                "mental_state": full.get("mental_state"),
                "behavior": full.get("behavior"),
                "thought": full.get("thought_process"),
            }
        except Exception:
            expression = None
        out.append({
            "text": text,
            "predicted_basic": BASIC_EMOTIONS[pred.basic],
            "caregiver_basic": BASIC_EMOTIONS[lab.basic],
            "correct": pred.basic == lab.basic,
            "confidence": round(pred.confidence, 4),
            "p_correct": round(pred.p_correct, 4),
            "decision": "answer" if answered else "ask_or_abstain",
            "affect": {AFFECT_DIMENSIONS[d]: round(pred.affect[d], 4) for d in range(ND)},
            "memory_similarity": round(pred.mem_sim, 4),
            "expression": expression,
            "coping_action": None if action < 0 else COPING_ACTIONS[action],
            "predicted_post_regulation_affect": post,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Child Brain v{CHILD_VERSION}")
    ap.add_argument("--develop", action="store_true", help="run the full developmental curriculum")
    ap.add_argument("--ablate", action="store_true", help="run memory/curiosity/sleep ablations")
    ap.add_argument("--demo", action="append", default=[], help="text to interpret (repeatable)")
    ap.add_argument("--teach", nargs=2, action="append", default=[],
                    metavar=("TEXT", "EMOTION"),
                    help="teach one labelled example (repeatable)")
    ap.add_argument("--report", action="store_true", help="print the saved report")
    ap.add_argument("--verify", action="store_true", help="check the saved checkpoint")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    ap.add_argument("--report-path", default=str(REPORT_PATH))
    ap.add_argument("--seed", type=int, default=CHILD_SEED)
    args = ap.parse_args(argv)

    engine = AffectiveEngine(cache=CACHE_PATH)
    caregiver = Caregiver.from_seed(args.seed ^ 0xF00D)

    if args.develop:
        child, report = develop(engine, caregiver, seed=args.seed)
        # train the regulator on a sample of lived experience
        reg_rng = SplitMix64(args.seed ^ 0x4E6)
        reg_stage = STAGES[-1]
        reg_ps = generate_perceptions(engine, reg_stage, 360, reg_rng)
        reg, reg_metrics = train_regulator(child, reg_ps, caregiver)
        child.regulator = reg  # type: ignore[attr-defined]
        report["regulation"] = reg_metrics
        report["ablations"] = run_ablation(engine, caregiver) if args.ablate else None
        report["milestones"] = milestones(report, report.get("ablations"), reg_metrics)
        report["demo"] = demo(child, DEMO_TEXTS)
        Path(args.checkpoint).write_text(json.dumps(child.to_json()))
        Path(args.report_path).write_text(json.dumps(report, indent=2))
        print(json.dumps({
            "checkpoint": args.checkpoint,
            "report": args.report_path,
            "final": report["stages"][-1]["metrics"],
            "milestones": report["milestones"],
            "seconds": report["total_seconds"],
        }, indent=2))
        return 0

    if args.verify:
        if not Path(args.checkpoint).exists():
            print("No checkpoint found. Run: python3 child_brain.py --develop", file=sys.stderr)
            return 1
        child = ChildBrain.from_json(engine, caregiver,
                                     json.loads(Path(args.checkpoint).read_text()))
        print(json.dumps(verify_child(child), indent=2))
        return 0

    if args.teach:
        if not Path(args.checkpoint).exists():
            print("No checkpoint found. Run: python3 child_brain.py --develop", file=sys.stderr)
            return 1
        child = ChildBrain.from_json(engine, caregiver,
                                     json.loads(Path(args.checkpoint).read_text()))
        for text, emotion in args.teach:
            emo = emotion.strip().lower()
            if emo not in BASIC_EMOTIONS:
                print(f"unknown emotion {emotion!r}; choose from {', '.join(BASIC_EMOTIONS)}",
                      file=sys.stderr)
                return 2
            p = perceive(engine, text)
            if p is None:
                print(json.dumps({"text": text, "taught": False,
                                  "reason": "not enough affective evidence"}))
                continue
            before = child.predict(p)
            base = child.caregiver.label(p)
            idx = BASIC_EMOTIONS.index(emo)
            label = Label(idx, list(p.innate_affect), onehot(idx, N_BASIC),
                          base.teacher_basic, base.teacher_basic_probs)
            child.learn(p, label, before)
            child.metacog.update(before.meta, before.basic == idx)
            after = child.predict(p)
            print(json.dumps({
                "text": text,
                "taught": emo,
                "before": BASIC_EMOTIONS[before.basic],
                "after": BASIC_EMOTIONS[after.basic],
                "after_confidence": round(after.confidence, 4),
                "memory_size": 0 if child.memory is None else len(child.memory.episodes),
            }))
        Path(args.checkpoint).write_text(json.dumps(child.to_json()))
        return 0

    if args.report:
        print(Path(args.report_path).read_text())
        return 0

    # demo / default: load checkpoint if present
    if Path(args.checkpoint).exists():
        child = ChildBrain.from_json(engine, caregiver,
                                     json.loads(Path(args.checkpoint).read_text()))
        texts = args.demo or list(DEMO_TEXTS)
        print(json.dumps(demo(child, texts), indent=2, ensure_ascii=False))
        return 0

    print("No checkpoint found. Run: python3 child_brain.py --develop", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
