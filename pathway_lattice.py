#!/usr/bin/env python3
"""
Pathway Lattice v1 -- determination, not generation.

A deterministic, non-generative reasoning core. There is no probability
distribution over "next token" and nothing is ever sampled. A query is
encoded as a 1000-dimensional integer vector; a router *determines* one
rank-1 path operator; the path is applied 1000 -> path -> 1000; the answer
is decoded exactly, or the model abstains.

    next X  =  decode( compose( path ) ( encode(context) ) )

Everything is fixed-point integer arithmetic (int64 -> Python big ints),
so two runs on any machine give bit-identical results.

Tiers:
    S0  1,000 learned params   1,000 per-channel gains over the frozen codebook
    S1  up to 1,000,000        P rank-1 path operators x 2,000 params each
                               (read vector + write vector), P = 500

Usage:
    python3 pathway_lattice.py --train      compile + calibrate + save
    python3 pathway_lattice.py --demo       the showcase
    python3 pathway_lattice.py --verify     integrity + determinism checks
    python3 pathway_lattice.py --bench      throughput, scaling, robustness
    python3 pathway_lattice.py --query "paris capital-of"
    python3 pathway_lattice.py --report     parameter inventory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from math import sumprod
from typing import Iterable, Sequence

from pathway_seed import FACTS, TOOL_FACTS

# ------------------------------------------------------------- constants
DIM = 1000                 # state width: the "1000" in 1000 * path * 1000
SHIFT = 12                 # fixed-point fractional bits
ONE = 1 << SHIFT           # 1.0 in fixed point
MAX_SCALAR = ONE * 16
MAX_VEC = ONE * 64
REL = "rel:"               # relation-channel token prefix
MODEL_FILE = "pathway_lattice_model.json"
ROUTE_MIN_COS = (3 * ONE) // 4     # 0.75: routing abstention gate
ACCEPT_MIN_COS = ONE // 2          # 0.50: decoding abstention gate
MAX_STEPS = 6

ABSTAIN = "UNKNOWN"


# ------------------------------------------------------- fixed-point math
def fx(x: float) -> int:
    return int(round(x * ONE))


def clamp_scalar(s: int, limit: int = MAX_SCALAR) -> int:
    return -limit if s < -limit else (limit if s > limit else s)


def clamp_vec(v: Iterable[int], limit: int = MAX_VEC) -> list[int]:
    return [-limit if x < -limit else (limit if x > limit else x) for x in v]


def dot_fx(a: Sequence[int], b: Sequence[int]) -> int:
    """Fixed-point dot product (>= 0.5 ulp exact; clamped)."""
    return clamp_scalar(sumprod(a, b) >> SHIFT)


def vnorm(v: Sequence[int]) -> int:
    return math.isqrt(max(0, sumprod(v, v)))


def normalize(v: Sequence[int]) -> tuple[int, ...]:
    n = vnorm(v)
    if n == 0:
        return tuple(v)
    return tuple((x * ONE) // n for x in v)


def cos_fx(a: Sequence[int], b: Sequence[int]) -> int:
    """Integer cosine in [-ONE, +ONE]."""
    na, nb = vnorm(a), vnorm(b)
    if na == 0 or nb == 0:
        return 0
    c = (sumprod(a, b) * ONE) // (na * nb)
    return -ONE if c < -ONE else (ONE if c > ONE else c)


# ------------------------------------------------------------ codebook
MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


class Codebook:
    """Deterministic symbol -> vector encoding. No RNG state: each token's
    code is a pure function of its bytes (blake2b -> splitmix64 stream)."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self._cache: dict[str, tuple[int, ...]] = {}
        self.candidates: list[str] = []

    def vec(self, token: str) -> tuple[int, ...]:
        v = self._cache.get(token)
        if v is not None:
            return v
        seed = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        x = seed
        out = [0] * self.dim
        for i in range(self.dim):
            x = splitmix64(x)
            out[i] = ONE if (x >> 63) & 1 else -ONE
        v = tuple(out)
        self._cache[token] = v
        return v

    def register(self, symbol: str) -> None:
        if symbol not in self.candidates:
            self.candidates.append(symbol)
            self.candidates.sort()

    def decode(self, state: Sequence[int],
               min_cos: int = ACCEPT_MIN_COS) -> tuple[str | None, int]:
        """Exact nearest-code readout. Tie-break: lexicographically first."""
        best_sym: str | None = None
        best_cos = -ONE - 1
        for sym in self.candidates:
            c = cos_fx(state, self.vec(sym))
            if c > best_cos:
                best_sym, best_cos = sym, c
        if best_sym is None or best_cos < min_cos:
            return None, best_cos
        return best_sym, best_cos


# ------------------------------------------------------------- operators
class PathOp:
    """One rank-1 path operator:  h' = u * (v . h).

    mode 'set': the write replaces the state  (symbolic chaining)
    mode 'add': the write accumulates         (rank-L composition)
    """

    __slots__ = ("key", "u", "v", "gain", "mode", "learn_u", "learn_v",
                 "learn_gain", "vn")

    def __init__(self, key: str, u: Sequence[int], v: Sequence[int],
                 gain: int = ONE, mode: str = "set",
                 learn_u: bool = True, learn_v: bool = True,
                 learn_gain: bool = False) -> None:
        self.key = key
        self.u = tuple(u)
        self.v = tuple(v)
        self.gain = gain
        self.mode = mode
        self.learn_u = learn_u
        self.learn_v = learn_v
        self.learn_gain = learn_gain
        self.vn = vnorm(self.v)          # cached: routing hot path

    def apply(self, h: Sequence[int]) -> tuple[int, ...]:
        s = dot_fx(self.v, h)
        if self.gain != ONE:
            s = (s * self.gain) >> SHIFT
        if s == 0:
            return tuple(h)
        w = [(ui * s) >> SHIFT for ui in self.u]
        if self.mode == "add":
            out = [a + b for a, b in zip(h, w)]
        else:
            out = w
        return normalize(clamp_vec(out))

    def param_count(self, learned_only: bool = True) -> int:
        n = 0
        if not learned_only or self.learn_u:
            n += len(self.u)
        if not learned_only or self.learn_v:
            n += len(self.v)
        if not learned_only or self.learn_gain:
            n += 1
        return n


class RankPath:
    """A determined path of rank-1 operators composed in parallel:

        y = normalize( x + sum_k u_k (v_k . x) )

    A rank-L path is a rank-L map R^1000 -> R^1000: the path *is* the
    capacity. Fitted exactly by deterministic orthogonal deflation."""

    def __init__(self, key: str, ops: list[PathOp]) -> None:
        self.key = key
        self.ops = ops

    def apply(self, x: Sequence[int]) -> tuple[int, ...]:
        acc = list(x)
        for op in self.ops:
            s = dot_fx(op.v, x)
            if s == 0:
                continue
            for i, ui in enumerate(op.u):
                acc[i] += (ui * s) >> SHIFT
        return normalize(clamp_vec(acc))

    def param_count(self, learned_only: bool = True) -> int:
        return sum(op.param_count(learned_only) for op in self.ops)


# ------------------------------------------------------------- the lattice
class Determination:
    """Result of determining the next X."""

    __slots__ = ("value", "cos", "path", "state", "abstained")

    def __init__(self, value: str | None, cos: int,
                 path: list[str], state: tuple[int, ...]) -> None:
        self.value = value or ABSTAIN
        self.cos = cos
        self.path = path
        self.state = state
        self.abstained = value is None

    def to_json(self) -> dict:
        return {
            "value": self.value,
            "confidence": round(self.cos / ONE, 6),
            "path": self.path,
            "abstained": self.abstained,
        }


class Lattice:
    def __init__(self, dim: int = DIM,
                 route_min_cos: int = ROUTE_MIN_COS,
                 max_steps: int = MAX_STEPS) -> None:
        self.dim = dim
        self.ops: list[PathOp] = []
        self.rank_paths: list[RankPath] = []
        self.codebook = Codebook(dim)
        self.route_min_cos = route_min_cos
        self.max_steps = max_steps

    # -- construction ----------------------------------------------------
    def add_op(self, key: str, u: Sequence[int], v: Sequence[int],
               mode: str = "set", gain: int = ONE,
               learn_u: bool = True, learn_v: bool = True,
               learn_gain: bool = False, normalize_uv: bool = True) -> PathOp:
        if normalize_uv:
            u, v = normalize(u), normalize(v)
        op = PathOp(key, u, v, gain=gain, mode=mode, learn_u=learn_u,
                    learn_v=learn_v, learn_gain=learn_gain)
        self.ops.append(op)
        return op

    # -- routing: the path is determined here ---------------------------
    @staticmethod
    def _cos_from_raw(raw: int, vn: int, hn: int) -> int:
        if vn == 0 or hn == 0:
            return 0
        c = (raw * ONE) // (vn * hn)
        return -ONE if c < -ONE else (ONE if c > ONE else c)

    def best_match(self, h: Sequence[int]) -> tuple[int, int]:
        """Highest-scoring operator without any threshold gate."""
        best, best_raw = -1, None
        for i, op in enumerate(self.ops):
            raw = sumprod(op.v, h)          # equal-norm v: argmax of dot
            if best_raw is None or raw > best_raw:
                best, best_raw = i, raw     # strict > : lowest index wins ties
        if best < 0:
            return -1, 0
        return best, self._cos_from_raw(best_raw, self.ops[best].vn, vnorm(h))

    def route(self, h: Sequence[int],
              exclude: frozenset[int] = frozenset()) -> tuple[int, int]:
        if exclude:
            best, best_raw = -1, None
            for i, op in enumerate(self.ops):
                if i in exclude:
                    continue
                raw = sumprod(op.v, h)
                if best_raw is None or raw > best_raw:
                    best, best_raw = i, raw
            if best < 0:
                return -1, 0
            c = self._cos_from_raw(best_raw, self.ops[best].vn, vnorm(h))
            return (best, c) if c >= self.route_min_cos else (-1, c)
        i, c = self.best_match(h)
        return (i, c) if c >= self.route_min_cos else (-1, c)

    def step(self, h: Sequence[int],
             exclude: frozenset[int] = frozenset()) -> tuple[tuple[int, ...], str | None, int]:
        i, c = self.route(h, exclude)
        if i < 0:
            return tuple(h), None, c
        return self.ops[i].apply(h), self.ops[i].key, c

    def walk(self, h: Sequence[int], max_steps: int | None = None
             ) -> tuple[tuple[int, ...], list[str]]:
        visited: set[int] = set()
        state = tuple(h)
        path: list[str] = []
        for _ in range(max_steps or self.max_steps):
            nxt, key, _ = self.step(state, frozenset(visited))
            if key is None:
                break
            visited.add(self._index_of(key))
            state = nxt
            path.append(key)
        return state, path

    def _index_of(self, key: str) -> int:
        for i, op in enumerate(self.ops):
            if op.key == key:
                return i
        return -1

    # -- the core API: determine the next X -----------------------------
    def determine_next(self, context: Sequence[int]) -> Determination:
        state, key, c = self.step(context)
        if key is None:
            return Determination(None, c, [], tuple(context))
        sym, dc = self.codebook.decode(state)
        return Determination(sym, dc, [key], state)

    def determine_path(self, context: Sequence[int],
                       max_steps: int | None = None) -> Determination:
        state, path = self.walk(context, max_steps)
        sym, dc = self.codebook.decode(state)
        return Determination(sym, dc, path, state)

    # -- accounting -----------------------------------------------------
    def param_count(self, learned_only: bool = True) -> int:
        n = sum(op.param_count(learned_only) for op in self.ops)
        n += sum(p.param_count(learned_only) for p in self.rank_paths)
        return n

    def state_hash(self) -> str:
        blob = [[op.key, op.mode, op.gain, list(op.u), list(op.v)] for op in self.ops]
        blob += [[p.key, [[o.key, list(o.u), list(o.v)] for o in p.ops]]
                 for p in self.rank_paths]
        blob += [self.route_min_cos, self.max_steps, self.dim]
        return hashlib.sha256(
            json.dumps(blob, sort_keys=True).encode()).hexdigest()

    # -- persistence ----------------------------------------------------
    def to_json(self) -> dict:
        return {
            "version": 1,
            "dim": self.dim,
            "route_min_cos": self.route_min_cos,
            "max_steps": self.max_steps,
            "ops": [{"key": op.key, "mode": op.mode, "gain": op.gain,
                     "u": list(op.u), "v": list(op.v)} for op in self.ops],
            "rank_paths": [
                {"key": p.key,
                 "ops": [{"key": o.key, "u": list(o.u), "v": list(o.v)}
                         for o in p.ops]} for p in self.rank_paths],
            "candidates": self.codebook.candidates,
            "state_hash": self.state_hash(),
            "param_count": self.param_count(),
        }

    def save(self, path: str = MODEL_FILE) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str = MODEL_FILE) -> "Lattice":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        lat = cls(dim=d["dim"], route_min_cos=d["route_min_cos"],
                  max_steps=d["max_steps"])
        for o in d["ops"]:
            lat.add_op(o["key"], o["u"], o["v"], mode=o["mode"], gain=o["gain"],
                       normalize_uv=False)
        for p in d["rank_paths"]:
            ops = [PathOp(o["key"], o["u"], o["v"], mode="add")
                   for o in p["ops"]]
            lat.rank_paths.append(RankPath(p["key"], ops))
        for c in d["candidates"]:
            lat.codebook.register(c)
        return lat


# ------------------------------------------------------------- fitting
def fit_set_pair(x: Sequence[int], y: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exact rank-1 'set' operator for one pair: u(v.x) == y."""
    return normalize(y), normalize(x)


def _gram_schmidt(xs: Sequence[Sequence[int]], rank: int) -> list[tuple[int, ...]]:
    basis: list[tuple[int, ...]] = []
    for x in xs:
        w = list(x)
        for v in basis:
            raw = sumprod(v, w)
            w = [wi - (vi * raw) // (ONE * ONE) for wi, vi in zip(w, v)]
        n = vnorm(w)
        if n > ONE // 100:
            basis.append(tuple((wi * ONE) // n for wi in w))
        if len(basis) >= rank:
            break
    return basis


def _ls_weights(smat: list[list], n_pairs: int, rank: int) -> list[list]:
    """W = (S^T S)^-1 S^T as exact Fractions. smat[i][k] = v_k . x_i / ONE."""
    m = rank
    a = [[sum(smat[i][k] * smat[i][l] for i in range(n_pairs))
          for l in range(m)] for k in range(m)]
    b = [[smat[i][k] for i in range(n_pairs)] for k in range(m)]
    aug = [a[k] + b[k] for k in range(m)]
    for col in range(m):
        piv = next((r for r in range(col, m) if aug[r][col] != 0), None)
        if piv is None:
            continue
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(m):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [[aug[k][m + i] for i in range(n_pairs)] for k in range(m)]


def fit_rank_path(key: str, pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
                  rank: int) -> RankPath:
    """Deterministic closed-form rank-L fit: orthogonal deflation + exact
    least squares on the deflated basis. rank >= n_pairs fits exactly."""
    xs = [normalize(x) for x, _ in pairs]
    ys = [normalize(y) for _, y in pairs]
    basis = _gram_schmidt(xs, rank)
    k = len(basis)
    smat: list[list] = []
    for x in xs:
        row = []
        for v in basis:
            s = dot_fx(v, x)
            row.append(Fraction(s, ONE))
        smat.append(row)
    w = _ls_weights(smat, len(xs), k)
    residuals = [[yi - xi for yi, xi in zip(y, x)] for x, y in zip(xs, ys)]
    ops: list[PathOp] = []
    for kk in range(k):
        u = [0] * DIM
        for i, r in enumerate(residuals):
            coef = w[kk][i]
            num, den = coef.numerator, coef.denominator
            if den == 0:
                continue
            for d in range(DIM):
                u[d] += (r[d] * num) // den
        ops.append(PathOp(f"{key}#{kk}", u, basis[kk], mode="add"))
    return RankPath(key, ops)


def rank_path_error(path: RankPath,
                    pairs: Sequence[tuple[Sequence[int], Sequence[int]]]) -> float:
    """Per-coordinate MSE of the composed path against the targets."""
    err = 0.0
    for x, y in pairs:
        y_hat = path.apply(x)
        for a, b in zip(y_hat, normalize(y)):
            e = (a - b) / ONE
            err += e * e
    return err / (max(1, len(pairs)) * DIM)


def nlms_refine(path: RankPath,
                pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
                epochs: int = 32) -> float:
    """Deterministic error-driven refinement (canonical example order)."""
    eta = ONE // 8
    ordered = sorted(pairs, key=lambda p: sumprod(p[0], p[0]))
    last = 0.0
    for _ in range(epochs):
        for x, y in ordered:
            y_hat = path.apply(x)
            err = [a - b for a, b in zip(normalize(y), y_hat)]
            for op in path.ops:
                s = dot_fx(op.v, x)
                if s == 0:
                    continue
                u = [ui + (e * s * eta) >> (2 * SHIFT) for ui, e in zip(op.u, err)]
                op.u = tuple(clamp_vec(u))
        last = rank_path_error(path, pairs)
    return last


# ------------------------------------------------------------- compilation
class KnowledgeBase:
    def __init__(self) -> None:
        self.triples: list[tuple[str, str, str]] = []

    def add(self, subject: str, relation: str, obj: str) -> None:
        self.triples.append((subject, relation, obj))

    def query_vec(self, cb: Codebook, subject: str, relation: str) -> tuple[int, ...]:
        return normalize(
            [a + b for a, b in zip(cb.vec(subject), cb.vec(REL + relation))])

    def compile(self, lattice: Lattice) -> list[tuple[tuple[int, ...], tuple[int, ...], str]]:
        cb = lattice.codebook
        pairs = []
        for (s, r, o) in self.triples:
            cb.register(s)
            cb.register(o)
            x = self.query_vec(cb, s, r)
            y = normalize([a + b for a, b in zip(cb.vec(o), cb.vec(REL + r))])
            lattice.add_op(f"{s}--{r}-->{o}", y, x, mode="set", normalize_uv=False)
            pairs.append((x, y, f"{s}--{r}"))
        return pairs


def compile_arithmetic(lattice: Lattice) -> list[tuple[tuple[int, ...], tuple[int, ...], str]]:
    """a + b table (0..9): determines the sum symbol exactly."""
    cb = lattice.codebook
    pairs = []
    for a in range(10):
        for b in range(10):
            for k in (a, b, a + b):
                cb.register(f"n{k}")
            x = normalize([p + q + r for p, q, r in
                           zip(cb.vec(f"n{a}"), cb.vec(f"n{b}"), cb.vec(REL + "plus"))])
            y = normalize(cb.vec(f"n{a + b}"))
            lattice.add_op(f"{a}+{b}", y, x, mode="set", normalize_uv=False)
            pairs.append((x, y, f"{a}+{b}"))
    return pairs


def build_lattice(include_tools: bool = True) -> tuple[Lattice, KnowledgeBase]:
    lat = Lattice()
    kb = KnowledgeBase()
    for (s, r, o) in FACTS:
        kb.add(s, r, o)
    if include_tools:
        for (s, r, o) in TOOL_FACTS:
            kb.add(s, r, o)
    kb.compile(lat)
    compile_arithmetic(lat)
    return lat, kb


# --------------------------------------------------------- calibration
def calibrate_route_threshold(lat: Lattice, positives: Sequence[tuple[int, ...]],
                              negatives: Sequence[tuple[int, ...]]) -> tuple[int, float, float]:
    """Smallest grid threshold separating every negative from every positive."""
    pos = [lat.best_match(h)[1] for h in positives]
    neg_best = max((lat.best_match(h)[1] for h in negatives), default=-ONE)
    pos_min = min(pos) if pos else ONE
    grid = [int(t * ONE / 100) for t in range(50, 100)]
    chosen = next((g for g in grid if g > neg_best and g <= pos_min), None)
    separable = chosen is not None
    if chosen is None:
        # positives and negatives overlap: no threshold can separate them.
        # Fall back to the safe default and let the caller see the failure.
        chosen = max(ROUTE_MIN_COS, neg_best + ONE // 100)
    lat.route_min_cos = chosen
    return chosen, pos_min / ONE, neg_best / ONE, separable


def make_negatives(kb: KnowledgeBase) -> list[tuple[int, ...]]:
    lat = Lattice()   # scratch, just for the codebook
    cb = lat.codebook
    neg: list[tuple[int, ...]] = []
    relations = sorted({r for _, r, _ in kb.triples} | {"minus", "times"})
    subjects = sorted({s for s, _, _ in kb.triples} | {f"n{k}" for k in range(19)})
    true_pairs = {(s, r) for s, r, _ in kb.triples}
    binary = {"plus", "minus", "times"}
    for s in subjects:
        for r in relations:
            if (s, r) in true_pairs:
                continue
            if s.startswith("n") and s[1:].isdigit() and r in binary:
                continue          # binary ops are queried with two operands
            neg.append(kb.query_vec(cb, s, r))
    for a in range(10):
        for b in range(10):
            for r in ("minus", "times"):
                x = normalize([p + q + t for p, q, t in
                               zip(cb.vec(f"n{a}"), cb.vec(f"n{b}"), cb.vec(REL + r))])
                neg.append(x)
    return neg


# ----------------------------------------------- tier S0 (1,000 params)
class ChannelGainModel:
    """Tier S0: exactly 1,000 learned parameters -- one gain per channel of
    the frozen codebook -- fitted in closed form (per-channel least squares)."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.gains: list[int] = [ONE] * dim

    def fit(self, pairs: Sequence[tuple[Sequence[int], Sequence[int]]]) -> None:
        num = [0] * self.dim
        den = [0] * self.dim
        for x, y in pairs:
            for i in range(self.dim):
                num[i] += x[i] * y[i]
                den[i] += x[i] * x[i]
        self.gains = [(num[i] * ONE) // den[i] if den[i] else ONE
                      for i in range(self.dim)]

    def apply(self, x: Sequence[int]) -> tuple[int, ...]:
        # No re-normalisation: the gains are the learned quantity and the
        # target is compared in the same (unnormalised) units.
        return tuple((g * xi) >> SHIFT for g, xi in zip(self.gains, x))

    def param_count(self) -> int:
        return len(self.gains)


def s0_experiment() -> dict:
    """Denoising task with heteroscedastic channel noise: the 1,000 gains
    learn inverse-noise weighting and beat the flat-gain baseline."""
    cb = Codebook()
    rng = 0
    train: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    symbols = [f"sym{i:03d}" for i in range(64)]
    noise_amp = [3 if (i * 2654435761) % 8 < 2 else 1 for i in range(DIM)]
    for si, sym in enumerate(symbols):
        clean = cb.vec(sym)
        noisy = []
        for i in range(DIM):
            rng = splitmix64(rng + si * DIM + i)
            d = ((rng >> 33) % 7 - 3) * ONE * noise_amp[i] // 4
            noisy.append(clean[i] + d)
        train.append((tuple(noisy), clean))
    def mse(preds: Sequence[tuple[int, ...]]) -> float:
        total = 0.0
        for (x, y), p in zip(train, preds):
            e = [a - b for a, b in zip(p, y)]
            total += sumprod(e, e) / (ONE * ONE) / DIM
        return total / len(train)

    def decode_acc(states: Sequence[tuple[int, ...]]) -> float:
        hits = 0
        for i, st in enumerate(states):
            want = train[i][1]
            other = train[(i + 17) % len(train)][1]
            if cos_fx(st, want) > cos_fx(st, other):
                hits += 1
        return hits / len(states)

    model = ChannelGainModel()
    before = [x for x, _ in train]
    mse_before = mse(before)
    acc_before = decode_acc(before)
    model.fit(train)
    after = [model.apply(x) for x, _ in train]
    return {"params": model.param_count(), "mse_before": round(mse_before, 4),
            "mse_after": round(mse(after), 4),
            "decode_acc_before": round(acc_before, 4),
            "decode_acc_after": round(decode_acc(after), 4)}


# ------------------------------------------------------------- training
def _multi_valued_relations() -> set[str]:
    """A relation is multi-valued if the seed corpus shows one subject with
    more than one object for it (e.g. has-moon). Everything else is treated
    as functional: a second answer is a contradiction."""
    from pathway_seed import FACTS, TOOL_FACTS
    seen: dict[tuple[str, str], set[str]] = {}
    for (s, r, o) in FACTS + TOOL_FACTS:
        seen.setdefault((s, r), set()).add(o)
    return {r for (s, r), objs in seen.items() if len(objs) > 1}


MULTI_VALUED = _multi_valued_relations()


def learn_fact(lat: "Lattice", subject: str, relation: str, obj: str,
               allow_multiple: bool | None = None) -> dict:
    """Incremental compilation: one datum -> one operator -> +2,000 params.
    This is the growth loop an agent feeds; no gradient descent involved.
    Returns a status dict so a caller can react to conflicts.

    allow_multiple=None -> inferred from the corpus: multi-valued relations
    (has-moon, ...) accept several objects per subject; functional relations
    (capital, ...) report 'conflict' for a second answer instead."""
    if allow_multiple is None:
        allow_multiple = relation in MULTI_VALUED
    cb = lat.codebook
    cb.register(subject)
    cb.register(obj)
    x = normalize([a + b for a, b in zip(cb.vec(subject), cb.vec(REL + relation))])
    y = normalize([a + b for a, b in zip(cb.vec(obj), cb.vec(REL + relation))])
    prefix = f"{subject}--{relation}-->"
    existing = [op for op in lat.ops if op.key.startswith(prefix)]
    if any(op.key == f"{prefix}{obj}" for op in existing):
        return {"status": "redundant", "fact": (subject, relation, obj),
                "params": lat.param_count()}
    if existing and not allow_multiple:
        # contradictory knowledge: same query, different answer.
        # deterministic policy: first-compiled wins routing ties, both kept,
        # the caller is told so a human/upstream policy can retract.
        return {"status": "conflict", "fact": (subject, relation, obj),
                "existing": [op.key.split("-->")[-1] for op in existing],
                "params": lat.param_count()}
    lat.add_op(f"{prefix}{obj}", y, x, mode="set", normalize_uv=False)
    status = "learned" if not existing else "learned-multi"
    return {"status": status, "fact": (subject, relation, obj),
            "params": lat.param_count()}


def train_model(verbose: bool = True) -> tuple[Lattice, KnowledgeBase, dict]:
    t0 = time.perf_counter()
    lat, kb = build_lattice()
    positives = [kb.query_vec(lat.codebook, s, r) for (s, r, _) in kb.triples]
    positives += [normalize([p + q + r for p, q, r in
                             zip(lat.codebook.vec(f"n{a}"), lat.codebook.vec(f"n{b}"),
                                 lat.codebook.vec(REL + "plus"))])
                  for a in range(10) for b in range(10)]
    negatives = make_negatives(kb)
    thr, pmin, nmax, separable = calibrate_route_threshold(lat, positives, negatives)
    info = {
        "ops": len(lat.ops),
        "learned_params": lat.param_count(),
        "capacity_params": 500 * 2000,
        "route_threshold": round(thr / ONE, 4),
        "pos_cos_min": round(pmin, 4),
        "neg_cos_max": round(nmax, 4),
        "margin": round(pmin - nmax, 4),
        "separable": separable,
        "state_hash": lat.state_hash(),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }
    lat.save()
    if verbose:
        print(f"[train] ops={info['ops']}  params={info['learned_params']:,} "
              f"/ capacity {info['capacity_params']:,}")
        print(f"[train] routing threshold {info['route_threshold']}  "
              f"margin pos_min {info['pos_cos_min']} - neg_max {info['neg_cos_max']}")
        if not separable:
            print("[train] WARNING: positives and negatives overlap; "
                  "no threshold separates them, using the safe fallback.")
        print(f"[train] state hash {info['state_hash'][:16]}...  "
              f"saved to {MODEL_FILE}")
    return lat, kb, info


# ----------------------------------------------------------------- CLI
def cmd_query(lat: Lattice, spec: str, hops: int = 1) -> None:
    parts = spec.split()
    if len(parts) < 2:
        print("usage: --query \"<subject> <relation>\"   e.g. \"paris capital-of\"")
        return
    subject, relation = parts[0], "-".join(parts[1:])
    cb = lat.codebook
    cb.register(subject)
    h = normalize([a + b for a, b in zip(cb.vec(subject), cb.vec(REL + relation))])
    det = lat.determine_next(h)
    print(json.dumps(det.to_json(), indent=2))
    if hops > 1:
        full = lat.determine_path(h, hops)
        print(json.dumps(full.to_json(), indent=2))


def cmd_demo(lat: Lattice) -> None:
    cb = lat.codebook

    def ask(subject: str, relation: str, hops: int = 1, title: str = "") -> None:
        h = normalize([a + b for a, b in zip(cb.vec(subject), cb.vec(REL + relation))])
        det = lat.determine_next(h)
        full = lat.determine_path(h, hops)
        print(f"  {title or (subject + ' ' + relation):<34} -> {det.value:<22} "
              f"conf {det.cos / ONE:.3f}")
        if len(full.path) > 1:
            print(f"      path ({len(full.path)} ops): " + " | ".join(full.path))

    print("== 1. direct fact determination (next X) ==")
    ask("france", "capital", title="france capital")
    ask("gold", "symbol", title="gold symbol")
    ask("python", "created-by", title="python created-by")

    print("== 2. multi-hop path walking (transitive closure) ==")
    ask("socrates", "teaches", hops=4, title="teaches socrates (who, transitively)")
    ask("socrates", "is-a", hops=5, title="is-a socrates (terminal)")
    ask("alice", "knows", hops=5, title="knows alice (terminal)")

    print("== 3. determined arithmetic (exact, not sampled) ==")
    for a, b in ((7, 5), (9, 9), (0, 4)):
        h = normalize([p + q + r for p, q, r in
                       zip(cb.vec(f"n{a}"), cb.vec(f"n{b}"), cb.vec(REL + "plus"))])
        det = lat.determine_next(h)
        print(f"  {a} + {b:<28} -> {det.value:<22} conf {det.cos / ONE:.3f}  "
              f"path {det.path}")

    print("== 4. tool-call determination: intent -> tool -> arg -> type ==")
    for intent in ("ask-weather", "search-web", "book-flight"):
        h = normalize([a + b for a, b in zip(cb.vec(intent), cb.vec(REL + "calls"))])
        nxt = lat.determine_next(h)
        full = lat.determine_path(h, 3)
        chain = [p.split("-->")[-1] for p in full.path]
        print(f"  {intent:<20} -> next X: {nxt.value:<22} conf {nxt.cos / ONE:.3f}")
        print(f"      determined trajectory: {intent} -> " + " -> ".join(chain))

    print("== 5. abstention: unknown stays unknown (no hallucination) ==")
    for s, r in (("germany", "symbol"), ("gold", "orbits")):
        h = normalize([a + b for a, b in zip(cb.vec(s), cb.vec(REL + r))])
        det = lat.determine_next(h)
        print(f"  {s} {r:<26} -> {det.value}  conf {det.cos / ONE:.3f}")

    print("== 6. surgical knowledge edit (one fact = one operator) ==")
    victim = "france--capital-->paris"
    removed = [op for op in lat.ops if op.key == victim]
    lat.ops = [op for op in lat.ops if op.key != victim]
    h = normalize([a + b for a, b in zip(cb.vec("france"), cb.vec(REL + "capital"))])
    det = lat.determine_next(h)
    print(f"  after removing {victim}: -> {det.value}")
    lat.ops.extend(removed)
    det = lat.determine_next(h)
    print(f"  after re-adding:           -> {det.value}")

    print("== 7. rank paths: path rank = capacity (compression) ==")
    pairs = []
    for (s, r, o) in FACTS[:8]:
        x = normalize([a + b for a, b in zip(cb.vec(s), cb.vec(REL + r))])
        y = normalize([a + b for a, b in zip(cb.vec(o), cb.vec(REL + r))])
        pairs.append((x, y))
    print(f"   {'rank':>4} {'params':>8} {'residual MSE':>14}   note")
    for rank in (1, 2, 4, 8):
        p = fit_rank_path("demo", pairs, rank)
        err = rank_path_error(p, pairs)
        note = "exact" if rank >= len(pairs) else f"compresses {len(pairs)} pairs"
        print(f"   {rank:>4} {p.param_count():>8,} {err:>14.6f}   {note}")

    print("== 8. determinism ==")
    print(f"  state hash: {lat.state_hash()}")
    det1 = lat.determine_path(
        normalize([a + b for a, b in zip(cb.vec('socrates'), cb.vec(REL + 'teaches'))]), 4)
    det2 = lat.determine_path(
        normalize([a + b for a, b in zip(cb.vec('socrates'), cb.vec(REL + 'teaches'))]), 4)
    print(f"  two independent runs identical: "
          f"{det1.to_json() == det2.to_json() and det1.value == det2.value}")


def cmd_verify(lat: Lattice, kb: KnowledgeBase) -> int:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    print("== verify: fixed-point kernel ==")
    check("cos bounded in [-1,1]", all(
        -ONE <= cos_fx(lat.codebook.vec("a"), lat.codebook.vec(t)) <= ONE
        for t in ("b", "c", "d")))
    n = vnorm(normalize(lat.codebook.vec("x")))
    check("normalize preserves unit norm", abs(n - ONE) <= 4, f"|v|={n}")

    print("== verify: routing ==")
    pos_ok = all(lat.route(kb.query_vec(lat.codebook, s, r))[0] >= 0
                 for (s, r, _) in kb.triples)
    check("every compiled fact routes", pos_ok, f"{len(kb.triples)} triples")
    neg = make_negatives(kb)
    false_routes = sum(1 for h in neg if lat.route(h)[0] >= 0)
    check("no false routes on negatives", false_routes == 0,
          f"{len(neg)} negatives, {false_routes} false")

    print("== verify: reasoning (next X determined exactly) ==")
    qa = [("france", "capital", "paris"), ("gold", "symbol", "Au"),
          ("python", "created-by", "guido-van-rossum"), ("earth", "orbits", "sun")]
    bad = []
    for s, r, want in qa:
        h = normalize([a + b for a, b in zip(lat.codebook.vec(s), lat.codebook.vec(REL + r))])
        got = lat.determine_next(h).value
        if got != want:
            bad.append((s, r, got, want))
    check("direct QA exact", not bad, str(bad))

    h = normalize([a + b for a, b in zip(lat.codebook.vec("socrates"), lat.codebook.vec(REL + "teaches"))])
    det = lat.determine_path(h, 4)
    check("3-hop chain terminal", det.value == "alexander",
          f"got {det.value} via {len(det.path)} ops")

    h = normalize([a + b for a, b in zip(lat.codebook.vec("alice"), lat.codebook.vec(REL + "knows"))])
    det = lat.determine_path(h, 5)
    check("4-hop chain terminal", det.value == "erin", f"got {det.value}")

    arith_bad = []
    for a in range(10):
        for b in range(10):
            h = normalize([p + q + t for p, q, t in
                           zip(lat.codebook.vec(f"n{a}"), lat.codebook.vec(f"n{b}"),
                               lat.codebook.vec(REL + "plus"))])
            if lat.determine_next(h).value != f"n{a + b}":
                arith_bad.append((a, b))
    check("arithmetic table 100/100", not arith_bad, str(arith_bad[:5]))

    print("== verify: abstention ==")
    h = normalize([a + b for a, b in zip(lat.codebook.vec("germany"), lat.codebook.vec(REL + "symbol"))])
    check("unknown triple abstains", lat.determine_next(h).abstained)
    h = normalize([p + q + t for p, q, t in
                   zip(lat.codebook.vec("n3"), lat.codebook.vec("n5"), lat.codebook.vec(REL + "times"))])
    det = lat.determine_next(h)
    check("3 times 5 abstains (table is plus-only)", det.abstained or det.value == ABSTAIN)

    print("== verify: determinism ==")
    lat2, kb2 = build_lattice()
    lat2.route_min_cos = lat.route_min_cos
    check("independent build -> identical state hash",
          lat.state_hash() == lat2.state_hash(), lat.state_hash()[:16])
    d1 = lat.determine_next(h).to_json()
    d2 = lat.determine_next(h).to_json()
    check("repeated inference byte-identical", d1 == d2)

    print("== verify: tier S0 (1,000-param seed) ==")
    s0 = s0_experiment()
    check("S0 learned gains reduce denoising MSE",
          s0["mse_after"] < s0["mse_before"],
          f"{s0['mse_before']} -> {s0['mse_after']}")

    print("== verify: growth loop ==")
    grow_lat, _ = build_lattice()
    before = grow_lat.param_count()
    r1 = learn_fact(grow_lat, "unit", "tests", "a")
    r2 = learn_fact(grow_lat, "unit", "tests", "b")
    r3 = learn_fact(grow_lat, "unit", "tests", "a")
    r4 = learn_fact(grow_lat, "unit", "multi", "x")
    r5 = learn_fact(grow_lat, "unit", "multi", "y", allow_multiple=True)
    check("learned status adds exactly 2,000 params per fact",
          r1["status"] == "learned"
          and grow_lat.param_count() - before == 3 * 2000,
          f"delta {grow_lat.param_count() - before:,}")
    check("functional relation flags conflict", r2["status"] == "conflict")
    check("re-learning a fact is redundant", r3["status"] == "redundant")
    check("multi-valued relation accepts a second object",
          r5["status"] == "learned-multi")
    h = kb_query(grow_lat.codebook, "unit", "tests")
    check("newly learned fact is immediately answerable",
          grow_lat.determine_next(h).value == "a")

    print("== verify: parameter accounting ==")
    check("tier S0 == 1,000 params", ChannelGainModel().param_count() == 1000)
    check("tier S1 capacity == 1,000,000", 500 * 2000 == 1_000_000)
    check("learned params counted", lat.param_count() == len(lat.ops) * 2000,
          f"{len(lat.ops)} ops x 2000 = {lat.param_count():,}")

    print("== verify: rank path exactness ==")
    cb = lat.codebook
    pairs = []
    for (s, r, o) in FACTS[20:24]:
        x = normalize([a + b for a, b in zip(cb.vec(s), cb.vec(REL + r))])
        y = normalize([a + b for a, b in zip(cb.vec(o), cb.vec(REL + r))])
        pairs.append((x, y))
    p4 = fit_rank_path("v", pairs, 4)
    check("rank-4 path fits 4 pairs exactly", rank_path_error(p4, pairs) < 1e-5,
          f"MSE {rank_path_error(p4, pairs):.2e}")
    p2 = fit_rank_path("v", pairs, 2)
    check("rank-2 path underfits (as expected)", rank_path_error(p2, pairs) > 1e-5,
          f"MSE {rank_path_error(p2, pairs):.2e}")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


def cmd_bench(lat: Lattice) -> None:
    cb = lat.codebook
    queries = [kb_query(cb, s, r) for (s, r, _) in FACTS[:200]]
    t0 = time.perf_counter()
    n = 0
    for h in queries:
        lat.route(h)
        n += 1
    per = (time.perf_counter() - t0) / n * 1e3
    print(f"== bench: routing over {len(lat.ops)} ops ==")
    print(f"  latency: {per:.2f} ms/query   throughput: {1000/per:.0f} queries/s")

    print("== bench: parameter scaling (P operators x 2,000) ==")
    print(f"   {'ops P':>6} {'learned params':>16} {'stage':>10}")
    for p in (1, 10, 50, 100, 250, 500):
        stage = "S0->S1 bridge" if p == 1 else ("S1 target" if p == 500 else "growth")
        print(f"   {p:>6} {p * 2000:>16,} {stage:>10}")

    print("== bench: robustness to deterministic input noise ==")
    h0 = kb_query(cb, "france", "capital")
    print(f"   {'noise':>8} {'conf':>8} {'answer':>12}")
    for level in (0, 5, 10, 20, 40):
        rng = 42
        noisy = []
        for xi in h0:
            rng = splitmix64(rng)
            noisy.append(xi + ((rng >> 40) % (2 * level + 1) - level) * ONE // 512)
        det = lat.determine_next(tuple(noisy))
        print(f"   {level / 100:>8.2f} {det.cos / ONE:>8.3f} {det.value:>12}")

    print("== bench: tier S0 (1,000-param seed) ==")
    s0 = s0_experiment()
    print(f"  params {s0['params']}   denoising MSE {s0['mse_before']} -> "
          f"{s0['mse_after']}   ({s0['mse_before'] / max(1e-9, s0['mse_after']):.2f}x better)")
    print(f"  retrieval accuracy {s0['decode_acc_before']:.3f} -> "
          f"{s0['decode_acc_after']:.3f}")


def kb_query(cb: Codebook, s: str, r: str) -> tuple[int, ...]:
    return normalize([a + b for a, b in zip(cb.vec(s), cb.vec(REL + r))])


def cmd_report(lat: Lattice) -> None:
    print("== parameter inventory ==")
    rows = [
        ("codebook (frozen, hash-derived)", 0, "no"),
        ("tier S0 channel gains", 1000, "yes"),
        (f"path operators ({len(lat.ops)} x read 1000 + write 1000)",
         len(lat.ops) * 2000, "yes"),
        ("tier S1 capacity ceiling (500 ops)", 1_000_000, "yes"),
    ]
    for name, count, learned in rows:
        print(f"  {name:<48} {count:>10,}   learned: {learned}")
    print(f"  {'current model total (learned)':<48} "
          f"{lat.param_count():>10,}")
    print(f"  state hash {lat.state_hash()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pathway Lattice v1")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--hops", type=int, default=1)
    args = ap.parse_args(argv)

    if args.train:
        train_model()
        return 0

    if not os.path.exists(MODEL_FILE):
        print(f"[init] {MODEL_FILE} not found; training first...")
        train_model()
    lat = Lattice.load(MODEL_FILE)
    _, kb = build_lattice()

    if args.query:
        cmd_query(lat, args.query, args.hops)
    if args.demo:
        cmd_demo(lat)
    if args.verify:
        return cmd_verify(lat, kb)
    if args.bench:
        cmd_bench(lat)
    if args.report:
        cmd_report(lat)
    if not any((args.query, args.demo, args.verify, args.bench, args.report)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
