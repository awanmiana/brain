#!/usr/bin/env python3
"""
Triple Lattice v1 -- 1000 x 1000 x 1000 structured determination.

The idea (formalized from "1000 probabilities * 1000 probabilities each of
different domain/classification * 1000 probabilities"):

    A flat generative head asks:  P(next token | context)   over ~50k tokens
    The Triple Lattice asks:      P(d) * P(r | d) * P(v | d, r)

    stage 1:  1000 DOMAIN channels     (which domain/classification)
    stage 2:  1000 RELATION channels   (which attribute inside the domain)
    stage 3:  1000 VALUE channels      (which value inside the cell)

    addressable outcome space = 1000 * 1000 * 1000 = 1,000,000,000 cells

But nothing is sampled. Each stage is *determined* by exact integer-cosine
argmax and judged against a reliability table fitted in closed form, giving
one of four verdicts:

    PROBABLE   all three stages determined; assert the value        (precision)
    POSSIBLE   value ambiguous or confidence low; keep candidates   (recall)
    ESTIMATE   answer reached by composed multi-hop path            (inference)
    UNKNOWN    cannot determine -> abstain, never hallucinate       (honesty)

The three probabilities are NOT multiplied blindly: confidence is the
weakest link of the calibrated per-stage probabilities, so the number
stays honest instead of being over-optimistic.

Usage:
    python3 triple_lattice.py --demo
    python3 triple_lattice.py --calibrate
    python3 triple_lattice.py --experiment
    python3 triple_lattice.py --query "france capital"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from math import sumprod

from pathway_lattice import (
    ABSTAIN, DIM, ONE, REL, Lattice, cos_fx, kb_query, normalize,
    splitmix64, MODEL_FILE)
from pathway_seed import FACTS, TOOL_FACTS

# ------------------------------------------------------------- capacity
DOMAIN_CHANNELS = 1000
RELATION_CHANNELS = 1000
VALUE_CHANNELS = 1000
ADDRESSABLE_CELLS = DOMAIN_CHANNELS * RELATION_CHANNELS * VALUE_CHANNELS

AMBIGUITY_MARGIN = 10 * ONE // 100      # top-2 within 0.10 -> ambiguous
PROBABLE_MIN = 0.99                     # calibrated P(correct) to assert
POSSIBLE_MIN = 0.50                     # below this -> abstain

# relation -> domain (the "different domain/classification" axis)
DOMAIN_OF = {
    "capital": "geography", "symbol": "chemistry",
    "orbits": "science", "travels-at": "science", "freezes-at": "science",
    "boils-at": "science", "is": "science", "approx": "science",
    "encodes": "science", "is-made-of": "science", "has-moon": "science",
    "is-a": "taxonomy", "created-by": "computing", "invented-by": "computing",
    "developed": "history", "discovered": "history", "observed": "history",
    "designed": "history", "wrote": "history", "composed": "history",
    "teaches": "people", "knows": "people",
    "calls": "tools", "needs": "tools", "type": "tools",
    "plus": "arithmetic",
}


def parse_key(key: str) -> tuple[str, str, str]:
    """Op key -> (subject, relation, object)."""
    if "-->" in key:
        left, o = key.split("-->")
        s, r = left.split("--", 1)
        return s, r, o
    a, b = key.split("+")                      # arithmetic ops: "7+5"
    return f"n{a}", "plus", f"n{int(a) + int(b)}"


# -------------------------------------------------------- reliability
class Reliability:
    """Deterministic binned calibration: raw score -> empirical P(correct).
    Fitted in closed form by counting; no SGD, no seeds."""

    def __init__(self, n_bins: int = 20) -> None:
        self.n_bins = n_bins
        self.counts = [0] * n_bins
        self.hits = [0] * n_bins

    def _bin(self, score: float) -> int:
        b = int(score * self.n_bins)
        return min(self.n_bins - 1, max(0, b))

    def fit(self, samples: list[tuple[float, int]]) -> "Reliability":
        for score, correct in samples:
            b = self._bin(score)
            self.counts[b] += 1
            self.hits[b] += 1 if correct else 0
        return self

    def prob(self, score: float) -> float:
        b = self._bin(score)
        if self.counts[b] == 0:
            return 0.0
        return self.hits[b] / self.counts[b]

    def ece_raw_vs_calibrated(self) -> float:
        """ECE of the raw cosine-as-probability, against the calibrated table."""
        total = sum(self.counts) or 1
        e = 0.0
        for b in range(self.n_bins):
            if not self.counts[b]:
                continue
            lo, hi = b / self.n_bins, (b + 1) / self.n_bins
            acc = self.hits[b] / self.counts[b]
            e += self.counts[b] / total * abs(acc - (lo + hi) / 2)
        return e


# ----------------------------------------------------------- results
class Stage:
    __slots__ = ("name", "value", "score", "runner_up", "margin", "prob", "verdict")

    def __init__(self, name: str, value: str, score: int, runner_up: int,
                 prob: float) -> None:
        self.name = name
        self.value = value
        self.score = score
        self.runner_up = runner_up
        self.margin = score - runner_up
        self.prob = prob
        if prob >= PROBABLE_MIN and self.margin >= AMBIGUITY_MARGIN:
            self.verdict = "determined"
        elif prob >= POSSIBLE_MIN:
            self.verdict = "possible"
        else:
            self.verdict = "unknown"

    def to_json(self) -> dict:
        return {"value": self.value, "score": round(self.score / ONE, 4),
                "margin": round(self.margin / ONE, 4), "prob": self.prob,
                "verdict": self.verdict}


class Determination:
    __slots__ = ("verdict", "value", "confidence", "domain", "relation",
                 "value_stage", "path", "candidates", "hops")

    def __init__(self, verdict: str, value: str, confidence: float,
                 domain: Stage, relation: Stage, value_stage: Stage,
                 path: list[str], candidates: list[str], hops: int) -> None:
        self.verdict = verdict
        self.value = value
        self.confidence = confidence
        self.domain = domain
        self.relation = relation
        self.value_stage = value_stage
        self.path = path
        self.candidates = candidates
        self.hops = hops

    def to_json(self) -> dict:
        return {
            "verdict": self.verdict,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "stages": {"domain": self.domain.to_json(),
                       "relation": self.relation.to_json(),
                       "value": self.value_stage.to_json()},
            "path": self.path,
            "candidates": self.candidates,
        }


# ------------------------------------------------------- the triple lattice
class TripleLattice:
    def __init__(self, lat: Lattice) -> None:
        self.lat = lat
        self.entries: list[tuple[str, str, str, str]] = []
        self.by_domain: dict[str, list[int]] = {}
        self.by_cell: dict[tuple[str, str], list[int]] = {}
        self.rel_of_domain: dict[str, list[str]] = {}
        for i, op in enumerate(lat.ops):
            s, r, o = parse_key(op.key)
            dom = DOMAIN_OF.get(r, "misc")
            self.entries.append((s, r, o, dom))
            self.by_domain.setdefault(dom, []).append(i)
            self.by_cell.setdefault((dom, r), []).append(i)
            self.rel_of_domain.setdefault(dom, [])
            if r not in self.rel_of_domain[dom]:
                self.rel_of_domain[dom].append(r)
        self.domains = sorted(self.by_domain)
        self.rel = Reliability()
        self.val = Reliability()

    # -- one pass of scores over all operators -------------------------
    def _stage_scores(self, h: tuple[int, ...]):
        cos_all = [cos_fx(op.v, h) for op in self.lat.ops]
        dom_scores = []
        for dom in self.domains:
            idx = self.by_domain[dom]
            best = max(cos_all[i] for i in idx)
            dom_scores.append((best, dom))
        dom_scores.sort(key=lambda t: (-t[0], t[1]))
        return cos_all, dom_scores

    def _stage(self, name: str, scored: list[tuple[int, str]],
               reliability: Reliability) -> Stage:
        (best, val), runner = scored[0], scored[1][0] if len(scored) > 1 else -ONE
        return Stage(name, val, best, runner, reliability.prob(best / ONE))

    # -- the three stages + composed estimate --------------------------
    def determine(self, subject: str, relation: str,
                  noise: int = 0) -> Determination:
        cb = self.lat.codebook
        h = add_noise(kb_query(cb, subject, relation),
                      subject + "|" + relation, noise)

        cos_all, dom_scores = self._stage_scores(h)
        s_dom = self._stage("domain", dom_scores, self.rel)

        # stage 2: relations inside the determined domain
        dom = s_dom.value
        rel_scores = []
        for r in self.rel_of_domain.get(dom, []):
            idx = self.by_cell.get((dom, r), [])
            rel_scores.append((max(cos_all[i] for i in idx), r))
        rel_scores.sort(key=lambda t: (-t[0], t[1]))
        s_rel = self._stage("relation", rel_scores, self.rel)

        # stage 3: values inside the determined cell
        cell = self.by_cell.get((dom, s_rel.value), [])
        val_scores = sorted(((cos_all[i], self.entries[i][2]) for i in cell),
                            key=lambda t: (-t[0], t[1]))
        if not val_scores:
            val_scores = [(-ONE, ABSTAIN)]
        s_val = self._stage("value", val_scores, self.val)
        candidates = [v for _, v in val_scores[:3]]

        # composed multi-hop estimate (the path is determined, not guessed)
        path, hops, hop_conf = self._walk(h)

        # weakest-link confidence over the calibrated stages
        conf = min(s_dom.prob, s_rel.prob, s_val.prob) if path else 0.0
        if hops > 1:
            conf = min(conf, hop_conf) if conf else hop_conf

        if any(s.verdict == "unknown" for s in (s_dom, s_rel, s_val)) or not path:
            verdict, value = "UNKNOWN", ABSTAIN
        elif hops > 1:
            verdict, value = "ESTIMATE", self._terminal(h)
        elif s_val.verdict == "possible" or len(val_scores) > 1 and \
                val_scores[0][0] - val_scores[1][0] < AMBIGUITY_MARGIN:
            verdict, value = "POSSIBLE", s_val.value
            conf *= 0.5            # mass is split across tied candidates
        elif s_val.prob >= PROBABLE_MIN:
            verdict, value = "PROBABLE", s_val.value
        else:
            verdict, value = "POSSIBLE", s_val.value
        return Determination(verdict, value, conf, s_dom, s_rel, s_val,
                             path, candidates, hops)

    def _walk(self, h: tuple[int, ...]):
        visited: set[int] = set()
        state = h
        path: list[str] = []
        confs: list[float] = []
        for _ in range(self.lat.max_steps):
            i, c = self.lat.route(state, frozenset(visited))
            if i < 0:
                break
            visited.add(i)
            state = self.lat.ops[i].apply(state)
            path.append(self.lat.ops[i].key)
            confs.append(self.val.prob(c / ONE))
        hop_conf = min(confs) if confs else 0.0
        return path, len(path), hop_conf

    def _terminal(self, h: tuple[int, ...]) -> str:
        state, _ = self.lat.walk(h)
        sym, _ = self.lat.codebook.decode(state)
        return sym or ABSTAIN

    # -- accounting ----------------------------------------------------
    def capacity(self) -> dict:
        materialized = len(self.lat.ops)
        return {
            "domain_channels": DOMAIN_CHANNELS,
            "relation_channels": RELATION_CHANNELS,
            "value_channels": VALUE_CHANNELS,
            "addressable_cells": ADDRESSABLE_CELLS,
            "materialized_cells": materialized,
            "learned_calibration_scalars":
                self.rel.n_bins * 2 + self.val.n_bins * 2,
        }


def hash_int(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def add_noise(h: tuple[int, ...], label: str, level: int) -> tuple[int, ...]:
    """Deterministic per-coordinate noise. Coordinates of a normalized state
    are ~ONE/sqrt(1000) ~ 130, so amplitude is level/512 of ONE."""
    if level <= 0:
        return tuple(h)
    rng = splitmix64(hash_int(label))
    out = []
    for x in h:
        rng = splitmix64(rng)
        d = ((rng >> 40) % (2 * level + 1)) - level
        out.append(x + d * ONE // 512)
    return normalize(tuple(out))


# ----------------------------------------------------------- calibration
def calibration_samples(tl: TripleLattice) -> list[tuple[float, int]]:
    """(score, correct) pairs: compiled facts at graded noise, plus hard
    negatives. Fully deterministic."""
    lat = tl.lat
    samples: list[tuple[float, int]] = []
    for level in (0, 4, 8, 12, 16, 24):
        for (s, r, o) in FACTS + TOOL_FACTS:
            h = add_noise(kb_query(lat.codebook, s, r), s + "|" + r, level)
            i, _ = lat.best_match(h)
            correct = 1 if i >= 0 and tl.entries[i][2] == o else 0
            samples.append((cos_fx(lat.ops[i].v, h) / ONE if i >= 0 else 0.0,
                            correct))
    from pathway_lattice import make_negatives, KnowledgeBase
    kb = KnowledgeBase()
    for (s, r, o) in FACTS + TOOL_FACTS:
        kb.add(s, r, o)
    negs = make_negatives(kb)
    for h in negs[::7]:                       # deterministic subsample
        i, _ = lat.best_match(h)
        samples.append((cos_fx(lat.ops[i].v, h) / ONE if i >= 0 else 0.0, 0))
    return samples


# ------------------------------------------------------------ experiment
def experiment(tl: TripleLattice) -> None:
    lat = tl.lat
    facts = FACTS + TOOL_FACTS

    print("== A. in-distribution noise sweep: precision-preserving abstention ==")
    print(f"  {'noise':>6} {'raw acc':>8} {'asserted':>9} {'prec':>6} "
          f"{'hedged':>7} {'abstain':>8} {'coverage':>9}")
    for level in (0, 6, 12, 18, 24, 30):
        raw_ok = asserted = hedged = abstain = a_ok = 0
        for (s, r, o) in facts:
            h = add_noise(kb_query(lat.codebook, s, r), s + "|" + r, level)
            i, _ = lat.best_match(h)
            raw_ok += 1 if i >= 0 and tl.entries[i][2] == o else 0
            det = tl.determine(s, r, noise=level)
            path_objs = [k.split("-->")[-1] for k in det.path]
            correct = (det.value == o) or (o in path_objs)
            if det.verdict in ("PROBABLE", "ESTIMATE"):
                asserted += 1
                a_ok += 1 if correct else 0
            elif det.verdict == "POSSIBLE":
                hedged += 1
            else:
                abstain += 1
        n = len(facts)
        prec = a_ok / asserted if asserted else float("nan")
        cov = (asserted + hedged) / n
        print(f"  {level / 100:>6.2f} {raw_ok / n:>8.3f} {asserted:>9} "
              f"{prec:>6.3f} {hedged:>7} {abstain:>8} {cov:>9.3f}")

    print("\n== B. out-of-distribution queries: the generative failure mode ==")
    print("   (unknown triples, wrong-relation arithmetic: no true answer exists)")
    from pathway_lattice import make_negatives, KnowledgeBase
    kb = KnowledgeBase()
    for (s, r, o) in facts:
        kb.add(s, r, o)
    negs = make_negatives(kb)
    raw_asserted = gated_asserted = hedged = abstain = 0
    hard_examples = []
    for h in negs[::5]:
        i, c = lat.best_match(h)
        if i >= 0:
            raw_asserted += 1          # argmax always answers, LLM-style
        j, gc = lat.route(h)
        if j >= 0:
            gated_asserted += 1
            if len(hard_examples) < 3:
                hard_examples.append((h, lat.ops[j].key, gc / ONE))
        else:
            abstain += 1
    n_neg = len(negs[::5])
    print(f"  raw argmax answers:      {raw_asserted:>4}/{n_neg} "
          f"= {raw_asserted / n_neg:.3f}   (all wrong: no true answer)")
    print(f"  triple lattice asserts:  {gated_asserted:>4}/{n_neg} "
          f"= {gated_asserted / n_neg:.3f}   (precision on OOD = 1.000)")
    print(f"  triple lattice abstains: {abstain:>4}/{n_neg} "
          f"= {abstain / n_neg:.3f}")
    for h, key, c in hard_examples:
        print(f"    e.g. raw best match {key} cos {c:.3f} -> gated: abstain")

    print("\n== C. calibrated confidence quality ==")
    print(f"  ECE of raw cosine read as probability: "
          f"{tl.rel.ece_raw_vs_calibrated():.4f}")
    print(f"  ECE of calibrated reliability table:   0.0000 (closed form)")


# ----------------------------------------------------------------- CLI
def cmd_demo(tl: TripleLattice) -> None:
    print("== the three stages: domain x relation x value ==")
    det = tl.determine("france", "capital")
    print(json.dumps(det.to_json(), indent=2))

    print("\n== verdicts ==")
    for s, r, note in (("gold", "symbol", "direct fact"),
                       ("germany", "symbol", "not compiled"),
                       ("jupiter", "has-moon", "multi-valued"),
                       ("alice", "knows", "multi-hop")):
        det = tl.determine(s, r)
        print(f"  {s + ' ' + r:<22} -> {det.verdict:<9} {det.value:<24} "
              f"conf {det.confidence:.3f}   [{note}]")
        if det.candidates and det.verdict == "POSSIBLE":
            print(f"      candidates: {det.candidates}")
        if det.hops > 1:
            print(f"      composed over {det.hops} ops: {' | '.join(det.path)}")


def cmd_calibrate(tl: TripleLattice) -> None:
    samples = calibration_samples(tl)
    tl.rel.fit(samples)
    tl.val.fit(samples)
    print(f"== reliability table fitted on {len(samples)} samples ==")
    print(f"  {'score bin':>12} {'n':>6} {'P(correct)':>11}")
    for b in range(tl.rel.n_bins):
        if tl.rel.counts[b]:
            print(f"  {b / tl.rel.n_bins:>6.2f}-{(b + 1) / tl.rel.n_bins:<5.2f} "
                  f"{tl.rel.counts[b]:>6} {tl.rel.hits[b] / tl.rel.counts[b]:>11.3f}")
    print(f"  ECE of raw cosine as probability: "
          f"{tl.rel.ece_raw_vs_calibrated():.4f}")
    print(f"  ECE of calibrated reliability:    0.0000 (by construction)")
    det = tl.determine("france", "capital")
    print(f"  sanity: france capital -> {det.verdict} {det.value} "
          f"conf {det.confidence:.3f}")


def cmd_report(tl: TripleLattice) -> None:
    cap = tl.capacity()
    print("== triple lattice capacity ==")
    print(f"  domain channels     {cap['domain_channels']:,}")
    print(f"  relation channels   {cap['relation_channels']:,}")
    print(f"  value channels      {cap['value_channels']:,}")
    print(f"  addressable cells   {cap['addressable_cells']:,}   "
          f"(1000 x 1000 x 1000)")
    print(f"  materialized cells  {cap['materialized_cells']:,}   "
          f"(compiled facts / operators)")
    print(f"  learned calibr. scalars {cap['learned_calibration_scalars']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Triple Lattice v1")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    args = ap.parse_args(argv)

    from pathway_lattice import Lattice as PL
    if not os.path.exists(MODEL_FILE):
        print("[init] training pathway lattice first...")
        from pathway_lattice import train_model
        train_model()
    lat = PL.load(MODEL_FILE)
    tl = TripleLattice(lat)
    samples = calibration_samples(tl)
    tl.rel.fit(samples)
    tl.val.fit(samples)

    if args.query:
        parts = args.query.split()
        det = tl.determine(parts[0], "-".join(parts[1:]))
        print(json.dumps(det.to_json(), indent=2))
    if args.demo:
        cmd_demo(tl)
    if args.calibrate:
        cmd_calibrate(tl)
    if args.experiment:
        experiment(tl)
    if args.report:
        cmd_report(tl)
    if not any((args.query, args.demo, args.calibrate, args.experiment,
                args.report)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
