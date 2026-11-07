#!/usr/bin/env python3
"""
Hybrid front-end v1 -- the cheap half of the hybrid.

Question: do we need billions of parameters to understand language?
Answer: not for *bounded* language. This is a ZERO-parameter deterministic
parser (grammar + lexicon, no weights, nothing trained) that maps plain
English questions to structured lattice queries, then lets the Triple
Lattice decide the verdict.

    "what is the capital of france"  ->  (france, capital)  ->  PROBABLE paris
    "what is 7 plus 5"               ->  arithmetic          ->  PROBABLE n12
    "what tool do i need for weather"->  (ask-weather,calls) ->  ESTIMATE ...
    "why is the sky blue"            ->  UNPARSED (front-end abstains too)

In production the front-end can be swapped for a borrowed open-weight model
(Qwen et al., inference only) or a small fine-tuned parser; the decision
layer stays the same deterministic lattice. The division of labor is the
point: language understanding may be statistical, decisions must not be.

Usage:
    python3 hybrid_demo.py --demo
    python3 hybrid_demo.py --ask "who created python"
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from pathway_lattice import MODEL_FILE, Lattice, normalize, kb_query, REL
from triple_lattice import TripleLattice
from pathway_seed import FACTS, TOOL_FACTS

# ------------------------------------------------------------- lexicon
# Every known symbol, longest-match-first. Built from the compiled corpus.
SYMBOLS = sorted({s for s, _, _ in FACTS + TOOL_FACTS} |
                 {o for _, _, o in FACTS + TOOL_FACTS},
                 key=lambda s: (-len(s), s))

# relation keywords, checked in order (first match wins)
RELATION_RULES: list[tuple[str, str]] = [
    ("capital", "capital"),
    ("symbol", "symbol"),
    ("created", "created-by"),
    ("invented", "invented-by"),
    ("orbit", "orbits"),
    ("moon", "has-moon"),
    ("teach", "teaches"),
    ("know", "knows"),
    ("develop", "developed"),
    ("discover", "discovered"),
    ("design", "designed"),
    ("observ", "observed"),
    ("compos", "composed"),
    ("wrote", "wrote"),
    ("write", "wrote"),
    ("boil", "boils-at"),
    ("freeze", "freezes-at"),
    ("speed", "travels-at"),
    ("travel", "travels-at"),
    ("value of pi", "approx"),
    ("approx", "approx"),
    ("encod", "encodes"),
    ("made of", "is-made-of"),
    ("is a", "is-a"),
    ("type", "type"),
]

# tool intents: keyword -> intent symbol
TOOL_RULES: list[tuple[str, str]] = [
    ("weather", "ask-weather"),
    ("time", "ask-time"),
    ("calc", "calculate"),
    ("math", "calculate"),
    ("search", "search-web"),
    ("web", "search-web"),
    ("file", "read-file"),
    ("read", "read-file"),
    ("message", "send-message"),
    ("send", "send-message"),
    ("translat", "translate"),
    ("flight", "book-flight"),
    ("book", "book-flight"),
]

# intent -> the tool it calls (from the compiled tool facts)
TOOL_OF = {s: o for s, r, o in TOOL_FACTS if r == "calls"}

ARITH_RE = re.compile(r"(\d+)\s*(\+|plus|minus|times|\*|x)\s*(\d+)")
ARITH_OP = {"+": "plus", "plus": "plus", "-": "minus", "minus": "minus",
            "*": "times", "x": "times", "times": "times"}


class Parse:
    __slots__ = ("subject", "relation", "kind", "raw")

    def __init__(self, subject: str, relation: str, kind: str, raw: str) -> None:
        self.subject = subject
        self.relation = relation
        self.kind = kind          # "fact" | "arith" | "tool"
        self.raw = raw

    def to_json(self) -> dict:
        return {"subject": self.subject, "relation": self.relation,
                "kind": self.kind}


def find_symbol(text: str) -> str | None:
    dashed = text.replace(" ", "-")
    for sym in SYMBOLS:
        if re.search(r"(?<![a-z0-9-])" + re.escape(sym.lower()) + r"(?![a-z0-9-])",
                     text) or ("-" + sym.lower() + "-") in ("-" + dashed + "-"):
            return sym
    return None


def parse(question: str) -> Parse | None:
    q = question.lower().strip().rstrip("?.")

    m = ARITH_RE.search(q)
    if m:
        return Parse(m.group(1), ARITH_OP[m.group(2)], "arith", question)

    if "tool" in q or any(k in q for k, _ in TOOL_RULES):
        for k, intent in TOOL_RULES:
            if k in q:
                if "need" in q and intent in TOOL_OF:
                    return Parse(TOOL_OF[intent], "needs", "tool", question)
                return Parse(intent, "calls", "tool", question)

    rel = None
    for k, r in RELATION_RULES:
        if k in q:
            rel = r
            break
    sym = find_symbol(q)
    if rel and sym:
        return Parse(sym, rel, "fact", question)
    return None


# -------------------------------------------------------------- driver
def answer(tl: TripleLattice, question: str) -> dict:
    p = parse(question)
    if p is None:
        return {"question": question, "verdict": "UNPARSED",
                "note": "front-end abstains: out of grammar/lexicon"}
    if p.kind == "arith":
        a, b = int(p.subject), _second_operand(p.raw)
        det = _arith(tl, a, b, p.relation)
    else:
        det = tl.determine(p.subject, p.relation)
    out = {"question": question, "parse": p.to_json(), "verdict": det.verdict,
           "value": det.value, "confidence": round(det.confidence, 3),
           "path": det.path}
    if det.candidates and det.verdict == "POSSIBLE":
        out["candidates"] = det.candidates
    return out


def _second_operand(q: str) -> int:
    m = ARITH_RE.search(q.lower())
    return int(m.group(3)) if m else 0


def _arith(tl: TripleLattice, a: int, b: int, op: str):
    from triple_lattice import Determination
    from pathway_lattice import ONE
    if op != "plus":
        # the compiled table is addition-only: honest abstention
        return Determination("UNKNOWN", "UNKNOWN", 0.0, None, None, None, [], [], 0)
    h = normalize([p + q + r for p, q, r in
                   zip(tl.lat.codebook.vec(f"n{a}"),
                       tl.lat.codebook.vec(f"n{b}"),
                       tl.lat.codebook.vec(REL + "plus"))])
    det = tl.lat.determine_next(h)
    path, hops, _ = tl._walk(h)
    verdict = "PROBABLE" if not det.abstained else "UNKNOWN"
    return Determination(verdict, det.value,
                         tl.val.prob(det.cos / ONE) if not det.abstained else 0.0,
                         None, None, None, det.path or path, [], 1)


DEMO_QUESTIONS = [
    "what is the capital of france",
    "what is the symbol of gold",
    "who created python",
    "what does the moon orbit",
    "what is a dog",
    "who did socrates teach",
    "who does alice know",
    "what is 7 plus 5",
    "what is 3 minus 5",
    "what tool do i need for weather",
    "what does the weather tool need",
    "why is the sky blue",
    "tell me about quantum entanglement",
]


def cmd_demo(tl: TripleLattice) -> None:
    print("== zero-parameter English front-end -> triple lattice ==")
    for q in DEMO_QUESTIONS:
        out = answer(tl, q)
        v = out["verdict"]
        if v == "UNPARSED":
            print(f"  {q:<38} -> UNPARSED (front-end abstains)")
            continue
        extra = f"   candidates {out.get('candidates')}" if "candidates" in out else ""
        if v == "ESTIMATE" and out.get("path"):
            extra += f"   path {' | '.join(out['path'])}"
        print(f"  {q:<38} -> {v:<9} {out['value']:<22} "
              f"conf {out['confidence']}{extra}")
    print("\n  parameter count of this front-end: 0 (grammar + lexicon)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid front-end v1")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ask", type=str, default=None)
    args = ap.parse_args(argv)

    from pathway_lattice import MODEL_FILE
    import os
    if not os.path.exists(MODEL_FILE):
        from pathway_lattice import train_model
        train_model()
    lat = Lattice.load(MODEL_FILE)
    tl = TripleLattice(lat)
    from triple_lattice import calibration_samples
    samples = calibration_samples(tl)
    tl.rel.fit(samples)
    tl.val.fit(samples)

    if args.ask:
        print(json.dumps(answer(tl, args.ask), indent=2))
    if args.demo:
        cmd_demo(tl)
    if not any((args.ask, args.demo)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
