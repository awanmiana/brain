#!/usr/bin/env python3
"""
Grow v1 -- the self-growth loop: "if I connect it to an agent that feeds it
data, does it get trained?"

Yes — in this architecture, *training == compilation*. There is no gradient
descent loop to run: every new datum is fitted in closed form as one rank-1
operator (+2,000 parameters), and the very next query can use it. An agent
that perceives, decides, and writes facts back is a training loop.

    perceive -> decide (lattice) -> act -> learn_fact() -> +2,000 params

Contradictions are not silently swallowed: learn_fact() reports
'conflict' when the same query acquires a second answer, so an upstream
policy (or a human) can retract. First-compiled wins until retracted.

Usage:
    python3 grow.py --demo          # stream facts into the brain, watch it grow
    python3 grow.py --stream N      # grow from a simulated agent stream
"""
from __future__ import annotations

import argparse
import json
import sys

from pathway_lattice import (
    Lattice, MODEL_FILE, learn_fact, kb_query, normalize, REL)


STREAM = [                     # what an agent might perceive and write back
    ("python", "has-version", "3.14"),
    ("mars", "has-moon", "phobos"),
    ("mars", "has-moon", "deimos"),
    ("tesla", "founded-by", "elon-musk"),
    ("ethereum", "created-by", "vitalik-buterin"),
    ("france", "capital", "lyon"),          # <-- contradiction on purpose
    ("bitcoin", "created-by", "satoshi-nakamoto"),
    ("earth", "has-moon", "luna"),
    ("python", "has-version", "3.15"),      # <-- multi-valued update
]


def cmd_demo(lat: Lattice, save: bool = False) -> None:
    print("== self-growth: training == compilation ==")
    print(f"  start: {len(lat.ops)} ops, {lat.param_count():,} learned params")
    for (s, r, o) in STREAM:
        res = learn_fact(lat, s, r, o)
        mark = {"learned": "+", "learned-multi": "+", "conflict": "!",
                "redundant": "="}[res["status"]]
        extra = ""
        if res["status"] == "conflict":
            extra = f"   (already known: {res['existing']})"
        elif res["status"] == "learned-multi":
            extra = "   (multi-valued relation)"
        print(f"  {mark} learn ({s}, {r}, {o}) -> {res['status']:<12} "
              f"params {res['params']:,}{extra}")

    print("\n  newly learned facts are immediately answerable:")
    for (s, r) in (("python", "has-version"), ("tesla", "founded-by"),
                   ("bitcoin", "created-by"), ("mars", "has-moon")):
        h = kb_query(lat.codebook, s, r)
        det = lat.determine_next(h)
        full = lat.determine_path(h, 1)
        print(f"    {s} {r:<12} -> {det.value}  conf {det.cos / 4096:.3f}")

    print("\n  contradiction policy: first-compiled wins, both kept, flagged:")
    h = kb_query(lat.codebook, "france", "capital")
    det = lat.determine_next(h)
    print(f"    france capital -> {det.value}   (the stream also offered lyon)")

    print("\n  multi-valued relations keep every answer (ambiguity is surfaced):")
    h = kb_query(lat.codebook, "mars", "has-moon")
    matched = [op.key.split("-->")[-1] for op in lat.ops
               if op.key.startswith("mars--has-moon-->")]
    print(f"    mars has-moon -> {matched}   "
          f"(triple lattice would answer POSSIBLE with both as candidates)")

    print(f"\n  end: {len(lat.ops)} ops, {lat.param_count():,} learned params, "
          f"state hash {lat.state_hash()[:16]}...")
    if save:
        lat.save()
        print(f"  saved grown model to {MODEL_FILE}")
    else:
        print("  (in-memory only; pass --save to persist the grown model)")


def cmd_stream(lat: Lattice, n: int) -> None:
    print(f"== simulated agent stream: {n} facts ==")
    base = lat.param_count()
    for i in range(n):
        s, r, o = f"entity-{i}", "relates-to", f"target-{i}"
        res = learn_fact(lat, s, r, o)
        if i % max(1, n // 10) == 0:
            print(f"  [{i:>5}] {res['status']}  params {res['params']:,}")
    print(f"  growth: {base:,} -> {lat.param_count():,} learned params "
          f"(+{lat.param_count() - base:,})")
    print(f"  per-fact cost: 2,000 params, closed form, no epochs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grow v1")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    import os
    if not os.path.exists(MODEL_FILE):
        from pathway_lattice import train_model
        train_model()
    lat = Lattice.load(MODEL_FILE)

    if args.demo:
        cmd_demo(lat, save=args.save)
    if args.stream:
        cmd_stream(lat, args.stream)
    if not any((args.demo, args.stream)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
