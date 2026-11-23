#!/usr/bin/env python3
"""Train a TinyCoder-1M variant on a purpose-specific corpus file.

Usage:
    python3 train_variant.py --corpus corpus_debug.txt \
        --weights coder_debug_weights.json --steps 500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coder_1m import Model, save_weights, split_ids, train


def main():
    ap = argparse.ArgumentParser(description="Train a TinyCoder-1M variant on a corpus file.")
    ap.add_argument("--corpus", required=True, help="path to the corpus text file")
    ap.add_argument("--weights", required=True, help="output weights file")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261129)
    args = ap.parse_args()
    text = Path(args.corpus).read_text(errors="replace")
    train_ids, val_ids = split_ids(text)
    print(f"corpus: {len(text)} chars; steps={args.steps}", flush=True)
    model = Model(seed=args.seed)
    train(model, train_ids, val_ids, steps=args.steps)
    save_weights(model, args.weights)
    print(f"saved variant weights -> {args.weights}", flush=True)


if __name__ == "__main__":
    main()
