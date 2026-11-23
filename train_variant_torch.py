#!/usr/bin/env python3
"""Train a TinyCoder-1M variant on a purpose-specific corpus — the GPU
(torch) port of train_variant.py. Saves in the stdlib weight format
(same as coder_1m.py / coder_1m_torch.py).

Usage:
    python3 train_variant_torch.py --corpus corpus_debug.txt \
        --weights coder_debug_weights.json --steps 500 --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coder_1m import split_ids
from coder_1m_torch import Brain, load_weights, save_weights, train


def main():
    ap = argparse.ArgumentParser(description="Train a TinyCoder-1M variant on a corpus file (GPU).")
    ap.add_argument("--corpus", required=True, help="path to the corpus text file")
    ap.add_argument("--weights", required=True, help="output weights file")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    text = Path(args.corpus).read_text(errors="replace")
    train_ids, val_ids = split_ids(text)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Brain().to(device)
    if args.resume and os.path.exists(args.weights):
        load_weights(model, args.weights)
        print(f"resumed weights <- {args.weights}", flush=True)
    print(f"training on {device}: corpus={len(text)} chars; steps={args.steps}", flush=True)
    train(model, train_ids, val_ids, steps=args.steps)
    save_weights(model, args.weights)
    print(f"saved variant weights -> {args.weights}", flush=True)


if __name__ == "__main__":
    main()
