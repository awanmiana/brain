#!/usr/bin/env python3
"""Train a 1M expert on a purpose-specific corpus — GPU trainer for text and
binary (vision/audio) corpora. Saves in the stdlib weight format
(compatible with coder_1m.py, coder_1m_torch.py, train_variant_torch.py).

Usage:
    python3 train_expert_torch.py --corpus corpus_vision.bin --weights coder_vision_weights.json --steps 500 --device cuda
    python3 train_expert_torch.py --corpus corpus_audio.bin --weights coder_audio_weights.json --steps 500 --device cuda
    python3 train_expert_torch.py --corpus corpus_english.txt --weights coder_english_weights.json --steps 500 --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coder_1m import BOS, EOS
from coder_1m_torch import Brain, load_weights, save_weights, train


def load_ids(path):
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"empty corpus: {path}")
    return [BOS] + list(data) + [EOS]


def split_train_val(ids, val_frac=0.10):
    n_val = max(256, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]


def main():
    ap = argparse.ArgumentParser(description="Train a 1M expert (text or binary corpus) on the GPU.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    ids = load_ids(args.corpus)
    train_ids, val_ids = split_train_val(ids)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Brain().to(device)
    if args.resume and os.path.exists(args.weights):
        load_weights(model, args.weights)
        print(f"resumed weights <- {args.weights}", flush=True)
    print(f"training on {device}: corpus={len(ids)} bytes; steps={args.steps}", flush=True)
    train(model, train_ids, val_ids, steps=args.steps)
    save_weights(model, args.weights)
    print(f"saved expert weights -> {args.weights}", flush=True)


if __name__ == "__main__":
    main()
