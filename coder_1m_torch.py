#!/usr/bin/env python3
"""TinyCoder-1M — PyTorch port (GPU-friendly) of coder_1m.py — same architecture
and same weight format (the stdlib `Model.params` JSON structure, so the stdlib
scripts load it).

Usage (same as coder_1m.py, plus --device):
    python3 coder_1m_torch.py --train --steps 500 --device cuda --resume --weights coder_coding_weights.json
    python3 coder_1m_torch.py --generate --prompt "def fizzbuzz(n):" --n 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coder_1m import BOS, D, EOS, FFN, L, SEED, T, V, build_corpus, split_ids


class Brain(torch.nn.Module):
    """TinyCoder-1M (stdlib architecture), torch port — same math as coder_1m.Model."""

    def __init__(self, d=D, L=L, t=T, v=V, heads=2, seed=SEED):
        super().__init__()
        self.d, self.L, self.t, self.v, self.heads = d, L, t, v, heads
        self.hd = d // heads
        self.sqrt_hd = math.sqrt(self.hd)
        scale = 1.0 / math.sqrt(d)
        self.tok_emb = torch.nn.Parameter(torch.empty(v, d)); torch.nn.init.normal_(self.tok_emb, 0.0, scale)
        self.pos_emb = torch.nn.Parameter(torch.empty(t, d)); torch.nn.init.normal_(self.pos_emb, 0.0, 0.02)
        self.blocks = torch.nn.ModuleList()
        for _ in range(L):
            blk = torch.nn.Module()
            for name in ("Wq", "Wk", "Wv", "Wo"):
                setattr(blk, name, torch.nn.Parameter(torch.empty(d, d)))
                torch.nn.init.normal_(getattr(blk, name), 0.0, scale)
            blk.W1 = torch.nn.Parameter(torch.empty(FFN, d)); torch.nn.init.normal_(blk.W1, 0.0, scale)
            blk.W2 = torch.nn.Parameter(torch.empty(d, FFN)); torch.nn.init.normal_(blk.W2, 0.0, scale)
            blk.ln1_w = torch.nn.Parameter(torch.ones(d))
            blk.ln1_b = torch.nn.Parameter(torch.zeros(d))
            blk.ln2_w = torch.nn.Parameter(torch.ones(d))
            blk.ln2_b = torch.nn.Parameter(torch.zeros(d))
            self.blocks.append(blk)
        self.ln_f_w = torch.nn.Parameter(torch.ones(d))
        self.ln_f_b = torch.nn.Parameter(torch.zeros(d))

    def forward(self, ids):
        S = len(ids)
        idx = torch.tensor(ids, dtype=torch.long, device=self.tok_emb.device)
        cur = self.tok_emb[idx].clone()
        ar = torch.arange(S, device=cur.device)
        m = ar < self.t
        cur[m] = cur[m] + self.pos_emb[ar[m]]
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=cur.device), diagonal=1)
        for blk in self.blocks:
            ln1 = F.layer_norm(cur, (self.d,), blk.ln1_w, blk.ln1_b, 1e-5)
            q = ln1 @ blk.Wq.T
            k = ln1 @ blk.Wk.T
            v = ln1 @ blk.Wv.T
            context = torch.zeros_like(cur)
            for h in range(self.heads):
                lo = h * self.hd
                qh, kh, vh = q[:, lo:lo + self.hd], k[:, lo:lo + self.hd], v[:, lo:lo + self.hd]
                scores = (qh @ kh.T) / self.sqrt_hd
                scores = scores.masked_fill(causal, float("-inf"))
                context[:, lo:lo + self.hd] = torch.softmax(scores, dim=-1) @ vh
            h1 = cur + context @ blk.Wo.T
            ln2 = F.layer_norm(h1, (self.d,), blk.ln2_w, blk.ln2_b, 1e-5)
            a = F.gelu(ln2 @ blk.W1.T)
            cur = h1 + a @ blk.W2.T
        lf = F.layer_norm(cur, (self.d,), self.ln_f_w, self.ln_f_b, 1e-5)
        return lf @ self.tok_emb.T

    @torch.no_grad()
    def loss_on(self, windows):
        """Mean NLL over all positions (matches coder_1m.Model.loss_on)."""
        self.eval()
        total, count = 0.0, 0
        for (x, y) in windows:
            logits = self.forward(x)
            total += F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum").item()
            count += len(x)
        return total / max(1, count)

    @torch.no_grad()
    def eval_perplexity(self, ids, seq_len=8, max_positions=256, seed=0):
        """Matches coder_1m.Model.eval_perplexity."""
        self.eval()
        rng = random.Random(seed)
        n = len(ids) - seq_len - 1
        if n <= 0:
            return float("inf")
        pos = list(range(0, n + 1))
        if len(pos) > max_positions:
            pos = rng.sample(pos, max_positions)
        nll, count = 0.0, 0
        for p in sorted(pos):
            x = ids[p:p + seq_len]
            y = ids[p + 1:p + seq_len + 1]
            logits = self.forward(x)
            nll += F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum").item()
            count += len(x)
        return math.exp(nll / max(1, count))

    @torch.no_grad()
    def generate(self, prompt_ids, n=32, greedy=True):
        ids = list(prompt_ids)
        out = []
        self.eval()
        for _ in range(n):
            logits = self.forward(ids)
            nxt = int(torch.argmax(logits[-1]).item())
            out.append(nxt)
            ids.append(nxt)
            if nxt == EOS:
                break
        return out


def train(model, train_ids, val_ids, steps, batch=2, seq_len=8, lr=1e-3,
          log_every=5, eval_every=20):
    rng = random.Random(999)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        windows = []
        for _ in range(batch):
            p = rng.randrange(0, len(train_ids) - seq_len - 2)
            windows.append((train_ids[p:p + seq_len], train_ids[p + 1:p + seq_len + 1]))
        opt.zero_grad()
        losses = []
        count = 0
        for (x, y) in windows:
            logits = model.forward(x)
            losses.append(F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum"))
            count += len(x)
        loss = torch.stack(losses).sum() / max(1, count)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"[train] step {step}/{steps} loss={loss.item():.4f} elapsed={time.time() - t0:.0f}s", flush=True)
        if step % eval_every == 0 or step == steps:
            print(f"[eval] step {step} val PPL={model.eval_perplexity(val_ids, seq_len=seq_len, max_positions=128):.2f}", flush=True)
    return model


def params_dict(model):
    """Same structure as coder_1m.Model.params (stdlib weight format)."""
    tl = lambda t: t.detach().cpu().tolist()
    return {
        "tok_emb": tl(model.tok_emb),
        "pos_emb": tl(model.pos_emb),
        "blocks": [{
            "Wq": tl(blk.Wq), "Wk": tl(blk.Wk), "Wv": tl(blk.Wv), "Wo": tl(blk.Wo),
            "W1": tl(blk.W1), "W2": tl(blk.W2),
            "ln1_w": tl(blk.ln1_w), "ln1_b": tl(blk.ln1_b),
            "ln2_w": tl(blk.ln2_w), "ln2_b": tl(blk.ln2_b),
        } for blk in model.blocks],
        "ln_f_w": tl(model.ln_f_w),
        "ln_f_b": tl(model.ln_f_b),
    }


def save_weights(model, path):
    with open(path, "w") as f:
        json.dump(params_dict(model), f)


def load_weights(model, path):
    with torch.no_grad():
        d = json.load(open(path))
        model.tok_emb.copy_(torch.tensor(d["tok_emb"], dtype=torch.float32))
        model.pos_emb.copy_(torch.tensor(d["pos_emb"], dtype=torch.float32))
        for blk, b in zip(model.blocks, d["blocks"]):
            for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "ln1_w", "ln1_b", "ln2_w", "ln2_b"):
                getattr(blk, k).copy_(torch.tensor(b[k], dtype=torch.float32))
        model.ln_f_w.copy_(torch.tensor(d["ln_f_w"], dtype=torch.float32))
        model.ln_f_b.copy_(torch.tensor(d["ln_f_b"], dtype=torch.float32))
    return model


def main():
    ap = argparse.ArgumentParser(description="TinyCoder-1M (torch port)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--weights", default="coder_coding_weights.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prompt", default="def fizzbuzz(n):")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Brain().to(device)
    if args.resume and os.path.exists(args.weights):
        load_weights(model, args.weights)
        print(f"resumed weights <- {args.weights}", flush=True)
    if args.generate:
        ids = [BOS] + list(args.prompt.encode("utf-8", "replace"))
        gen = model.generate(ids, n=args.n)
        print(args.prompt + bytes(t for t in gen if t < 256).decode("utf-8", "replace"), flush=True)
        return
    if args.train:
        corpus = build_corpus()
        train_ids, val_ids = split_ids(corpus)
        print(f"training on {device}: corpus={len(corpus)} chars; steps={args.steps}", flush=True)
        train(model, train_ids, val_ids, steps=args.steps, batch=args.batch,
              seq_len=args.seq_len, lr=args.lr)
        save_weights(model, args.weights)
        print(f"saved weights -> {args.weights}", flush=True)


if __name__ == "__main__":
    main()
