#!/usr/bin/env python3
"""
Brain11 Base — PyTorch port of the TinyCoder architecture scaled to ~11M params.

Architecture (from coder_1m.py, scaled up):
- Byte-level vocab (260 = 256 bytes + BOS/EOS/PAD/UNK)
- Context length T=128
- d=512, L=4 transformer blocks, FFN=2048, heads=4
- Pre-LN + causal attention + GELU FFN
- ~12.8M parameters

Usage:
    from brain11_base import Brain, V, BOS, EOS, D, L, T
    model = Brain().to("cuda")
"""

from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F

# Constants (~11M params target)
V = 260          # 256 bytes + BOS/EOS/PAD/UNK
BOS, EOS, PAD, UNK = 256, 257, 258, 259
D = 512          # model dim
L = 4            # transformer blocks
HEADS = 4
HD = D // HEADS  # 128
FFN = 2048       # FFN hidden dim
T = 128          # context length
SEED = 20261130


def total_params(d=D, L=L, v=V, t=T, ffn=FFN):
    """Exact parameter count of the architecture."""
    return L * (4 * d * d + 2 * d * ffn + 4 * d) + 2 * d + v * d + t * d


class Brain(torch.nn.Module):
    """Base Brain11 model (torch port of TinyCoder with d=512, L=4)."""

    def __init__(self, d=D, L=L, t=T, v=V, heads=HEADS, ffn=FFN, seed=SEED):
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
            blk.W1 = torch.nn.Parameter(torch.empty(ffn, d)); torch.nn.init.normal_(blk.W1, 0.0, scale)
            blk.W2 = torch.nn.Parameter(torch.empty(d, ffn)); torch.nn.init.normal_(blk.W2, 0.0, scale)
            blk.ln1_w = torch.nn.Parameter(torch.ones(d))
            blk.ln1_b = torch.nn.Parameter(torch.zeros(d))
            blk.ln2_w = torch.nn.Parameter(torch.ones(d))
            blk.ln2_b = torch.nn.Parameter(torch.zeros(d))
            self.blocks.append(blk)
        self.ln_f_w = torch.nn.Parameter(torch.ones(d))
        self.ln_f_b = torch.nn.Parameter(torch.zeros(d))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids):
        S = len(ids)
        idx = torch.tensor(ids, dtype=torch.long, device=self.tok_emb.device)
        cur = self.tok_emb[idx].clone()
        ar = torch.arange(S, device=cur.device)
        m = ar < self.t
        if m.any():
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


def encode_text(text: str) -> list[int]:
    """Encode a UTF-8 string as byte tokens with BOS."""
    return [BOS] + list(text.encode("utf-8", "replace"))


def decode_text(ids: list[int]) -> str:
    """Decode byte tokens back to UTF-8 string."""
    return bytes(t for t in ids if t < 256).decode("utf-8", "replace")


def train_model(model, train_ids, val_ids, steps, batch=4, seq_len=128, lr=1e-3,
                log_every=5, eval_every=25):
    """Train the model with AdamW on random windows."""
    rng = random.Random(20261130)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        windows = []
        for _ in range(batch):
            p = rng.randrange(0, max(1, len(train_ids) - seq_len - 2))
            windows.append((train_ids[p:p + seq_len], train_ids[p + 1:p + seq_len + 1]))
        opt.zero_grad()
        loss = 0.0
        for (x, y) in windows:
            logits = model.forward(x)
            loss += F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum")
        loss /= max(1, len(windows) * seq_len)
        loss.backward()
        opt.step()
        sched.step()
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"[train] step {step}/{steps} loss={loss.item():.4f} elapsed={time.time() - t0:.0f}s", flush=True)
        if (step % eval_every == 0 or step == steps) and val_ids:
            print(f"[eval] step {step} val PPL={eval_perplexity(model, val_ids):.2f}", flush=True)
    return model


@torch.no_grad()
def eval_perplexity(model, ids, seq_len=128, max_positions=64, seed=0):
    model.eval()
    rng = random.Random(seed)
    n = max(1, len(ids) - seq_len - 1)
    pos = list(range(0, min(n, max_positions)))
    if len(pos) > max_positions:
        pos = rng.sample(pos, max_positions)
    nll, count = 0.0, 0
    for p in sorted(pos):
        x = ids[p:p + seq_len]
        y = ids[p + 1:p + seq_len + 1]
        logits = model.forward(x)
        nll += F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum").item()
        count += len(x)
    return math.exp(nll / max(1, count))
