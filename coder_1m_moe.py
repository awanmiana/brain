#!/usr/bin/env python3
"""TinyCoder Brain AI model — a Mixture-of-Experts version of TinyCoder-1M.

Architecture:
- Same embeddings, blocks layout (minus W1/W2), and final layernorm as
  TinyCoder-1M (tok_emb, pos_emb, ln_f, L blocks, D dim, V vocab).
- Each block: shared attention (Wq/Wk/Wv/Wo + LN1/LN2) plus a Mixture of
  Experts over the block's MLP: N_EXPERTS experts, each with its own W1/W2;
  a router (Linear d -> N_EXPERTS, no bias) produces per-expert probabilities;
  the output is the top-2 weighted mixture of the selected experts' outputs.
- Load-balancing aux loss: n_experts * sum(f_i * P_i) (f_i = fraction of
  tokens dispatched to expert i, P_i = mean router probability).

Training: torch.optim.Adam (bias-corrected, like coder_1m's defaults),
autograd through the router, aux loss added to the NLL with AUX_COEFF.

Usage:
    python3 coder_1m_moe.py --train --steps 2000 --device cuda --resume
    python3 coder_1m_moe.py --generate --prompt "def fizzbuzz(n):" --n 30
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

N_EXPERTS = 4
TOPK = 2
AUX_COEFF = 0.01


class MoEBlock(torch.nn.Module):
    def __init__(self, d=D, ffn=FFN, heads=2, n_experts=N_EXPERTS, topk=TOPK, seed=SEED):
        super().__init__()
        scale = 1.0 / math.sqrt(d)
        self.d, self.heads = d, heads
        self.Wq = torch.nn.Parameter(torch.empty(d, d)); torch.nn.init.normal_(self.Wq, 0.0, scale)
        self.Wk = torch.nn.Parameter(torch.empty(d, d)); torch.nn.init.normal_(self.Wk, 0.0, scale)
        self.Wv = torch.nn.Parameter(torch.empty(d, d)); torch.nn.init.normal_(self.Wv, 0.0, scale)
        self.Wo = torch.nn.Parameter(torch.empty(d, d)); torch.nn.init.normal_(self.Wo, 0.0, scale)
        self.ln1_w = torch.nn.Parameter(torch.ones(d))
        self.ln1_b = torch.nn.Parameter(torch.zeros(d))
        self.ln2_w = torch.nn.Parameter(torch.ones(d))
        self.ln2_b = torch.nn.Parameter(torch.zeros(d))
        self.n_experts = n_experts
        self.topk = topk
        self.experts = torch.nn.ModuleList()
        for _ in range(n_experts):
            e = torch.nn.Module()
            e.W1 = torch.nn.Parameter(torch.empty(ffn, d)); torch.nn.init.normal_(e.W1, 0.0, scale)
            e.W2 = torch.nn.Parameter(torch.empty(d, ffn)); torch.nn.init.normal_(e.W2, 0.0, scale)
            self.experts.append(e)
        self.router = torch.nn.Linear(d, n_experts, bias=False)
        torch.nn.init.normal_(self.router.weight, 0.0, scale)

    def _attention(self, cur):
        # Shared attention — same math as coder_1m (W[out, in], y = W x).
        S = cur.shape[0]
        hd = self.d // self.heads
        sqrt_hd = math.sqrt(hd)
        ln1 = F.layer_norm(cur, (self.d,), self.ln1_w, self.ln1_b, 1e-5)
        q = ln1 @ self.Wq.T
        k = ln1 @ self.Wk.T
        v = ln1 @ self.Wv.T
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=cur.device), diagonal=1)
        context = torch.zeros_like(cur)
        for h in range(self.heads):
            lo = h * hd
            qh, kh, vh = q[:, lo:lo + hd], k[:, lo:lo + hd], v[:, lo:lo + hd]
            scores = (qh @ kh.T) / sqrt_hd
            scores = scores.masked_fill(causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            context[:, lo:lo + hd] = probs @ vh
        return context @ self.Wo.T, ln1

    def moe_mlp(self, x):
        # Top-2 mixture of experts over ln2 outputs.
        probs = torch.softmax(x @ self.router.weight.T, dim=-1)  # [S, N]
        topk = torch.topk(probs, self.topk, dim=-1)
        out = torch.zeros_like(x)
        S = x.shape[0]
        for e in range(self.n_experts):
            sel = (topk.indices == e)
            if sel.any():
                w = (topk.values * sel.to(topk.values.dtype)).sum(dim=1)  # [S]
                a = F.gelu(x @ self.experts[e].W1.T)
                out = out + w[:, None] * (a @ self.experts[e].W2.T)
        # Load-balancing aux: n_experts * sum(f_i * P_i)
        f = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            f[e] = (topk.indices == e).sum().float() / (S * self.topk)
        P = probs.mean(dim=0)
        self._aux = self.n_experts * torch.dot(f.detach(), P)
        return out


class BrainMoE(torch.nn.Module):
    def __init__(self, d=D, L=L, t=T, v=V, heads=2, n_experts=N_EXPERTS, topk=TOPK, seed=SEED):
        super().__init__()
        self.d, self.L, self.t, self.v, self.heads = d, L, t, v, heads
        self.n_experts = n_experts
        scale = 1.0 / math.sqrt(d)
        self.tok_emb = torch.nn.Parameter(torch.empty(v, d)); torch.nn.init.normal_(self.tok_emb, 0.0, scale)
        self.pos_emb = torch.nn.Parameter(torch.empty(t, d)); torch.nn.init.normal_(self.pos_emb, 0.0, 0.02)
        self.blocks = torch.nn.ModuleList(
            MoEBlock(d, n_experts=n_experts, topk=topk, seed=seed + i) for i in range(L))
        self.ln_f_w = torch.nn.Parameter(torch.ones(d))
        self.ln_f_b = torch.nn.Parameter(torch.zeros(d))

    def forward(self, ids):
        S = len(ids)
        idx = torch.tensor(ids, dtype=torch.long, device=self.tok_emb.device)
        cur = self.tok_emb[idx].clone()
        ar = torch.arange(S, device=cur.device)
        m = ar < self.t
        cur[m] = cur[m] + self.pos_emb[ar[m]]
        for blk in self.blocks:
            attn_out, _ln1 = blk._attention(cur)
            h1 = cur + attn_out
            ln2 = F.layer_norm(h1, (self.d,), blk.ln2_w, blk.ln2_b, 1e-5)
            cur = h1 + blk.moe_mlp(ln2)
        lf = F.layer_norm(cur, (self.d,), self.ln_f_w, self.ln_f_b, 1e-5)
        return lf @ self.tok_emb.T

    def train_loss(self, windows, aux_coeff=AUX_COEFF):
        self.train()
        losses = []
        count = 0
        for (x, y) in windows:
            logits = self.forward(x)
            nll = F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum")
            losses.append(nll)
            count += len(x)
        aux = sum(blk._aux for blk in self.blocks)
        return torch.stack(losses).sum() / max(1, count) + aux_coeff * aux

    @torch.no_grad()
    def loss(self, windows):
        self.eval()
        total = 0.0
        count = 0
        for (x, y) in windows:
            logits = self.forward(x)
            total += F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum").item()
            count += len(x)
        return total / max(1, count)

    @torch.no_grad()
    def eval_perplexity(self, ids, seq_len=8, max_positions=256, seed=0):
        self.eval()
        rng = random.Random(seed)
        n = len(ids) - seq_len - 1
        if n <= 0:
            return float("inf")
        pos = list(range(0, n + 1))
        if len(pos) > max_positions:
            pos = rng.sample(pos, max_positions)
        nll = 0.0
        count = 0
        for p in sorted(pos):
            x = ids[p:p + seq_len]
            y = ids[p + 1:p + seq_len + 1]
            logits = self.forward(x)
            nll += F.cross_entropy(logits, torch.tensor(y, device=logits.device), reduction="sum").item()
            count += len(x)
        return math.exp(nll / max(1, count))

    @torch.no_grad()
    def generate(self, prompt_ids, n=32, greedy=True, temperature=1.0, rng=None):
        ids = list(prompt_ids)
        out = []
        rng = rng or random.Random(0)
        self.eval()
        for _ in range(n):
            logits = self.forward(ids)
            last = logits[-1]
            if greedy:
                nxt = int(torch.argmax(last).item())
            else:
                probs = torch.softmax(last / max(temperature, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            out.append(nxt)
            ids.append(nxt)
            if nxt == EOS:
                break
        return out


def params_dict(model):
    tl = lambda t: t.detach().cpu().tolist()
    d = {
        "tok_emb": tl(model.tok_emb),
        "pos_emb": tl(model.pos_emb),
        "ln_f_w": tl(model.ln_f_w),
        "ln_f_b": tl(model.ln_f_b),
        "meta": {"n_experts": model.n_experts, "topk": model.blocks[0].topk,
                 "d": model.d, "L": model.L, "v": model.v, "t": model.t},
        "blocks": [],
    }
    for blk in model.blocks:
        d["blocks"].append({
            "Wq": tl(blk.Wq), "Wk": tl(blk.Wk), "Wv": tl(blk.Wv), "Wo": tl(blk.Wo),
            "ln1_w": tl(blk.ln1_w), "ln1_b": tl(blk.ln1_b),
            "ln2_w": tl(blk.ln2_w), "ln2_b": tl(blk.ln2_b),
            "router": tl(blk.router.weight),
            "experts": [{"W1": tl(e.W1), "W2": tl(e.W2)} for e in blk.experts],
        })
    return d


def load_weights(model, path):
    with torch.no_grad():
        d = json.load(open(path))
        model.tok_emb.copy_(torch.tensor(d["tok_emb"], dtype=torch.float32))
        model.pos_emb.copy_(torch.tensor(d["pos_emb"], dtype=torch.float32))
        model.ln_f_w.copy_(torch.tensor(d["ln_f_w"], dtype=torch.float32))
        model.ln_f_b.copy_(torch.tensor(d["ln_f_b"], dtype=torch.float32))
        for blk, b in zip(model.blocks, d["blocks"]):
            blk.Wq.copy_(torch.tensor(b["Wq"], dtype=torch.float32))
            blk.Wk.copy_(torch.tensor(b["Wk"], dtype=torch.float32))
            blk.Wv.copy_(torch.tensor(b["Wv"], dtype=torch.float32))
            blk.Wo.copy_(torch.tensor(b["Wo"], dtype=torch.float32))
            blk.ln1_w.copy_(torch.tensor(b["ln1_w"], dtype=torch.float32))
            blk.ln1_b.copy_(torch.tensor(b["ln1_b"], dtype=torch.float32))
            blk.ln2_w.copy_(torch.tensor(b["ln2_w"], dtype=torch.float32))
            blk.ln2_b.copy_(torch.tensor(b["ln2_b"], dtype=torch.float32))
            blk.router.weight.copy_(torch.tensor(b["router"], dtype=torch.float32))
            for e, be in zip(blk.experts, b["experts"]):
                e.W1.copy_(torch.tensor(be["W1"], dtype=torch.float32))
                e.W2.copy_(torch.tensor(be["W2"], dtype=torch.float32))
    return model


def save_weights(model, path):
    with open(path, "w") as f:
        json.dump(params_dict(model), f)


def train(model, train_ids, val_ids, steps, batch=2, seq_len=8, lr=1e-3,
          log_every=5, eval_every=20, aux_coeff=AUX_COEFF):
    rng = random.Random(999)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        windows = []
        for _ in range(batch):
            p = rng.randrange(0, len(train_ids) - seq_len - 2)
            windows.append((train_ids[p:p + seq_len], train_ids[p + 1:p + seq_len + 1]))
        opt.zero_grad()
        loss = model.train_loss(windows, aux_coeff)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"[train] step {step}/{steps} loss={loss.item():.4f} elapsed={time.time() - t0:.0f}s", flush=True)
        if step % eval_every == 0 or step == steps:
            print(f"[eval] step {step} val PPL={model.eval_perplexity(val_ids, seq_len=seq_len, max_positions=128):.2f}", flush=True)
    return model


def combined_corpus():
    text = build_corpus()
    for name in ("corpus_debug.txt", "corpus_conversation.txt", "corpus_tools.txt",
                 "corpus_logic.txt", "corpus_lang.txt"):
        p = Path(name)
        if p.exists():
            text += "\n" + p.read_text(errors="replace")
    return text


def main():
    ap = argparse.ArgumentParser(description="TinyCoder Brain AI model (MoE)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--weights", default="coder_1m_moe_weights.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--prompt", default="def fizzbuzz(n):")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainMoE().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BrainMoE: {n_params:,} params, n_experts={N_EXPERTS}, topk={TOPK}, device={args.device}", flush=True)

    if args.resume and os.path.exists(args.weights):
        load_weights(model, args.weights)
        print(f"resumed weights <- {args.weights}", flush=True)

    if args.generate:
        ids = [BOS] + list(args.prompt.encode("utf-8", "replace"))
        gen = model.generate(ids, n=args.n, greedy=True)
        print(args.prompt + bytes(t for t in gen if t < 256).decode("utf-8", "replace"), flush=True)
        return

    if args.train:
        corpus = combined_corpus()
        train_ids, val_ids = split_ids(corpus)
        print(f"training on combined corpus: {len(corpus)} chars -> steps={args.steps}", flush=True)
        train(model, train_ids, val_ids, steps=args.steps, batch=args.batch,
              seq_len=args.seq_len, lr=args.lr)
        save_weights(model, args.weights)
        print(f"saved weights -> {args.weights}", flush=True)


if __name__ == "__main__":
    main()
