#!/usr/bin/env python3
"""Brain11 model — a compact decoder-only transformer for edge devices.

Architecture upgrades over the old byte-level model:
  * byte-level BPE (4096 tokens) instead of raw bytes
  * RoPE instead of learned positions
  * RMSNorm instead of LayerNorm
  * SwiGLU feed-forward instead of GELU MLP
  * tied input/output embeddings
  * causal flash attention (F.scaled_dot_product_attention)

Config: d=384, 5 layers, 6 heads, ffn=1088, ctx=320, vocab=4096
Params: 10,793,088 per model (32.4M for all three, under the 33M cap).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain11_tokenizer import BOS_ID, EOS_ID, VOCAB_SIZE, decode, encode


@dataclass
class Brain11Config:
    vocab_size: int = VOCAB_SIZE
    d: int = 384
    layers: int = 5
    heads: int = 6
    ffn: int = 1088
    ctx: int = 320
    dropout: float = 0.05

    @property
    def head_dim(self) -> int:
        return self.d // self.heads

    def n_params(self) -> int:
        per_layer = 4 * self.d * self.d + 3 * self.d * self.ffn + 2 * self.d
        return self.layers * per_layer + self.vocab_size * self.d + self.d


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def build_rope(ctx: int, head_dim: int, device=None):
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(ctx, device=device, dtype=torch.float32), inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, HD]; cos/sin: [S, HD]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]


class Attention(nn.Module):
    def __init__(self, cfg: Brain11Config):
        super().__init__()
        self.h = cfg.heads
        self.hd = cfg.head_dim
        self.drop = cfg.dropout
        self.q = nn.Linear(cfg.d, cfg.d, bias=False)
        self.k = nn.Linear(cfg.d, cfg.d, bias=False)
        self.v = nn.Linear(cfg.d, cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, S, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, S, self.h, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, S, self.h * self.hd)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, d: int, ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: Brain11Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d)
        self.mlp = SwiGLU(cfg.d, cfg.ffn)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class Brain(nn.Module):
    """Brain11 decoder (one domain per weights file)."""

    def __init__(self, cfg: Brain11Config | None = None):
        super().__init__()
        self.cfg = cfg or Brain11Config()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm_f = RMSNorm(c.d)
        cos, sin = build_rope(c.ctx, c.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # scaled init for residual output projections
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o.weight, 0.0, 0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(blk.mlp.w2.weight, 0.0, 0.02 / math.sqrt(2 * c.layers))
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, S = idx.shape
        if S > self.cfg.ctx:
            raise ValueError(f"sequence length {S} > ctx {self.cfg.ctx}")
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.cos[:S], self.sin[:S]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = x @ self.tok_emb.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], n: int = 96, temperature: float = 0.7,
                 top_k: int = 40, repetition_penalty: float = 1.05,
                 stop_ids: tuple[int, ...] = (EOS_ID,)) -> list[int]:
        """Autoregressive sampling. Returns only the newly generated ids."""
        self.eval()
        device = self.tok_emb.weight.device
        # keep at least one slot for generation; the prompt keeps its tail so
        # the current task (always last) is never truncated away
        ids = list(prompt_ids)[-(self.cfg.ctx - 1):]
        n = max(1, min(n, self.cfg.ctx - len(ids)))
        out: list[int] = []
        for _ in range(n):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = self.forward(x)
            logits = logits[0, -1].float()
            if repetition_penalty and repetition_penalty != 1.0:
                for tok in set(ids[-64:]):
                    v = logits[tok]
                    logits[tok] = v / repetition_penalty if v > 0 else v * repetition_penalty
            if temperature <= 0:
                nxt = int(torch.argmax(logits).item())
            else:
                logits = logits / temperature
                if top_k and top_k < logits.numel():
                    kth = torch.topk(logits, top_k).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            out.append(nxt)
            ids.append(nxt)
            if nxt in stop_ids:
                break
        return out

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        """Average log-prob of each choice after the prompt (higher is better)."""
        self.eval()
        device = self.tok_emb.weight.device
        p = encode(prompt, add_eos=False)
        scores = []
        for choice in choices:
            c = encode(choice, add_bos=False, add_eos=False)
            if not c:
                scores.append(float("-inf"))
                continue
            ids = (p + c)[-self.cfg.ctx:]
            offset = max(0, len(p) - self.cfg.ctx)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = self.forward(x)
            logits = logits[0]
            total = 0.0
            for j, tok in enumerate(c):
                pos = len(p) - offset + j - 1
                if pos < 0:
                    continue
                logp = F.log_softmax(logits[pos], dim=-1)
                total += float(logp[tok].item())
            scores.append(total / len(c))
        return scores


def save_checkpoint(path: str | Path, model: Brain, step: int, extra: dict | None = None):
    """Atomic checkpoint write. Never leaves a half-written file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "config": asdict(model.cfg),
        "step": step,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        import os
        os.fsync(f.fileno())
    import os
    os.replace(tmp, path)


def load_model(path: str | Path, device=None) -> Brain:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Brain11Config(**ckpt["config"])
    model = Brain(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def generate_text(model: Brain, prompt: str, n: int = 96, **kwargs) -> str:
    ids = encode(prompt, add_bos=True, add_eos=False)
    out = model.generate(ids, n=n, **kwargs)
    return decode(out)


if __name__ == "__main__":
    for name, cfg in [
        ("brain11", Brain11Config()),
        ("small", Brain11Config(d=384, layers=4, ffn=1024)),
    ]:
        print(name, f"{cfg.n_params():,} params")
