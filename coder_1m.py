#!/usr/bin/env python3
"""TinyCoder-1M — a ~1,000,000-parameter char-level coding LM with self-learning.

Pure Python (stdlib only). Architecture: 3-layer pre-LN GPT-style transformer
with 2 heads and a tied byte-level output head. Total parameters: 999,540
(≈ 1M). Training: hand-written Adam. Self-learning: bounded online adaptation
(SelfLearner) that can be invoked any time the model is used.

Commands:
    python3 coder_1m.py --selftest           # gradient check on a tiny config
    python3 coder_1m.py --train --steps 25  # train, save coder_1m_weights.json
    python3 coder_1m.py --eval               # perplexity + coding micro-bench, write report
    python3 coder_1m.py --selflearn-demo     # online-learning demo, append to report
    python3 coder_1m.py --generate --prompt "def fizzbuzz(n):" --n 40
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import deque
from operator import mul
from pathlib import Path

# ------------------------------------------------------------------ constants
V = 260                    # 256 byte tokens + BOS/EOS/PAD/UNK
BOS, EOS, PAD, UNK = 256, 257, 258, 259
D = 162                    # model dim
L = 3                      # transformer blocks
HEADS = 2
HD = D // HEADS            # 81
FFN = 4 * D                # 648
T = 64                     # context length (positions beyond T use zero pos-emb)
SEED = 20261129


def total_params(d=D, L=L, v=V, t=T):
    """Exact parameter count of the architecture."""
    # per block: 4 d^2 (q,k,v,o) + 2 * d*ffn (W1, W2) + 4d (2 LayerNorms)
    return L * (4 * d * d + 2 * d * (4 * d) + 4 * d) + 2 * d + v * d + t * d


# ------------------------------------------------------------------ utilities
def zeros_like(p):
    if isinstance(p, dict):
        return {k: zeros_like(v) for k, v in p.items()}
    if isinstance(p, list) and p and isinstance(p[0], list):
        return [[0.0] * len(row) for row in p]
    if isinstance(p, list) and p and isinstance(p[0], dict):
        return [zeros_like(x) for x in p]
    return [0.0] * len(p)


def adam_update(p, g, m, v, lr, t, b1, b2, eps):
    if isinstance(p, dict):
        for k in p:
            adam_update(p[k], g[k], m[k], v[k], lr, t, b1, b2, eps)
    elif isinstance(p, list):
        if p and isinstance(p[0], list):
            for r in range(len(p)):
                adam_update(p[r], g[r], m[r], v[r], lr, t, b1, b2, eps)
        elif p and isinstance(p[0], dict):
            for i in range(len(p)):
                adam_update(p[i], g[i], m[i], v[i], lr, t, b1, b2, eps)
        else:
            for i in range(len(p)):
                mv = b1 * m[i] + (1.0 - b1) * g[i]
                vv = b2 * v[i] + (1.0 - b2) * g[i] * g[i]
                m[i] = mv
                v[i] = vv
                p[i] = p[i] - lr * (mv / (1.0 - b1 ** t)) / (
                    math.sqrt(vv / (1.0 - b2 ** t)) + eps)


class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.model = model
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = zeros_like(model.params)
        self.v = zeros_like(model.params)
        self.t = 0

    def step(self, params, grads, lr=None):
        self.t += 1
        adam_update(params, grads, self.m, self.v,
                    lr if lr is not None else self.lr, self.t,
                    self.b1, self.b2, self.eps)


def dot(a, b):
    return sum(map(mul, a, b))


def matvec(W, x):
    return [sum(map(mul, row, x)) for row in W]


def ln_forward(x, w, b):
    n = len(x)
    mu = sum(x) / n
    var = sum((xi - mu) ** 2 for xi in x) / n
    rstd = 1.0 / math.sqrt(var + 1e-5)
    xhat = [(xi - mu) * rstd for xi in x]
    out = [w[i] * xhat[i] + b[i] for i in range(n)]
    return out, xhat, rstd, mu


def ln_bwd(dout, xhat, rstd, w, gw, gb):
    """Backward of y = xhat*w + b. Returns dx; accumulates gw, gb."""
    n = len(dout)
    dxhat = [dout[i] * w[i] for i in range(n)]
    m1 = sum(dxhat) / n
    m2 = sum(dxhat[i] * xhat[i] for i in range(n)) / n
    for i in range(n):
        gw[i] += dout[i] * xhat[i]
        gb[i] += dout[i]
    return [rstd * (dxhat[i] - m1 - xhat[i] * m2) for i in range(n)]


def gelu(z):
    return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))


def gelu_prime(z):
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) + z * phi


# ------------------------------------------------------------------ the model
class Model:
    def __init__(self, d=D, L=L, t=T, v=V, heads=HEADS, seed=SEED, lr=1e-3):
        self.d, self.L, self.t, self.v, self.heads = d, L, t, v, heads
        self.hd = d // heads
        self.ffn = 4 * d
        self.sqrt_hd = math.sqrt(self.hd)
        rng = random.Random(seed)

        def mat(r, c, scale):
            return [[rng.gauss(0.0, scale) for _ in range(c)] for _ in range(r)]

        scale = 1.0 / math.sqrt(d)
        self.params = {
            'tok_emb': mat(v, d, scale),
            'pos_emb': mat(t, d, 0.02),
            'blocks': [{
                'Wq': mat(d, d, scale), 'Wk': mat(d, d, scale),
                'Wv': mat(d, d, scale), 'Wo': mat(d, d, scale),
                'W1': mat(self.ffn, d, scale), 'W2': mat(d, self.ffn, scale),
                'ln1_w': [1.0] * d, 'ln1_b': [0.0] * d,
                'ln2_w': [1.0] * d, 'ln2_b': [0.0] * d,
            } for _ in range(L)],
            'ln_f_w': [1.0] * d, 'ln_f_b': [0.0] * d,
        }
        self.opt = Adam(self, lr=lr)

    # -- forward ------------------------------------------------------------
    def forward(self, ids):
        d, S = self.d, len(ids)
        P = self.params
        tok, pos = P['tok_emb'], P['pos_emb']
        cache = {'S': S, 'layers': [], 'ln_f': [], 'ln_f_xhat': [], 'ln_f_rstd': []}
        cur = []
        for s, t in enumerate(ids):
            row = tok[t]
            if s < self.t:
                cur.append([a + b for a, b in zip(row, pos[s])])
            else:
                cur.append(list(row))
        for li in range(self.L):
            blk = P['blocks'][li]
            lc = {'ln1': [], 'ln1_xhat': [], 'ln1_rstd': [], 'q': [], 'k': [],
                  'v': [], 'probs': [], 'context': [], 'ln2': [], 'ln2_xhat': [],
                  'ln2_rstd': [], 'z': [], 'a': []}
            new_cur = []
            for s in range(S):
                x = cur[s]
                ln1, xhat, rstd, _ = ln_forward(x, blk['ln1_w'], blk['ln1_b'])
                lc['ln1'].append(ln1); lc['ln1_xhat'].append(xhat); lc['ln1_rstd'].append(rstd)
                q = matvec(blk['Wq'], ln1)
                k = matvec(blk['Wk'], ln1)
                vv = matvec(blk['Wv'], ln1)
                lc['q'].append(q); lc['k'].append(k); lc['v'].append(vv)
                context = [0.0] * d
                probs_head = []
                for h in range(self.heads):
                    lo = h * self.hd
                    scores = [dot(q[lo:lo + self.hd], lc['k'][j][lo:lo + self.hd]) / self.sqrt_hd
                              for j in range(s + 1)]
                    mx = max(scores)
                    exps = [math.exp(sc - mx) for sc in scores]
                    Z = sum(exps)
                    probs = [e / Z for e in exps]
                    probs_head.append(probs)
                    for a in range(self.hd):
                        acc = 0.0
                        for j in range(s + 1):
                            acc += probs[j] * lc['v'][j][lo + a]
                        context[lo + a] = acc
                lc['probs'].append(probs_head)
                lc['context'].append(context)
                attn_out = matvec(blk['Wo'], context)
                h1 = [x[i] + attn_out[i] for i in range(d)]
                ln2, x2hat, rstd2, _ = ln_forward(h1, blk['ln2_w'], blk['ln2_b'])
                lc['ln2'].append(ln2); lc['ln2_xhat'].append(x2hat); lc['ln2_rstd'].append(rstd2)
                z = matvec(blk['W1'], ln2)
                a_g = [gelu(zv) for zv in z]
                lc['z'].append(z); lc['a'].append(a_g)
                b_out = matvec(blk['W2'], a_g)
                new_cur.append([h1[i] + b_out[i] for i in range(d)])
            cur = new_cur
            cache['layers'].append(lc)
        for s in range(S):
            lf, xhat, rstd, _ = ln_forward(cur[s], P['ln_f_w'], P['ln_f_b'])
            cache['ln_f'].append(lf)
            cache['ln_f_xhat'].append(xhat)
            cache['ln_f_rstd'].append(rstd)
        logits = [[dot(cache['ln_f'][s], tok[v]) for v in range(self.v)] for s in range(S)]
        return logits, cache

    # -- training ------------------------------------------------------------
    def loss_and_grads(self, windows):
        grads = zeros_like(self.params)
        total = 0.0
        count = 0
        for (x, y) in windows:
            S = len(x)
            logits, cache = self.forward(x)
            p_all = []
            nll = 0.0
            for s in range(S):
                lg = logits[s]
                mx = max(lg)
                exps = [math.exp(v - mx) for v in lg]
                Z = sum(exps)
                p = [e / Z for e in exps]
                p_all.append(p)
                nll -= math.log(p[y[s]] if p[y[s]] > 1e-300 else 1e-300)
            total += nll
            count += S
            dlogits = [[(p_all[s][v] - (1.0 if v == y[s] else 0.0)) / S
                        for v in range(self.v)] for s in range(S)]
            d_cur = self.backward(cache, dlogits, grads)
            for s in range(S):
                t = x[s]
                row = grads['tok_emb'][t]
                for a in range(self.d):
                    row[a] += d_cur[s][a]
                if s < self.t:
                    prow = grads['pos_emb'][s]
                    for a in range(self.d):
                        prow[a] += d_cur[s][a]
        return total / count, grads

    def backward(self, cache, dlogits, grads):
        P = self.params
        S = cache['S']
        d = self.d
        # head: logits[s][v] = dot(ln_f[s], tok_emb[v])
        d_ln_f = [[0.0] * d for _ in range(S)]
        for s in range(S):
            lf = cache['ln_f'][s]
            for v in range(self.v):
                g = dlogits[s][v]
                if g:
                    te = P['tok_emb'][v]
                    gt = grads['tok_emb'][v]
                    for a in range(d):
                        gt[a] += g * lf[a]
                        d_ln_f[s][a] += g * te[a]
        # final LN backward
        d_cur = []
        for s in range(S):
            dx = ln_bwd(d_ln_f[s], cache['ln_f_xhat'][s], cache['ln_f_rstd'][s],
                        P['ln_f_w'], grads['ln_f_w'], grads['ln_f_b'])
            d_cur.append(dx)
        for li in reversed(range(self.L)):
            blk = P['blocks'][li]
            gblk = grads['blocks'][li]
            lc = cache['layers'][li]
            d_cur = self.layer_backward(blk, gblk, lc, d_cur, S)
        return d_cur

    def layer_backward(self, blk, gblk, lc, d_cur, S):
        d, ffn, hd = self.d, self.ffn, self.hd
        sqrt_hd = self.sqrt_hd
        accum_dk = [[0.0] * d for _ in range(S)]
        d_h1_all = [[0.0] * d for _ in range(S)]
        d_ln1_all = [[0.0] * d for _ in range(S)]
        for s in range(S):
            d_h1 = d_cur[s]
            # ---- MLP backward: x_new = h1 + b_out ----
            W2 = blk['W2']
            gW2 = gblk['W2']
            d_b_out = d_h1
            for j in range(ffn):
                aj = lc['a'][s][j]
                if aj:
                    for i in range(d):
                        gW2[i][j] += aj * d_b_out[i]
            d_a_g = [0.0] * ffn
            for i in range(d):
                Wi = W2[i]
                for j in range(ffn):
                    if Wi[j]:
                        d_a_g[j] += Wi[j] * d_b_out[i]
            z = lc['z'][s]
            dz = [d_a_g[j] * gelu_prime(z[j]) for j in range(ffn)]
            ln2 = lc['ln2'][s]
            d_ln2 = [0.0] * d
            W1 = blk['W1']
            gW1 = gblk['W1']
            for j in range(ffn):
                dzj = dz[j]
                if dzj:
                    W1j = W1[j]
                    gW1j = gW1[j]
                    for i in range(d):
                        gW1j[i] += ln2[i] * dzj
                        d_ln2[i] += dzj * W1j[i]
            dx2 = ln_bwd(d_ln2, lc['ln2_xhat'][s], lc['ln2_rstd'][s],
                         blk['ln2_w'], gblk['ln2_w'], gblk['ln2_b'])
            d_h1 = [d_h1[i] + dx2[i] for i in range(d)]
            d_h1_all[s] = d_h1
            # ---- attention backward: h1 = x + attn_out ----
            d_attn = d_h1
            context = lc['context'][s]
            Wo = blk['Wo']
            gWo = gblk['Wo']
            d_ctx = [0.0] * d
            for i in range(d):
                Wi = Wo[i]
                di = d_attn[i]
                gWi = gWo[i]
                for a in range(d):
                    d_ctx[a] += di * Wi[a]
                    gWi[a] += context[a] * di
            ln1 = lc['ln1'][s]
            q = lc['q'][s]
            k = lc['k'][s]
            vv = lc['v'][s]
            Wq = blk['Wq']
            Wk = blk['Wk']
            Wv = blk['Wv']
            gWq = gblk['Wq']
            gWv = gblk['Wv']
            for h in range(self.heads):
                lo = h * hd
                probs = lc['probs'][s][h]
                # dv path: dv[j][d] = probs[j]*d_ctx[d]; v[j][d] = dot(Wv[d], ln1_j)
                for j in range(s + 1):
                    pj = probs[j]
                    if pj:
                        ln1_j = lc['ln1'][j]
                        for a in range(hd):
                            dv = pj * d_ctx[lo + a]
                            if dv:
                                gWv_row = gWv[lo + a]
                                Wv_row = Wv[lo + a]
                                for i in range(d):
                                    gWv_row[i] += ln1_j[i] * dv
                                    d_ln1_all[j][i] += dv * Wv_row[i]
                # dscores: dsc[j] = dot(d_ctx[lo:lo+hd], v[j][lo:lo+hd])
                dsc_all = [dot(d_ctx[lo:lo + hd], lc['v'][j][lo:lo + hd]) for j in range(s + 1)]
                dot_sum = sum(p * dd for p, dd in zip(probs, dsc_all))
                dq_acc = [0.0] * d
                for j in range(s + 1):
                    dzj = probs[j] * (dsc_all[j] - dot_sum)
                    if dzj:
                        kj = lc['k'][j]
                        for a in range(hd):
                            dq_acc[lo + a] += dzj * kj[lo + a] / sqrt_hd
                            accum_dk[j][lo + a] += dzj * q[lo + a] / sqrt_hd
                for a in range(hd):
                    dq = dq_acc[lo + a]
                    if dq:
                        gWq_row = gWq[lo + a]
                        Wq_row = Wq[lo + a]
                        for i in range(d):
                            gWq_row[i] += ln1[i] * dq
                            d_ln1_all[s][i] += dq * Wq_row[i]
        # dk contributions: k[j] = matvec(Wk, ln1_j)
        for j in range(S):
            ln1_j = lc['ln1'][j]
            for a in range(d):
                dk = accum_dk[j][a]
                if dk:
                    gWk_row = gblk['Wk'][a]
                    Wk_row = blk['Wk'][a]
                    for i in range(d):
                        gWk_row[i] += dk * ln1_j[i]
                        d_ln1_all[j][i] += dk * Wk_row[i]
        # LN1 backward
        d_new = []
        for s in range(S):
            dx1 = ln_bwd(d_ln1_all[s], lc['ln1_xhat'][s], lc['ln1_rstd'][s],
                         blk['ln1_w'], gblk['ln1_w'], gblk['ln1_b'])
            d_new.append([d_h1_all[s][i] + dx1[i] for i in range(d)])
        return d_new

    def train_step(self, windows, opt=None, lr=None):
        loss, grads = self.loss_and_grads(windows)
        opt = opt or self.opt
        opt.step(self.params, grads, lr)
        return loss

    def loss_on(self, windows):
        total = 0.0
        count = 0
        for (x, y) in windows:
            logits, _ = self.forward(x)
            for s in range(len(x)):
                lg = logits[s]
                mx = max(lg)
                exps = [math.exp(v - mx) for v in lg]
                Z = sum(exps)
                total -= math.log(exps[y[s]] / Z)
                count += 1
        return total / max(1, count)

    def eval_perplexity(self, ids, seq_len=8, max_positions=256, seed=0):
        rng = random.Random(seed)
        n = len(ids) - seq_len - 1
        if n <= 0:
            return float('inf')
        pos = list(range(0, n + 1))
        if len(pos) > max_positions:
            pos = rng.sample(pos, max_positions)
        nll = 0.0
        count = 0
        for p in sorted(pos):
            x = ids[p:p + seq_len]
            y = ids[p + 1:p + seq_len + 1]
            logits, _ = self.forward(x)
            for s in range(seq_len):
                lg = logits[s]
                mx = max(lg)
                exps = [math.exp(v - mx) for v in lg]
                Z = sum(exps)
                nll -= math.log(exps[y[s]] / Z)
                count += 1
        return math.exp(nll / count)

    def generate(self, prompt_ids, n=32, greedy=True, temperature=1.0, rng=None):
        ids = list(prompt_ids)
        out = []
        rng = rng or random.Random(0)
        for _ in range(n):
            logits, _ = self.forward(ids)
            last = logits[-1]
            if greedy:
                nxt = max(range(self.v), key=lambda i: last[i])
            else:
                mx = max(last)
                exps = [math.exp((x - mx) / temperature) for x in last]
                Z = sum(exps)
                r = rng.random() * Z
                acc = 0.0
                nxt = self.v - 1
                for i, e in enumerate(exps):
                    acc += e
                    if r <= acc:
                        nxt = i
                        break
            out.append(nxt)
            ids.append(nxt)
            if nxt == EOS:
                break
        return out


# ------------------------------------------------------------------ grad check
def check_gradients(d=8, L=3, t=16, v=16, seed=1, n_checks=14):
    model = Model(d=d, L=L, t=t, v=v, seed=seed)
    rng = random.Random(seed + 1)
    S = 5
    x = [rng.randrange(v) for _ in range(S)]
    y = [rng.randrange(v) for _ in range(S)]
    windows = [(x, y)]
    loss, grads = model.loss_and_grads(windows)
    print(f"grad check: d={d} L={L} t={t} v={v} loss={loss:.6f}")

    def getp(path):
        o = model.params
        for p in path:
            o = o[p]
        return o

    def setp(path, val):
        o = model.params
        for p in path[:-1]:
            o = o[p]
        o[path[-1]] = val

    paths = [
        ('tok_emb', 0, 3), ('tok_emb', v - 1, 5),
        ('blocks', 0, 'Wq', 1, 2), ('blocks', 0, 'Wk', 3, 1),
        ('blocks', 0, 'Wv', 0, 4), ('blocks', 0, 'Wo', 2, 5),
        ('blocks', 0, 'W1', 1, 0), ('blocks', 0, 'W2', 0, 3),
        ('blocks', 0, 'ln1_w', 4), ('blocks', 0, 'ln2_b', 6),
        ('blocks', 2, 'Wq', 5, 2), ('blocks', 2, 'W2', 1, 7),
        ('ln_f_w', 7), ('ln_f_b', 5), ('pos_emb', 3, 2),
    ]
    worst, worst_path = 0.0, None
    for path in paths:
        orig = getp(path)
        g = grads
        for p in path:
            g = g[p]
        eps = 1e-4
        setp(path, orig + eps)
        lp = model.loss_and_grads(windows)[0]
        setp(path, orig - eps)
        lm = model.loss_and_grads(windows)[0]
        setp(path, orig)
        num = (lp - lm) / (2 * eps)
        rel = abs(g - num) / (abs(g) + abs(num) + 1e-8)
        if rel > worst:
            worst, worst_path = rel, path
        print(f"  {path}: analytic={g:+.6e} numeric={num:+.6e} relerr={rel:.2e}")
    ok = worst < 1e-3
    print("GRAD CHECK", "PASS" if ok else "FAIL", f"(worst relerr {worst:.2e} at {worst_path})")
    return ok


# ------------------------------------------------------------------ corpus
VARS = ["x", "y", "n", "i", "items", "data", "result", "buf", "out", "total",
        "key", "value", "name", "count", "score"]
FUNCS = ["fetch", "parse", "compute", "render", "count", "summarize", "validate",
         "convert", "build", "emit", "scan", "score", "sort", "index", "map"]
ATTRS = ["length", "size", "value", "name", "total", "count", "score", "items"]


def gen_code_corpus(n_chars, rng):
    lines = []
    while sum(map(len, lines)) < n_chars:
        k = rng.randrange(10)
        if k == 0:
            fn = rng.choice(FUNCS) + rng.choice(["_v1", "_2", "", "_b"]) + str(rng.randrange(1, 99))
            args = ", ".join(rng.choice(VARS) for _ in range(rng.randrange(0, 4)))
            lines.append(f"def {fn}({args}):\n")
            lines.append(rng.choice([
                '    """Compute ' + fn + ' over the inputs."""\n',
                "    return " + rng.choice(["None", "0", "True", "[]", "total",
                                            f"{rng.choice(VARS)}.{rng.choice(ATTRS)}",
                                            f"sum({rng.choice(VARS)})", f"len({rng.choice(VARS)})"]) + "\n",
                "    " + rng.choice(VARS) + " = " + rng.choice(
                    [f"{rng.choice(VARS)}.{rng.choice(ATTRS)}",
                     f"[i for i in range({rng.choice(VARS)})]",
                     f"{rng.choice(VARS)} + {rng.choice(VARS)}"]) + "\n",
                "    if " + rng.choice(VARS) + ":\n        return " + rng.choice(VARS) + "\n",
                "    print(f\"{rng.choice(VARS)} = {rng.choice(VARS)}\")\n",
                "    for " + rng.choice(VARS) + " in " + rng.choice(VARS) + ":\n        " + rng.choice(["pass", "continue", "break"]) + "\n",
            ]))
        elif k == 1:
            lines.append(f"class {rng.choice(VARS).title()}:\n    \"\"\"A small helper type.\"\"\"\n")
        elif k == 2:
            lines.append(f"{rng.choice(VARS)} = {rng.choice(['[]', '{}', '0', 'None', 'True', f'\"{rng.choice(ATTRS)}\"'])}\n")
        elif k == 3:
            lines.append(f"{rng.choice(VARS)} = {rng.choice(VARS)} + {rng.choice(VARS)}\n")
        elif k == 4:
            lines.append('if __name__ == "__main__":\n    ' + rng.choice(["main()", "run()", "demo()"]) + "\n")
        elif k == 5:
            lines.append('def main():\n    """Entry point for the tool."""\n')
        elif k == 6:
            lines.append(f"    {rng.choice(VARS)} = {rng.choice(VARS)}[:{rng.randrange(1, 9)}]\n")
        elif k == 7:
            lines.append(f"    try:\n        {rng.choice(VARS)} = int({rng.choice(VARS)})\n"
                         f"    except ValueError:\n        {rng.choice(VARS)} = 0\n")
        elif k == 8:
            lines.append(f"    {rng.choice(VARS)} = \"\"\"triple quoted text\nwith a few lines\"\"\"\n")
        else:
            lines.append("\n")
    return "".join(lines)


def repo_code_text(max_chars=60_000):
    texts = []
    for p in sorted(glob.glob("*.py")):
        try:
            t = Path(p).read_text(errors="replace")
        except OSError:
            continue
        texts.append(t[:max_chars])
    return "\n".join(texts)


def build_corpus(n_synth=240_000):
    rng = random.Random(12345)
    return repo_code_text() + "\n\n" + gen_code_corpus(n_synth, rng)


def split_ids(text, val_frac=0.10):
    ids = [BOS] + list(text.encode('utf-8', 'replace')) + [EOS]
    n_val = max(256, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]


# ------------------------------------------------------------------ train
def train(model, train_ids, val_ids, steps, batch=2, seq_len=8, lr=1e-3,
          log_every=5, eval_every=20, eval_positions=128):
    rng = random.Random(999)
    t0 = time.time()
    for step in range(1, steps + 1):
        windows = []
        for _ in range(batch):
            p = rng.randrange(0, len(train_ids) - seq_len - 2)
            windows.append((train_ids[p:p + seq_len], train_ids[p + 1:p + seq_len + 1]))
        loss = model.train_step(windows, lr=lr)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"[train] step {step}/{steps} loss={loss:.4f} elapsed={time.time() - t0:.0f}s")
        if step % eval_every == 0 or step == steps:
            print(f"[eval] step {step} val PPL={model.eval_perplexity(val_ids, seq_len=seq_len, max_positions=eval_positions):.2f}")
    return model


# ------------------------------------------------------------------ weights
def save_weights(model, path):
    with open(path, "w") as f:
        json.dump(model.params, f)


def load_weights(model, path):
    with open(path) as f:
        model.params = json.load(f)


# ------------------------------------------------------------------ eval
def coding_microbench(model, val_text, n=12, snippets=3):
    results = []
    if len(val_text) < n + 20:
        return results
    step = max(1, len(val_text) // snippets)
    for i in range(0, len(val_text) - n - 20, step):
        text = val_text[i:i + 40]
        cut = len(text) - n
        prefix, target = text[:cut], text[cut:]
        ids = [BOS] + list(prefix.encode('utf-8', 'replace'))
        gen = model.generate(ids, n=n, greedy=True)
        tgt = list(target.encode('utf-8', 'replace'))
        exact = 1 if gen[:len(tgt)] == tgt else 0
        acc = sum(1 for a, b in zip(gen, tgt) if a == b) / max(1, len(tgt))
        results.append((prefix[-25:], exact, acc, target, bytes(t for t in gen if t < 256).decode('utf-8', 'replace')))
    return results


def write_report(path, model, args, train_ppl, val_ppl, micro, samples):
    lines = [
        "# TinyCoder-1M report",
        "",
        f"- parameters: {total_params(model.d, model.L, model.v, model.t):,} (target ~1,000,000)",
        f"- dim={model.d} layers={model.L} heads={model.heads} head_dim={model.hd} ffn={model.ffn} ctx={model.t}",
        f"- vocab={model.v} (256 byte tokens + BOS/EOS/PAD/UNK)",
        f"- steps={args.steps} batch={args.batch} seq_len={args.seq_len} lr={args.lr} seed={args.seed}",
        f"- training PPL (subset): {train_ppl:.2f}   held-out val PPL: {val_ppl:.2f}",
        "",
        "## Coding micro-bench (greedy generation vs held-out text)",
        f"- exact-match snippets: {sum(1 for r in micro if r[1])}/{len(micro)}",
        f"- avg char accuracy of generated continuations: {sum(r[2] for r in micro) / max(1, len(micro)):.3f}",
        "",
        "## Samples (greedy from prompt)",
    ]
    for prompt, gen in samples:
        lines.append(f"- `{prompt}` -> `{gen[:80]}...`")
    lines += [
        "",
        "## Honest notes",
        "- This is a ~1M-parameter byte-level coding LM: a 3-layer pre-LN transformer (d=162, 2 heads, FFN=648), trained with hand-written Adam.",
        "- Small PPL deltas are expected at this size; real coding ability needs more data, steps, and ideally a bigger model.",
        "- The self-learning demo (`--selflearn-demo`) measures PPL deltas before/after bounded online steps (see `SelfLearner`).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote report -> {path}")


def append_report(path, lines):
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


# ------------------------------------------------------------------ self-learning
class SelfLearner:
    """Bounded online adaptation: remembers interactions and learns from them,
    with replay of earlier examples (sleep replay) to limit drift."""

    def __init__(self, model, replay_cap=256, lr=5e-4, seq_len=8):
        self.model = model
        self.replay = deque(maxlen=replay_cap)
        self.opt = Adam(model, lr=lr)
        self.lr = lr
        self.seq_len = seq_len
        self.steps_taken = 0
        self.rng = random.Random(7)

    def _windows(self, ids, seq_len=None):
        seq_len = seq_len or self.seq_len
        windows = []
        if len(ids) < seq_len + 2:
            return windows
        for _ in range(4):
            p = self.rng.randrange(0, len(ids) - seq_len - 1)
            windows.append((ids[p:p + seq_len], ids[p + 1:p + seq_len + 1]))
        return windows

    def remember(self, ids):
        self.replay.extend(self._windows(ids))

    def learn(self, ids, steps=3):
        windows = self._windows(ids)
        if not windows:
            return 0.0
        losses = []
        for _ in range(steps):
            batch = list(windows)
            if self.replay:
                batch += [self.replay[self.rng.randrange(len(self.replay))] for _ in range(6)]
            losses.append(self.model.train_step(batch, opt=self.opt, lr=self.lr))
            self.steps_taken += 1
        return sum(losses) / len(losses)

    def sleep(self, epochs=2):
        if not self.replay:
            return 0.0
        total = 0.0
        n = 0
        for _ in range(epochs):
            for _ in range(min(4, len(self.replay))):
                w = [self.replay[self.rng.randrange(len(self.replay))] for _ in range(3)]
                total += self.model.train_step(w, opt=self.opt, lr=self.lr)
                n += 1
                self.steps_taken += 1
        return total / max(1, n)


def selflearn_demo(model, val_ids, report_path):
    learner = SelfLearner(model, lr=5e-4)
    probe = val_ids[:64]
    learner.remember(probe)
    before = model.eval_perplexity(probe, seq_len=8, max_positions=24)
    learner.learn(probe, steps=3)
    after = model.eval_perplexity(probe, seq_len=8, max_positions=24)
    learner.sleep(epochs=1)
    after_sleep = model.eval_perplexity(probe, seq_len=8, max_positions=24)
    lines = [
        "## Self-learning demo",
        f"- probe perplexity before learning: {before:.3f}",
        f"- probe perplexity after 3 Adam steps on the probe example: {after:.3f} (delta {after - before:+.3f})",
        f"- probe perplexity after 1 replay epoch (sleep): {after_sleep:.3f}",
        f"- total self-learning Adam steps taken: {learner.steps_taken}",
        "- honest note: at this model size the measured delta is small; real learning needs more data and steps.",
    ]
    append_report(report_path, lines)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="TinyCoder-1M: a ~1M-parameter coding LM with self-learning")
    ap.add_argument('--selftest', action='store_true', help='gradient check on a tiny config')
    ap.add_argument('--train', action='store_true', help='train and save weights')
    ap.add_argument('--eval', action='store_true', help='perplexity + coding micro-bench, write report')
    ap.add_argument('--selflearn-demo', action='store_true', help='online-learning demo, append to report')
    ap.add_argument('--generate', action='store_true', help='greedy generation from a prompt')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seq-len', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weights', default='coder_1m_weights.json')
    ap.add_argument('--report', default='coder_1m_report.txt')
    ap.add_argument('--prompt', default='def main():')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--resume', action='store_true', help='load existing weights before training')
    args = ap.parse_args()

    if args.selftest:
        ok = check_gradients()
        sys.exit(0 if ok else 1)

    model = Model(seed=args.seed)
    corpus = None
    train_ids = val_ids = val_text = None
    if args.train or args.eval or args.selflearn_demo or args.generate:
        corpus = build_corpus()
        train_ids, val_ids = split_ids(corpus)
        val_text = bytes(t for t in val_ids if t < 256).decode('utf-8', 'replace')

    if args.train:
        if args.resume and os.path.exists(args.weights):
            load_weights(model, args.weights)
            print(f"resumed weights <- {args.weights}")
        train(model, train_ids, val_ids, steps=args.steps, batch=args.batch,
              seq_len=args.seq_len, lr=args.lr)
        save_weights(model, args.weights)
        print(f"saved weights -> {args.weights}")
    elif os.path.exists(args.weights):
        load_weights(model, args.weights)

    if args.eval:
        val_ppl = model.eval_perplexity(val_ids, seq_len=args.seq_len, max_positions=256)
        train_ppl = model.eval_perplexity(train_ids, seq_len=args.seq_len, max_positions=64)
        micro = coding_microbench(model, val_text)
        samples = []
        for prompt in ['def fizzbuzz(n):', 'class Counter:']:
            ids = [BOS] + list(prompt.encode('utf-8', 'replace'))
            gen = model.generate(ids, n=args.n, greedy=True)
            samples.append((prompt, bytes(t for t in gen if t < 256).decode('utf-8', 'replace')))
        write_report(args.report, model, args, train_ppl, val_ppl, micro, samples)

    if args.selflearn_demo:
        selflearn_demo(model, val_ids, args.report)

    if args.generate:
        ids = [BOS] + list(args.prompt.encode('utf-8', 'replace'))
        gen = model.generate(ids, n=args.n, greedy=True)
        print(args.prompt + bytes(t for t in gen if t < 256).decode('utf-8', 'replace'))


if __name__ == "__main__":
    main()
