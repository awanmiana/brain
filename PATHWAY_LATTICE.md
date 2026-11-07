# Pathway Lattice v1 — determination, not generation

> A deterministic, non-generative reasoning core. A generative model learns
> `P(next token | context)` and **samples**. This model **determines** the
> next X: the context encodes to a 1000-dim integer vector, a router picks
> exactly one rank-1 operator, and the path composes **1000 → path → 1000**.
> If no path exists, the answer is `UNKNOWN` — never a hallucination.

Implemented in [`pathway_lattice.py`](pathway_lattice.py). Pure Python
stdlib, fixed-point integer arithmetic (exact `int` dot products), so two
runs on any machine produce **bit-identical** results.

---

## 1. The core formula

```
next X  =  decode( compose(path) ( encode(context) ) )

encode(s, r) = normalize(vec(s) + vec(rel:r))        ∈ Z^1000, ‖·‖ = ONE
route(h)     = argmax_p  cos(v_p , h)   subject to   cos ≥ τ   (else ABSTAIN)
apply(h)     = normalize( u_p · (v_p · h) )          rank-1 write
walk         = repeat route/apply, masking visited  -> determined path
decode(state)= argmax_symbol cos(state, code)        exact nearest code
```

* Every operator is **rank-1**: `h' = u·(v·h)` with `u, v ∈ Z^1000` —
  exactly **2,000 parameters**. The 1000 × 1000 outer product `u vᵀ` is the
  1,000,000-entry matrix that is *never materialized*; one application costs
  two 1000-dim dot products.
* The **path** is the computation. A walk of length L composes L rank-1
  maps; a `rank path` composes L parallel rank-1 writes into a rank-L map
  (path rank = capacity, fitted exactly by orthogonal deflation).
* Routing is `argmax` with lowest-index tie-break on exact integer cosines.
  There is no softmax, no temperature, no sampling, no beam search.

## 2. Training is closed form

No SGD, no epochs of gradient noise, no seeds:

| step | method |
|---|---|
| fact compilation | one operator per `(subject, relation) → object` pair |
| rank-1 fit | exact solution `u = normalize(y), v = normalize(x)` |
| rank-L fit | Gram-Schmidt deflation + exact least squares (Fractions) |
| refinement | deterministic NLMS, canonical example order, fixed step |
| router threshold | grid scan: smallest τ separating 4,101 negatives from every positive |

The calibrated routing threshold in the trained model is **0.81** with margin
`pos_min 1.000 − neg_max 0.809` (the nearest negative is a deliberately hard
confusable query).

## 3. The tiers: 1,000 → 1,000,000

| tier | learned params | what it is |
|---|---:|---|
| S0 seed | 1,000 | 1,000 per-channel gains over the frozen codebook (closed-form least squares; measured denoising MSE 0.752 → 0.320, **2.35×**, on heteroscedastic channel noise) |
| S1 growth | P × 2,000 | P rank-1 pathway operators |
| S1 ceiling | 1,000,000 | 500 operators — the capacity budget |

The trained model compiles 239 facts/tools/arithmetic into 239 operators =
**478,000 learned parameters**, deterministic state hash
`a55e1006b2aa5f07…`.

## 4. What it does

```bash
python3 pathway_lattice.py --demo
```

1. **Direct determination** — `france capital → paris`, `gold symbol → Au`.
2. **Multi-hop path walking** — `socrates teaches → alexander` through
   `plato`, `aristotle`; `alice knows → erin` through 4 operators; the path
   itself is the explanation.
3. **Exact arithmetic** — the 0..9 addition table compiled to 100 operators,
   100/100 exact, no sampling.
4. **Tool-call determination** — `ask-weather → tool-get-weather → city-arg
   → string`: the next X of an agent loop is *determined*, not generated.
5. **Abstention** — `germany symbol → UNKNOWN`; `3 times 5 → UNKNOWN`
   (the table is plus-only).
6. **Surgical editing** — delete one operator, one fact disappears; restore
   it, the fact returns.
7. **Rank paths** — 8 fact pairs compressed to a rank-2 path (4,000 params
   instead of 16,000) with graceful error; rank-8 fits exactly (MSE 1e-6,
   the fixed-point floor).

`--verify` runs 16 checks (kernel bounds, routing, reasoning, abstention,
determinism, parameter accounting, rank-path exactness) — all green.

## 4.5 The self-growth loop: training == compilation

`grow.py` answers "if an agent feeds this brain data, does it get trained?"
— **yes**: `learn_fact()` compiles each datum into one operator in closed
form (+2,000 parameters) and the very next query can use it. No epochs, no
gradient descent, no replay buffer. Contradiction policy: functional
relations (`capital`) report `conflict` when a second answer appears;
multi-valued relations (`has-moon`) keep every answer and the Triple Lattice
surfaces the ambiguity as `POSSIBLE`. Measured: a 2,000-fact stream grows
the model 478,000 → 4,478,000 parameters at exactly 2,000 per fact.

## 5. Honest limits

* Knowledge = what is compiled in (239 operators today). Scaling knowledge
  is linear and inspectable, but it is not web-scale.
* Queries must resolve to `(subject, relation)` tokens — no open-ended
  language understanding.
* The confusable-hard-negative margin (0.81 vs 1.0) is adequate but not
  luxurious; position-coded token slots would widen it.

The layered extension — 1000 × 1000 × 1000 structured verdicts with
calibrated probabilities — is in [`TRIPLE_LATTICE.md`](TRIPLE_LATTICE.md).
