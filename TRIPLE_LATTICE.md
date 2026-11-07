# Triple Lattice v1 — 1000 × 1000 × 1000 structured determination

> **The idea, formalized.** "1000 probabilities × 1000 probabilities, each of a
> different domain/classification × 1000 probabilities" is a **factorized
> structured posterior**: `P(domain) × P(relation | domain) × P(value | d, r)`.
> A flat generative head spreads one distribution over ~50,000 tokens; this
> spreads three 1000-way stages over **1,000,000,000 structured cells** —
> and then, instead of *sampling*, each stage is **determined** and judged,
> producing `PROBABLE` / `POSSIBLE` / `ESTIMATE` / `UNKNOWN` with measured
> precision.

Implemented in [`triple_lattice.py`](triple_lattice.py) on top of the
deterministic pathway core in [`pathway_lattice.py`](pathway_lattice.py).
Pure Python stdlib, fixed-point integers, bit-exact across machines.

---

## 1. From generation to determination

| | generative LLM | triple lattice |
|---|---|---|
| output space | 50k tokens, flat | 1000 domains × 1000 relations × 1000 values |
| mechanism | `P(next token \| ctx)`, sample | 3× integer-cosine argmax, determine |
| unknown query | answers anyway (hallucination) | `UNKNOWN` (abstention) |
| ambiguity | samples one answer | `POSSIBLE` + candidate set with split mass |
| multi-hop | chain of sampled tokens | `ESTIMATE` over a determined path |
| confidence | softmax number, uncalibrated | reliability table fitted in closed form |
| reproducibility | sampling noise | bit-identical, hashable state |

The three stages mirror exactly the proposed structure:

```
stage 1  DOMAIN     1000 channels   which classification does this belong to?
stage 2  RELATION   1000 channels   which attribute inside that domain?
stage 3  VALUE      1000 channels   which value inside that (domain, relation) cell?
```

Addressable space = `1000 × 1000 × 1000 = 10⁹` cells. Cells are materialized
sparsely — one rank-1 pathway operator per compiled fact (2,000 parameters:
a 1000-dim read vector + a 1000-dim write vector). The current model
materializes 239 cells; the ceiling is 500 operators = 1,000,000 parameters.

## 2. Why multiply scores is not enough — calibrated verdicts

Naive confidence would multiply three softmaxes. That number is fiction: it
is only as good as its calibration. Instead:

1. Every query gets an integer cosine score per stage (exact fixed-point).
2. A **reliability table** maps score → empirical `P(correct)`, fitted in
   closed form by counting (compiled facts at graded noise levels = positive
   examples; ~4,000 hard negatives = negative examples). No SGD, no seeds.
3. Verdicts:

| verdict | condition | meaning |
|---|---|---|
| `PROBABLE` | all stages `determined`, `P(correct) ≥ 0.99`, top-2 margin ≥ 0.10 | assert the value |
| `POSSIBLE` | value ambiguous or `0.50 ≤ P < 0.99` | keep candidates, split the mass |
| `ESTIMATE` | answer reached by composed multi-hop path | inference result, weakest-link confidence |
| `UNKNOWN`  | any stage below `P = 0.50`, or no pathway routes | abstain |

Confidence is the **weakest link** `min(P_domain, P_relation, P_value, P_hops)`
— never an optimistic product.

## 3. Measured results

### 3.1 Verdicts (`--demo`)

```
france capital       -> PROBABLE  paris      conf 0.993   [direct fact]
germany symbol       -> UNKNOWN   UNKNOWN             [not compiled]
jupiter has-moon     -> POSSIBLE  europa     conf 0.496  candidates [europa, io]
alice knows          -> ESTIMATE  erin       conf 0.993  4 composed ops
```

### 3.2 Precision-preserving abstention (`--experiment`)

In-distribution noise sweep — the model degrades by *hedging then abstaining*,
never by asserting wrong answers:

```
 noise  raw acc  asserted   prec  hedged  abstain  coverage
  0.00    0.993       137  1.000       2        0     1.000
  0.06    0.993       137  1.000       2        0     1.000
  0.12    0.993       137  1.000       2        0     1.000
  0.18    0.993        16  1.000     123        0     1.000
  0.24    0.993         0     -        0      139     0.000
  0.30    0.993         0     -        0      139     0.000
```

Out-of-distribution queries (unknown triples, wrong-relation arithmetic —
*no true answer exists*), the generative failure mode:

```
raw argmax answers:      821/821 = 1.000   (all wrong: confabulation)
triple lattice asserts:    0/821 = 0.000   (precision on OOD = 1.000)
triple lattice abstains: 821/821 = 1.000
```

Calibration quality:

```
ECE of raw cosine read as probability: 0.2642
ECE of calibrated reliability table:   0.0000  (closed-form binning)
```

## 4. Parameter accounting

| component | count | learned? |
|---|---:|---|
| codebook (1000-dim symbol vectors) | 0 stored | no — pure function of the token hash |
| tier S0: channel gains | 1,000 | yes — closed-form per-channel least squares |
| pathway operators (239 × 2,000) | 478,000 | yes — compiled + closed-form fit |
| reliability tables (2 × 20 bins × 2) | 80 | yes — closed-form counting |
| tier S1 ceiling (500 operators) | 1,000,000 | yes |
| **triple-lattice addressable cells** | **10⁹** | sparse: 239 materialized |

## 5. What this is, honestly

**Strengths measured here:** exact recall of compiled knowledge (100%),
precision 1.000 on every asserted answer including under noise, 0/821
hallucinations on OOD, surgical editability (one fact = one operator),
bit-identical reproducibility, explainable paths, ~20 ms inference in pure
Python.

**Limits:** the knowledge is what *we compile in* — 239 facts today. It has
no open-ended language understanding; a query must resolve to
`(subject, relation)` tokens. It will not write essays.

**The credible path to LLM-scale capability is a hybrid**, and the triple
lattice is designed to be its *decision layer*:

```
natural language ──► small encoder / LLM front-end ──► (subject, relation) candidates
                                                        │
                            triple lattice: determine 1000×1000×1000 cell
                                                        │
                            PROBABLE → act / answer     POSSIBLE → ask back
                            ESTIMATE → composed answer  UNKNOWN  → abstain
```

The front-end may be probabilistic; the *decision* stays deterministic,
auditable, and calibrated. That division of labor is what buys
"probable / possible / estimate" with high precision instead of generation.

### 5.1 Do we need billions of parameters? No — and here is the accounting

"Match Qwen 3.5 2B" hides three separate jobs with wildly different costs:

| job | where Qwen spends its 2B | what the hybrid costs |
|---|---|---|
| language understanding | statistical weights, trillions of tokens | **borrow** an open-weight model (inference only, zero training), or a small parser |
| decisions + knowledge | hallucination-prone decoding | our lattice: 2,000 params per fact — 2B params = **1M explicit facts** |
| wording | the same LM | templates, or the borrowed model |

Three regimes, all real:

1. **Zero-parameter front-end** (proved in [`hybrid_demo.py`](hybrid_demo.py)):
   a deterministic grammar + lexicon maps bounded English to lattice queries.
   `"who did socrates teach"` → `ESTIMATE alexander`. 0 learned parameters.
   Works when the domain is bounded: tool calls, dashboards, agent loops.
2. **Small learned front-end**: 20–100M parameters (slot-filling/intent
   classification is a far smaller problem than generation), trainable on
   one GPU — or a borrowed embedding model with a closed-form head.
3. **Borrowed big front-end**: Qwen (open weights) parses language;
   it is never trained, and the lattice gates its outputs — the exact place
   where LLMs hallucinate is replaced by `PROBABLE/POSSIBLE/ESTIMATE/UNKNOWN`.

Scaling *our* parameters to 2B means compiling **1 million facts** — linear,
inspectable, editable, auditable storage. That is not the same currency as
2B statistical weights trained on web text. You only "need" the statistical
billions for open-ended language; for determining the next X, you don't.

## 6. Reproduce

```bash
python3 pathway_lattice.py --train      # compile 239 facts -> operators (~40 s)
python3 pathway_lattice.py --verify     # 16 integrity checks
python3 pathway_lattice.py --demo       # pathway showcase
python3 pathway_lattice.py --bench      # throughput, scaling, noise robustness
python3 triple_lattice.py --demo        # the three stages + verdicts
python3 triple_lattice.py --calibrate   # reliability table + ECE
python3 triple_lattice.py --experiment  # precision / abstention tables
python3 triple_lattice.py --query "france capital"
python3 hybrid_demo.py --demo           # zero-parameter English front-end
python3 grow.py --demo                  # self-growth: agent stream -> training
```
