# Affective-State Engine v5 — the folded Bayesian engine

A deterministic, pre-trained affective-state engine whose **1,000,000 weights are
folded from exactly 1,000 parameters**, with a real Gaussian posterior over the
readout instead of a softmax temperature.

Runs on **pure Python 3.10+ stdlib** (no numpy, no pip, no network).

```bash
python3 affective_engine_v5.py --demo          # 7 illustrative inputs
python3 affective_engine_v5.py --verify        # unfold 1M weights + integrity checks
python3 affective_engine_v5.py --report        # fold + head + weight inventory
python3 affective_engine_v5.py --text "..."    # one input, full JSON
echo "..." | python3 affective_engine_v5.py --stdin
```

First run pretrains (~3.5 min) and writes `affective_v5_readout.json`; later runs
load it in ~1.6 s. `--rebuild` ignores the cache, `--cache none` disables it.

---

## 1. The fold: 1000 → 1,000,000, exactly

The genome is a vector `c ∈ R^1000`. It is embedded as a **circulant** matrix:

```
W[i, j] = sqrt(N) * c[(i - j) mod N],      N = 1000
```

* **Exactly 1,000,000 entries**, from exactly 1,000 parameters (1 : 1000).
* `c` is built from a **unit-modulus spectrum**: `c = IDFT(λ)` with `|λ_k| = 1`.
  Because a circulant is diagonalised by the DFT, `W x = sqrt(N) · IDFT(λ ⊙ DFT(x))`
  and therefore

  ```
  (1/N) WᵀW = I          (measured defect: 2.7e-13)
  ```

  The unfolded matrix is an **exactly orthogonal random rotation**: norm-preserving,
  perfectly conditioned, and the best-known random-feature mixer.

Costs:

| operation                  | cost      | measured        |
|----------------------------|-----------|-----------------|
| read one weight `W[i,j]`    | O(1)      | exact           |
| materialise one column      | O(N)      | exact           |
| materialise all 1M weights  | O(N²)     | 1.2 s           |
| apply `W x`                 | O(N log N)| ~10 ms          |
| apply during inference      | O(N log N)| no dense matrix |

A hand-written mixed-radix FFT is included (N = 1000 = 2·2·2·5·5·5).

## 2. Where the million weights act

```
text ──► symbolic appraisal ──► φ ∈ R^1000, ‖φ‖ = 1   (hashed stems, char trigrams,
                                                        category saturation, affect)
                                    │
                     h = tanh(W φ + b)      ◄── all 1,000,000 folded weights
                                    │
        z = [cats(32), evidence(1), magnitude(1), h[0:256]]   ∈ R^290
                                    │
                     y = Mᵀz + e,   e ~ N(0, σ²I)            (Bayesian head)
                                    │
        30 state logits (temperature-calibrated) + 10 affect dimensions
```

The teacher's own latent affect vector is **withheld** from the head, so the
million-weight core is the only route from text to affect — it cannot be
bypassed. (A diagnostic "oracle" view that does receive the latent is trained
alongside, purely as an upper bound.)

## 3. Pre-training in closed form (no SGD, no gradients)

Distillation targets: the v3 symbolic appraiser's 30 state scores + 10 affect
values, over a deterministic synthetic curriculum (2000 train / 1000 validation).

```
hbar, sd = feature mean / std                       # standardisation
A        = Zcᵀ Zc + λI                              # Zc = standardised features
M        = A⁻¹ Zcᵀ Y                                # posterior mean (d×40)
L        = chol(A)                                  # posterior covariance factor
σ²       = residual variance on held-out data
T        = softmax temperature minimising held-out NLL
```

`λ` is selected from a log grid on the validation split. Everything is the
unique solution of a linear system: no seeds to initialise, no stopping rule,
no run-to-run variance.

**Measured ablation** (held-out 1000 samples, teacher = reference):

| view                        | features | val MSE     | top-1 |
|-----------------------------|---------:|------------:|------:|
| cats + evidence (no core)   |       34 | 0.039111    | 0.660 |
| folded core only            |      256 | 0.021304    | 0.577 |
| **cats + evidence + core**  |  **290** | **0.017340**| **0.649** |
| oracle (+ teacher latent)   |      300 | 0.012068    | 0.728 |

The core **halves the error** of the symbolic skip (2.25×) and recovers ~63% of
the distance to the oracle. Head calibration: NLL 1.032, ECE 0.035 on 30 states.

## 4. The probabilistic mechanism

Two independent epistemic sources are propagated, both sampled deterministically
from fixed input-independent noise banks (so the same text always gives the same
entropy):

1. **Posterior over the 1,000,000 core weights.**
   A norm-preserving perturbation of the spectrum,

   ```
   λ_k → λ_k · exp(i ε_k),    ε_{-k} = -ε_k,   ε_k ~ N(0, 0.02²)
   ```

   which is a draw from the orthogonal group and preserves `(1/N)WᵀW = I`.
   Each draw is one extra FFT pair (O(N log N)), never a dense matvec.

2. **Posterior over the readout weights.**

   ```
   w ~ N(M, σ²A⁻¹),    u = L⁻¹(z - z̄),    draw = Mᵀz + σ·(E·u),  E ~ N(0, I)
   ```

   4 core draws × 4 head draws = 16 predictive samples per input.

From those samples:

* `predictive_entropy` — total uncertainty of the state distribution,
* `expected_entropy` — aleatoric part,
* `mutual_information` — epistemic part (what more data would remove),
* per-dimension `epistemic_std_affect`.

That is a genuine decomposition, not a temperature trick. Low-evidence inputs
abstain instead of hallucinating a state.

## 5. Reproducibility

* Same text → byte-identical JSON across processes (verified).
* `--verify` re-derives the fold and checks: 1000×1000 shape, entry formula,
  FFT-vs-dense matvec agreement (1e-9), norm preservation, orthogonality defect,
  head dimensions, lower-triangular Cholesky, entropy decomposition.
* Cache is keyed by a SHA-256 digest of the lexicon, affect table, prototypes,
  seeds and training sizes; any change invalidates it automatically.
* Genome SHA-256: printed by `--verify`.

## 6. Files

```
affective_engine_v5.py        the engine (single file, stdlib only)
affective_v5_readout.json     cached posterior (auto-generated, ~1.3 MB)
```

## 7. Parameter accounting — what is actually "real"

| kind                         |    count | learned? |
|------------------------------|---------:|----------|
| core genome `c`              |    1,000 | no — deterministic from a seed via a unit-modulus spectrum |
| unfolded core `W`            | 1,000,000 | no — derived O(1) from `c`; a real operator, not a million independent knobs |
| readout mean `M`             |   11,600 | yes — closed-form ridge |
| readout precision factor `L` |   42,195 | yes — from the same solve |
| feature centre / scale       |      580 | yes |
| output bias `ybar`           |       40 | yes |
| **stored scalars**           | **55,415** | |
| **used per inference**       | **1,011,600** | |

This is the reservoir / random-features design (Rahimi & Recht "Random Kitchen
Sinks"; Yu et al. orthogonal random features; echo-state networks): a large
*frozen* structured operator followed by a small *learned* readout. For scale,
GPT-2 small has 124M *learned* parameters; this engine has ~12k learned weights
plus a 1M-entry frozen orthogonal operator. It is not a 1M-parameter LLM, and it
does not claim to be.

## 8. Why this beats the earlier fold attempts

* Hash-PRF "folds" store a seed and call it zero parameters; here the genome is
  a real 1000-float vector with a one-line inverse and a printed hash.
* Butterfly folds need ~11,000 parameters for a 1000×1000 matrix; this needs
  exactly 1000 and is provably orthogonal.
* The head is trained (closed form, validated, calibrated) rather than four
  hand-written prototypes, and its uncertainty is a posterior rather than a
  softmax temperature.
