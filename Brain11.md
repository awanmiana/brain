# Brain11 — Three ~10.8M Models (33M total)

Edge-friendly, fully local. **One shared 4,096-token byte-level BPE tokenizer**, three
independent 10,793,088-parameter specialists, one router, one verifier loop.

| Model | Corpus | Val PPL (measured) |
|---|---|---|
| `brain11_coder` | `brain11_code.jsonl` | **7.33** |
| `brain11_language` | `brain11_language.jsonl` | **14.47** |
| `brain11_agentic` | `brain11_agentic.jsonl` | **1.21** |

**Total 32,379,264 params**, ~43.2 MB fp32 per model. Cap: 33M total / ~11M each.

## Architecture (v2)

`brain11_model.py` — `Brain11Config(d=384, layers=5, heads=6, ffn=1088, ctx=320, vocab=4096, dropout=0.05)`

- RoPE rotary embeddings, **no learned positional table**
- RMSNorm (pre-norm), **SwiGLU** FFN
- **Tied** input/output embeddings
- Fused causal scaled-dot-product attention
- Deterministic per-step sampling (reproducible eval)
- Forward/backward verified; ~107 MB VRAM for the smoke test; **45–60k tok/s** on an RTX 3050

## Corpus (`brain11_prepare.py`)

| Domain | Source | Docs | Size |
|---|---|---|---|
| **Code** | curated task→code→tests `corpus_teacher.txt` (indentation preserved) | 1,930 | 0.88 MB |
| **Code** | this repo | 222 | 1.7 MB |
| **Code** | Python stdlib | 249 | 4.2 MB |
| **English** | TinyStories V2 valid | 10,506 | 8.4 MB |
| **Urdu** | Urdu Wikipedia (HF `wikimedia/wikipedia` `20231101.ur`) | 3,128 docs | 4.56 MB |
| **English+Urdu chat** | hand-written dialogues, translations, slot-filled variants | ~600 | <0.1 MB |
| **Agentic** | generated task/status → tool-call JSON, 8 tool families | 12,000 | 2.47 MB |

`brain11_code.jsonl` 2,447 docs / 6.39 MB · `brain11_language.jsonl` 13,974 / 12.71 MB ·
`brain11_agentic.jsonl` 12,000 / 2.47 MB. ~2.13M / ~3.43M / ~0.68M train tokens.
**Nothing is deleted; corpora are read-only.**

```bash
python3 brain11_prepare.py --english-mb 8 --urdu-mb 4 --agentic 12000
python3 brain11_prepare.py --only code          # rebuild one corpus; tokenizer untouched
```

## Training — chunked, atomic, resumable

`python3 brain11_train.py --model coder --steps 3000 --device cuda`
`--model all` trains all three. Chunk 100 steps, **save every 50**, warmup+cosine LR, grad-clip 1.0,
AdamW (0.9/0.95, wd 0.1 on 2-D only), AMP fp16, TF32, 90% VRAM cap, disk check, OOM→save→exit.

Every write is `tmp + fsync + os.replace`. On interrupt you lose ≤50 steps, **never data, never a valid checkpoint**.

```text
brain11_<name>_weights.pt   brain11_<name>_optim.pt   brain11_<name>_best.pt
brain11_<name>_state.json   brain11_<name>_log.jsonl (append-only)
```

Resume prints `resumed at step N (... tokens seen)`. **Stop anytime; rerun to continue.**
Use the Python that has the CUDA build of torch; the trainer refuses `--device cuda` otherwise.

## Unified router (`brain11_router.py`)

```python
from brain11_router import Brain11Router
r = Brain11Router(device="cuda")
r.chat("Hello")                                    # language
r.chat("آپ کیسے ہیں؟")                              # Urdu
r.code("write a function add(a, b) that returns a + b")   # coder
r.agentic("What is the weather in Lahore?", status="new") # agentic -> valid JSON
```

Lazy loading: only the needed model occupies VRAM. Auto-routing by task type.
`repetition_penalty`, `top_k`, `temperature`, context-length clamping included.

## OpenAI-compatible server

```bash
python3 brain_http_server.py --host 127.0.0.1 --port 8765
```
`/v1/models`, `/v1/chat/completions`, `/health`; streaming SSE. Model IDs:
**`brain`**, `brain11-coder`, `brain11-language`, `brain11-agentic`. One name **`brain`** everywhere.

## Self-learning / healing loop (`brain11_loop.py`)

```
route -> retrieve verified analogy -> best-of-N
      -> sandboxed test -> deterministic heal -> error-driven repair
      -> remember only verified wins -> next chunk re-trains on them
```

- `--selftest` 9/9 · `--doctor` (+`--fix`) · `--solve T --tests "assert ..."`
- `--learn N` / `--eval` with `--no-memory`, reproducible seeds, writes `brain11_eval_last.json`
- **Nothing is learned without passing real tests.** Verified examples go to `brain11_selflearn.jsonl`
  (dedup + provenance), repairs to `brain11_selflearn_repairs.jsonl`; self-learned data is capped at
  25% of base corpus; base data is never overwritten.
- **Verifier**: AST safety, forbidden imports, timeout, sandbox; `assert`-based tests.
- **Healer**: missing colon, indentation, wrong function name, undefined name, truncated EOF,
  markdown stripping, `unexpected eof`.
- **Memory**: BM25 over verified traces + corpus; name normalization to the task-specified function.

### Measured (12 held-out renamed families, `n=3`, `rounds=2`, seeds fixed)

| | one-shot | +healing | best-of-N | loop |
|---|---|---|---|---|
| **no memory** | 0.50 | — | 0.67 | **0.67** |
| with BM25 memory | 1.00 | — | 1.00 | **1.00** |

## Prerequisite-first / reactive construction

`brain11_planner.py` (`plan-first SFT` + AST dataflow) and **`brain11_reactive.py`**.

**`ReactiveCoder`** implements the state machine explicitly:

```
state_t = prompt + current code
  -> per-section logical requirements
  -> done[] + pending[]
  -> choose action: FORWARD (append/complete) | BACKWARD (reorder/patch earlier text)
  -> re-run full check from the top + re-run tests
  -> state_t+1
```
Every hop is recorded (`Hop.round/action/detail/pending_before/pending_after`).
`python3 brain11_reactive.py --selftest` → **15/15**.

## Honest position

A 10.8M model is a **pattern + policy engine, not a general coder**. Narrow, verified tasks are
reliable; novel open-ended coding is not. Expect roughly 0.5–1.0 on supported families with the
loop, and **abstain** otherwise. It cannot match Qwen-class models on open coding.

## v1 (superseded, archived)

Byte-level 12.8M `brain11_base.py`, `train_brain11_*`, `gen_agentic_data.py`, `*_weights.json`.
