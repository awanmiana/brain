# Brain — progress log

State as of 2026-11-23 (continued after leaving for the office).

## Status
- Model implementation + training + eval + self-learning: **done** (`coder_1m.py`).
- Gradient check: **PASS** on tiny config (worst relerr 1.66e-06).
- Best checkpoint: **160-step run, val PPL 16.90** — restored as the canonical
  weights (`coder_coding_weights.json` (renamed from `coder_1m_weights.json`),
  byte-identical to `coder_1m_weights_step160.json`).
- Resumed run (2026-11-23): `python3 coder_1m.py --train --steps 160 --resume`
  — 160 more steps from the canonical checkpoint (run took ~11 min, faster
  than the ~30 min estimate); val PPL improved — final train-loop eval
  **10.44**.
- `--eval` re-verification of the new checkpoint: held-out val PPL **10.17**,
  training PPL 16.28, micro-bench 0/3 exact snippets, avg char accuracy
  **0.306** (was 0.194). Report rewritten to `coder_1m_report.txt`; note the
  report's `steps=` line reflects the `--eval` invocation (default 20) — the
  canonical weights are the resumed run's.
- `coder_coding_weights.json` now holds the resumed checkpoint;
  `coder_1m_weights_step160.json` still snapshots the older one.
- Background round: 120 steps from scratch, val PPL 17.07 → **discarded**
  (worse than the 160-step checkpoint).
- Self-learning capability: built in (`SelfLearner`) and demonstrated
  (`--selflearn-demo`: probe PPL 17.48 → 11.51 → 9.12).
- Six expert models — the coding LM + the five purpose variants — now trained
  on the RTX 3050 GPU (torch ports `coder_1m_torch.py` /
  `train_variant_torch.py`):
  - `coder_coding_weights.json` — the coding LM (renamed from
    `coder_1m_weights.json`) — resumed training on GPU (500 steps).
  - `coder_debug_weights.json`, `coder_conversation_weights.json`,
    `coder_tools_weights.json`, `coder_logic_weights.json`,
    `coder_lang_weights.json` — the five purpose variants — GPU training.
- Four new experts — trained on the RTX 3050 GPU:
  - `coder_english_weights.json` — the 1M English language expert
    (`train_variant_torch.py --corpus corpus_english.txt`) — final val PPL 10.45
  - `coder_urdu_weights.json` — the 1M Urdu language expert
    (`train_variant_torch.py --corpus corpus_urdu.txt`) — final val PPL 2.84
  - `coder_vision_weights.json` — the 1M vision expert — image-captioning
    style: reads raw BMP image bytes (`corpus_vision.bin`) and emits the
    caption/label — final val PPL 1.21
  - `coder_audio_weights.json` — the 1M audio expert — reads raw WAV bytes
    (`corpus_audio.bin`) and emits a tone label — final val PPL 2.83
- Brain AI model (MoE) resume training on GPU: `python3 -u coder_1m_moe.py
  --train --steps 20000 --device cuda --resume` — completed: 20,000 more
  steps from the previous checkpoint (val PPL 27.59) → final val PPL 23.12.
- Brain AI model scaled up (Big MoE): `python3 -u coder_1m_moe.py --train
  --steps 2000 --device cuda --experts 8 --topk 2 --ffn 1024` — the Big MoE:
  **8,336,196 params** (n_experts=8, topk=2, ffn=1024) — trained on the
  combined corpus including the real repos (`requests`, `click`, `django`) —
  final val PPL **27.33** (step 2000; min 27.34 at step 1280).
- Brain Coder — the coding/agentic/debugging/troubleshooting expert — built
  and trained on GPU: `python3 -u train_brain_coder.py --steps 4000 --device cuda`
  (`BrainMoE(n_experts=6, topk=2, ffn=1024)`) — **6,344,568 params** — corpus:
  repo code + real repos + `corpus_debug.txt`, `corpus_tools.txt`,
  `corpus_logic.txt`, `corpus_lang.txt`, `corpus_english.txt`
  (`coder_expert_corpus()` in `coder_1m_moe.py`, `train_brain_coder.py`) —
  final val PPL **30.14** (step 4000; min 27.09 at step 2560; 4000 steps in
  ~594 s on the RTX 3050) — weights: `brain_coder_weights.json`.
- **Brain** — the generative MoE prior — implemented and trained on GPU:
  `brain.py` (class `Brain`), `train_brain.py`.
  `Brain(FIT_8GB)` is **429,999,104 total params / 128,009,216 active per
  token**: 93.64% of the parameters are the 8 SwiGLU experts per layer
  (6.29M each, 50.33M per layer); top-2 routing activates 2 experts per token.
  200-step base run (non-RoPE, `brain_weights.json`): loss **12.25** (21.30 at
  step 1), ~926 s on the RTX 3050.
- **Brain distillation (RoPE, teacher corpus)**: completed **800/800 steps**,
  final val **PPL 25.08**, loss 7.45 → 0.22, weights: `brain_distill_weights.json`.
  The trainer now uses `CE + 0.01 * aux`. **However**, the neural path still
  fails generation on unseen names (24 attempts, all NameError/SyntaxError).
  Root cause: not enough data (44k tokens).
- **Larger verified corpus generated**: `gen_verified_corpus.py` produced
  1,480 verified examples (39 families × 40 name variants), every one executed
  against its tests before writing. Appended to `corpus_teacher.txt` → combined
  corpus now **889,056 tokens** (20× more). Training resumed on this larger
  corpus for 300 more steps (PID 22828, saving every 50 steps).
- **Checkpoint cleanup**: deleted `brain_weights.json` (200-step base) and
  `brain_distill_weights_bad_aux.json` (corrupted run). Kept only
  `brain_distill_weights.json` (800-step RoPE, the only checkpoint worth
  keeping). Directory went from ~5.3 GB to ~2.1 GB.
- **BrainCortex** (verified plastic MoE) — design + first implementation:
  `BRAIN_CORTEX_DESIGN.md`, `brain_cortex.py` (BM25 memory + case-based
  analogy + sandboxed verifier + best-of-N/repair loop + eval harness),
  `seed_teacher_corpus.py` (450 verified task→solution pairs; 66 eval tasks),
  `corpus_teacher.txt`. Memory/analogy path (no model): **60/60** known-family
  tasks with held-out function names solved and verified; **0/6** novel tasks
  (the neural frontier). The generative neural path has **not yet passed** an
  eval; **lattice is now wired into brain_api.py's solve_code** via
  `lattice_manager.py` as the first exact check across 4 domains:
  - arithmetic (0-9 addition): `2 + 3 = 5`
  - code: `python comment-char = #`, `rust file-extension = .rs`
  - commands: `linux ls = ls [path]`, `git commit = git commit -m [message]`
  - formats: `json extension = .json`, `utf-8 bits-per-char = variable`
  It abstains if unsure, and the neural path takes over.
  pathway_lattice.py and pathway_seed.py are NOT modified — new domains are
  additive only via lattice_code_seed.py, lattice_commands_seed.py,
  lattice_formats_seed.py, and lattice_manager.py.

## Model map (do not confuse these)
| name | file | params | type | status |
|---|---|---:|---|---|
| **Brain** (new) | `brain.py` | 429,999,104 total / 128,009,216 active | generative MoE LM, BPE 2,048, RoPE | distillation resuming on 889k-token corpus; single entry point via brain_api.py |
| BrainMoE (server) | `archive/saas_backend.py` | 8,336,196 | generative byte-level MoE, ctx 64 | **ARCHIVED** — served wrong model on :8123 |
| Brain Coder | `archive/train_brain_coder.py` | 6,344,568 | generative byte-level MoE | **ARCHIVED** — superseded by 430M Brain |
| TinyCoder variants | `archive/coder_1m.py` + `archive/coder_*_weights.json` | ~1M each | generative byte-level LMs | **ARCHIVED** — separate files, not connected |
| Lattice | `pathway_lattice.py` | symbolic (~2k params/fact) | deterministic multi-hop reasoner | **Wired in** — exact pre-check in solve_code |
| BrainCortex | `brain_cortex.py` | wrapper | memory + verifier + repair loop | memory path measured; neural path in training |

The 8.3M server model is **not** the 430M Brain. All obsolete models are in
`archive/`. Brain (`brain.py` + `brain_api.py`) is the single active model.

## Files (active)

- `brain.py` — Brain MoE model (`Brain`, FIT_8GB/QWEN2B presets, SwiGLU
  experts, GQA, RMSNorm, RoPE, tied lm_head). Next-token prior, not a chatbot.
- `train_brain.py` — Brain trainer (torch + AdamW + cosine schedule, GPU).
- `brain_api.py` — **Single entry point** — loads brain_distill_weights.json,
  provides generate/solve_code/chat + lattice pre-check. No server needed.
- `brain_cortex.py` — BrainCortex loop: BM25 memory, case-based analogy,
  sandboxed verifier, router, best-of-N generation, repair, trace store, eval.
- `brain_stats.py` — parameter accounting (total vs active, per-component) and
  expert-routing audit (`--routing`).
- `BRAIN_CORTEX_DESIGN.md` — BrainCortex design: verified plastic MoE,
  verified expert iteration, hybrid neural-symbolic loop, roadmap, honest limits.
- `skills/brain.yaml` — OpenCode skill definition (wraps brain_api.py).
- `SKILLS_USAGE.md` — usage guide for brain_api.py + OpenCode skill.
- `gen_verified_corpus.py` — verified synthetic data generator.
- `seed_teacher_corpus.py` — verified teacher-corpus generator.
- `corpus_teacher.txt` — 1,930 verified task→solution examples (~889k tokens).
- `brain_cortex_eval.json` — 66 eval tasks (60 known-family with held-out
  function names + 6 novel tasks).
- `brain_distill_weights.json` — **800-step RoPE checkpoint** — the only Brain
  weights worth keeping (1.64 GB).
- `brain_distill_weights.partial.json` — step-250 partial checkpoint from
  cancelled run (resumable).
- `brain_distill_weights_bad_aux.json` — archived bad 300-step checkpoint
  (unscaled aux loss); do not use.
- `diag_overfit.py` — small-model memorization sanity check (loss 0.043 on a
  256-token sequence), used to isolate the aux-loss bug.
- `pathway_lattice.py` — deterministic symbolic reasoner (exact arithmetic
  + multi-hop facts). **NOT modified.** Wired into brain_api.py via
  lattice_manager.py as exact pre-check.
- `pathway_seed.py` — original seed facts. **NOT modified.**
- `lattice_manager.py` — multi-domain lattice router. Loads domain lattices
  independently, routes queries, abstains if unsure.
- `lattice_code_seed.py` — code/programming domain facts (languages, syntax,
  extensions, creators).
- `lattice_commands_seed.py` — curated command reference (Linux, Windows,
  macOS, Android, ESP, x86-64, git).
- `lattice_formats_seed.py` — data formats, encodings, data structures.
- `PROGRESS.md` — this file.

## Archived (in `archive/`, not used)

- `coder_1m.py` — TinyCoder-1M (1M params, stdlib-only, not MoE)
- `coder_1m_torch.py`, `train_variant_torch.py`, `train_expert_torch.py`
- `coder_1m_weights_step160.json`, `coder_coding_weights.json`,
  `coder_debug_weights.json`, `coder_conversation_weights.json`,
  `coder_tools_weights.json`, `coder_logic_weights.json`, `coder_lang_weights.json`,
  `coder_english_weights.json`, `coder_urdu_weights.json`,
  `coder_vision_weights.json`, `coder_audio_weights.json` — ten 1M variants
- `train_brain_coder.py`, `brain_coder_weights.json` — Brain Coder (6.3M MoE,
  superseded by 430M Brain)
- `saas_backend.py`, `test_saas.py` — served wrong 8.3M model on :8123


## Metrics history
- 20-step run: val PPL 30.61
- 160-step run trajectory: 39.60(20) → 25.13(40) → 22.86(60) → 20.08(80)
  → 19.00(100) → 17.66(120) → 17.11(140) → 16.90(160)
- 120-step background round (restarted from scratch): val PPL 17.07
- Resumed run (160 steps from the 16.90 checkpoint, max_positions=128 eval):
  14.93(20) → 14.28(40) → 12.53(60) → 11.50(80) → 12.37(100) → 11.52(120)
  → 11.60(140) → 10.44(160)
- Six-expert GPU training (RTX 3050, `coder_1m_torch.py` /
  `train_variant_torch.py`): final val PPL — coding 9.29, debug 7.65,
  conversation 8.39, tools 3.05, logic 11.37, lang 18.74 (500 steps each)
- Brain AI model (MoE) GPU resume training (20,000 steps): val PPL 27.59
  (checkpoint) → 23.12 (step 20000)
- Four new experts (RTX 3050 GPU): English 10.45, Urdu 2.84, vision 1.21,
  audio 2.83 (final val PPL per expert, 500 steps each)
- Big MoE (8,336,196 params, experts 8, ffn 1024): trained on the combined corpus incl. real repos (`requests`, `click`, `django`) — final val PPL 27.33 (2000 steps; trajectory: 33.98@560 → 31.58@800 → 30.03@920 → 27.34@1280 → 27.33@2000)
- Brain Coder (6,344,568 params, experts 6, topk 2, ffn 1024) — coding/agentic/debugging/troubleshooting expert — trained on GPU (4000 steps, RTX 3050) - final val PPL 30.14 (min 27.09 at step 2560) - `brain_coder_weights.json`
- Brain (429,999,104 total; 128,009,216 active; FIT_8GB: d=1024, 8 layers,
  8 heads, kv_heads=4, ffn=2048, experts=8, topk=2) — parameter split:
  experts 402,653,184 (93.64%), attention 25,165,824 (5.85%), embedding
  2,097,152 (0.49%), routers 65,536 (0.02%), norms 17,408. Each expert is
  6.29M; each layer has 50.33M of experts; top-2 routing uses 2 per token.
  200-step base run (~926 s, RTX 3050): loss 12.25 (start 21.30) —
  `brain_weights.json`
- BrainCortex memory/analogy eval (no model): 60/60 seen-family tasks with
  held-out function names solved and verified; 0/6 novel tasks
- Expert routing audit (`brain_stats.py --routing`, 64 teacher tokens, base
  checkpoint): all 8 experts used; overall utilization 10.1–15.5% (perfect
  balance = 12.5%). Per-layer skew exists (layer 0 e6 = 32%, layer 1 e2 = 0%)
  but 64 tokens is a small sample. Utilization is not quality: every expert is
  the same size (6.29M), and no expert is individually "bigger".
- First Brain distillation (300 steps, unscaled aux): CE 3.60, aux 8.12 →
  gibberish; trainer fixed to `CE + 0.01 * aux`; fresh 800-step retrain
  completed: val PPL 25.08.
- Random baseline (uniform over 260 tokens): ln(260) ≈ 5.56

## Self-learning demo (from report)
- probe PPL before learning: 17.479
- after 3 Adam steps on the probe example: 11.513 (delta −5.966)
- after 1 replay epoch (sleep): 9.116
- total self-learning steps: 7

## Brain11 — Three ~11M Models (new)

All three models completed training on RTX 3050 (GPU was free throughout).

| Model | Files | Train Time | Final Metrics |
|---|---|---|---|
| **Brain11 Coder** | `brain11_coder_weights.json` | 22s | loss 1.49, val PPL 5.10 |
| **Brain11 Language** | `brain11_language_weights.json` | 21s | loss 1.94, val PPL 5.17 |
| **Brain11 Agentic** | `brain11_agentic_weights.json` | 32s | loss 0.19, val PPL 1.21 |

Corpus sizes: Coder=873K train tokens, Language=62K train tokens, Agentic=43K train tokens.

`brain11_base.py` + three trainers build 3 independent ~12.8M models:

| Model | File | Corpus | Status |
|---|---|---|---|
| **Brain11 Coder** | `brain11_coder_weights.json` | `corpus_teacher.txt` | ✅ Done — 200 steps, loss 1.49, val PPL 5.10 |
| **Brain11 Language** | `brain11_language_weights.json` | `corpus_english.txt` + `corpus_urdu.txt` | ✅ Done — 200 steps, loss 1.94, val PPL 5.17 |
| **Brain11 Agentic** | `brain11_agentic_weights.json` | `corpus_agentic.txt` (generated from TOOL_FACTS) | ✅ Done — 300 steps, loss 0.19, val PPL 1.21 |

### Unified Entry Point

`brain11_router.py` — routes tasks to the right model (Coder/Language/Agentic):

| Task type | Routes to | Example |
|---|---|---|
| Coding | `brain11_coder` | "Write Python code for fizzbuzz" |
| Language | `brain11_language` | "Hello, what is the capital of France?" |
| Agentic | `brain11_agentic` | "Predict the next action in the plan" |

Lazy loading — loads only the model needed per task. Edge-friendly (load one model at a time).

See `Brain11.md` for architecture and usage.

## How to run Brain (OpenCode model provider)

Brain now appears as a **selectable model** in OpenCode:

1. **HTTP server:** `python3 -u brain_http_server.py --port 8765 --device cuda` (background)
   - OpenAI-compatible API: `/v1/models`, `/v1/chat/completions`
   - In OpenCode: run `/models` — Brain should appear as "Brain (430M MoE)"

2. **Stop server:** `.\stop_brain_server.ps1` — kills server processes

## Lattice train generator

- `lattice_train_generator.py` — generates training examples from lattice queries
- Output: `corpus_lattice_generated.txt` (46 examples from 4 domains)
- To use: `Get-Content corpus_lattice_generated.txt | Add-Content corpus_teacher.txt` then retrain

2. **Config files** (already set up):
   - `~/.config/opencode/opencode.jsonc` — provider config
   - `skills/brain.yaml` / `~/.opencode/skills/brain.yaml` — capabilities (optional)

## How to run Brain (no server needed)

**Direct Python (recommended for OpenCode subagents):**
```python
from brain_api import BrainAPI

brain = BrainAPI()  # auto-loads brain_distill_weights.json

# Raw generation
text = brain.generate("# task: write a function add(a, b) that returns a + b\n")
print(text)

# Code solving with full loop: lattice → memory → neural → verifier → repair
result = brain.solve_code(
    task="write a function add_func(a, b) that returns the sum of a and b",
    tests="assert add_func(2, 3) == 5",
    n=8, rounds=3,
)
print("ok:", result["ok"])
print("source:", result["source"])   # "lattice" | "memory" | "neural"
print("code:", result["code"])
```

**CLI (no server needed):**
```bash
# Solve a coding task (full verify/repair loop)
python3 brain_api.py --task "write a function add(a, b)" --tests "assert add(2,3)==5"

# Raw generation from a prompt
python3 brain_api.py --prompt "# task: write a function add(a, b)\n"

# Conversational (best-effort; Brain is code-trained, not chat-tuned)
python3 brain_api.py --chat "hello"

# Disable memory/analogy path (neural only)
python3 brain_api.py --task "..." --tests "..." --no-memory

# Disable lattice exact check
python3 brain_api.py --task "..." --tests "..." --no-lattice
```

**OpenCode skill:**
```powershell
# Install (one-time):
mkdir -p ~/.opencode/skills
Copy-Item skills/brain.yaml ~/.opencode/skills/brain.yaml

# In OpenCode:
/skill load brain
/skill list
```

**BrainCortex loop (memory + verifier + repair):**
```bash
# Self-test (no model needed)
python3 brain_cortex.py --selftest

# Full eval on seen-family vs novel splits
python3 brain_cortex.py --eval-file brain_cortex_eval.json --split seen_family --n 4 --rounds 2 --weights brain_distill_weights.json --rope
python3 brain_cortex.py --eval-file brain_cortex_eval.json --split novel --n 4 --rounds 2 --weights brain_distill_weights.json --rope
```

## Notes for continuation
1. Resume training (cancelled at step 260/300, partial at step 250):
   `python3 -u train_brain.py --train --steps 300 --device cuda --corpus corpus_teacher.txt --rope --weights brain_distill_weights.json --lr 5e-4 --save-every 50 --resume`
2. Re-evaluate: `python3 brain_cortex.py --eval-file brain_cortex_eval.json --split seen_family --n 4 --rounds 2 --weights brain_distill_weights.json --rope`
3. Generate more verified data: `python3 gen_verified_corpus.py` then append to corpus_teacher.txt and retrain.
4. Honest status: the ~1M-param model learns measurably (val PPL ~10.2 vs
   random 5.56) but remains compact; more steps/data/compute would improve it.

---

## 2026-11-26 — Brain11 v2 corpus + training + reactive loop (measured)

- Corpus rebuilt: `brain11_code.jsonl` 2,447/6.39MB; `brain11_language.jsonl` 13,974/12.71MB;
  `brain11_agentic.jsonl` 12,000/2.47MB; tokenizer `brain11_tokenizer.json` 4096 BPE.
- Urdu fixed: Wikipedia MediaWiki API hit HTTP 429 -> 17 articles/0.01MB; now HF datasets-server
  `wikimedia/wikipedia` `20231101.ur`, 327 articles, **4.56MB / 3,128 docs**.
- **Trained**: coder 3000 steps val PPL **7.33**; language 3000 steps val PPL **14.47**;
  agentic 1500 steps val PPL **1.21**. All `10,793,088` params. RTX 3050 ~50k tok/s. Total **32,379,264**.
- **Kill/resume proven**: kill mid-run -> last atomic checkpoint intact -> `resumed at step N`, no `.tmp`.
- Bugs found & fixed (this session): infinite loop in `build_agentic` (len-based variants); `np.save`
  appends `.npy` breaking `os.replace`; `nonlocal best_val`; wrong Python 3.9/CPU-torch interpreter;
  **`clean_text` collapsed all code indentation in `corpus_teacher.txt`**; ctx overflow in `generate`;
  non-reproducible eval; UTF-8 BOM.
- Coding loop eval (12 renamed families, seeded): neural-only **one-shot 0.50 / loop 0.67**;
  +BM25 memory **1.00**. Agentic JSON + tool selection reliable.
- **New `brain11_reactive.py`**: explicit `CodeState`, `pending_requirements`, `already_satisfied`,
  forward/backward hops, full re-check + test re-run each round; `--selftest` 15/15.
- **`brain11_planner.py`**: plan-first + AST dataflow + insert_setup + compose ordering; selftest 11/11.
