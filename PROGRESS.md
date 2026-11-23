# TinyCoder-1M — progress log

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
- Brain AI model (MoE) resume training on GPU: `python3 -u coder_1m_moe.py
  --train --steps 20000 --device cuda --resume` — completed: 20,000 more
  steps from the previous checkpoint (val PPL 27.59) → final val PPL 23.12.

## Files
- `coder_1m.py` — the model. TinyCoder-1M: 3-layer pre-LN GPT-style transformer,
  d=162, 2 heads (head_dim 81), FFN=648, ctx=64, tied byte-level head.
  Parameters: **999,540 (≈ 1M)**. Hand-written Adam + `SelfLearner`
  (bounded online adaptation with replay/"sleep" replay).
- `coder_1m_torch.py` — torch port of the coding model (same weight format).
- `train_variant_torch.py` — torch port of the variant trainer (GPU).
- `coder_coding_weights.json` — canonical coding weights (the resumed
  160-step run; see Status).
- `coder_1m_weights_step160.json` — snapshot of the pre-resume checkpoint
  (the older, 16.90 one).
- `coder_1m_report.txt` — eval report (held-out val PPL 10.17, train PPL
  16.28, micro-bench 0/3 exact, avg char accuracy 0.306). The self-learning
  demo section is not in the report — re-append with `--selflearn-demo`
  (see Status/Notes).
- `PROGRESS.md` — this file.

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
- Random baseline (uniform over 260 tokens): ln(260) ≈ 5.56

## Self-learning demo (from report)
- probe PPL before learning: 17.479
- after 3 Adam steps on the probe example: 11.513 (delta −5.966)
- after 1 replay epoch (sleep): 9.116
- total self-learning steps: 7

## Notes for continuation
1. To keep improving the coding model: `python3 -u coder_1m_torch.py --train
   --steps 500 --device cuda --resume --weights coder_coding_weights.json`
   (torch port, GPU — the earlier stdlib run pulled val PPL from ~16.9 to
   ~10.4; the GPU runs are faster and explore more).
2. To re-evaluate the current model: `python3 -u coder_1m.py --eval`
   (rewrites `coder_1m_report.txt`). Note: `--eval` drops the
   "Self-learning demo" section from the report — re-append it with
   `--selflearn-demo` if the demo record is wanted.
3. Optional sample: `python3 -u coder_1m.py --generate --prompt "def fizzbuzz(n):" --n 30`.
4. Honest status: the ~1M-param model learns measurably (val PPL ~10.2 vs
   random 5.56) but remains compact; more steps/data/compute would improve it.
