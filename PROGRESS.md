# TinyCoder-1M — progress log

State as of 2026-11-23 (continued after leaving for the office).

## Status
- Model implementation + training + eval + self-learning: **done** (`coder_1m.py`).
- Gradient check: **PASS** on tiny config (worst relerr 1.66e-06).
- Best checkpoint: **160-step run, val PPL 16.90** — restored as the canonical
  weights (`coder_1m_weights.json`, byte-identical to
  `coder_1m_weights_step160.json`).
- Background round: 120 steps from scratch, val PPL 17.07 → **discarded**
  (worse than the 160-step checkpoint).
- Self-learning capability: built in (`SelfLearner`) and demonstrated
  (`--selflearn-demo`: probe PPL 17.48 → 11.51 → 9.12).

## Files
- `coder_1m.py` — the model. TinyCoder-1M: 3-layer pre-LN GPT-style transformer,
  d=162, 2 heads (head_dim 81), FFN=648, ctx=64, tied byte-level head.
  Parameters: **999,540 (≈ 1M)**. Hand-written Adam + `SelfLearner`
  (bounded online adaptation with replay/"sleep" replay).
- `coder_1m_weights.json` — canonical weights (the 160-step checkpoint).
- `coder_1m_weights_step160.json` — snapshot of the same checkpoint.
- `coder_1m_report.txt` — eval report (val PPL 15.31, train PPL 23.34,
  micro-bench 0/3 exact, avg char accuracy 0.194) + self-learn demo section
  (probe PPL 17.48 → 11.51 → 9.12).
- `PROGRESS.md` — this file.

## Metrics history
- 20-step run: val PPL 30.61
- 160-step run trajectory: 39.60(20) → 25.13(40) → 22.86(60) → 20.08(80)
  → 19.00(100) → 17.66(120) → 17.11(140) → 16.90(160)
- 120-step background round (restarted from scratch): val PPL 17.07
- Random baseline (uniform over 260 tokens): ln(260) ≈ 5.56

## Self-learning demo (from report)
- probe PPL before learning: 17.479
- after 3 Adam steps on the probe example: 11.513 (delta −5.966)
- after 1 replay epoch (sleep): 9.116
- total self-learning steps: 7

## Notes for continuation
1. To keep improving: `python3 -u coder_1m.py --train --steps 160 --resume`
   (resumes from the canonical checkpoint; ~30 min). Expect mild gains
   (~0.1 PPL per 20 steps); diminishing returns below ~15.
2. To re-evaluate the current model: `python3 -u coder_1m.py --eval`
   (rewrites `coder_1m_report.txt`) then `--selflearn-demo` to re-append.
3. Optional sample: `python3 -u coder_1m.py --generate --prompt "def fizzbuzz(n):" --n 30`.
4. Honest status: the ~1M-param model learns measurably (val PPL ~15-17 vs
   random 5.56) but remains compact; more steps/data/compute would improve it.
