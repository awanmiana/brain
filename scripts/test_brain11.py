#!/usr/bin/env python3
"""Reproducible before/after model test + 'what max can it produce' probe.

    python3 scripts/test_brain11.py --label before --out models/eval_before.json
    python3 scripts/test_brain11.py --label after  --out models/eval_after.json

Measures, honestly:
  * held-out val PPL per model (same val split the trainer uses; not training data)
  * 12 seeded renamed-family coding tasks: one-shot / +healing / +best-of-N / loop
    with and without BM25 memory, using the real sandboxed verifier
  * language sanity on fixed prompts (non-empty, no runaway repetition, length)
  * agentic tool-call validity + correctness on fixed prompts
  * MAX production: huge `n`, ctx clamp, total length, params, cap
Nothing here is a claim of open-domain ability. Sampling is seeded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from brain11_loop import Cortex11, Teacher11  # noqa: E402
from brain11_model import Brain11Config  # noqa: E402
from brain11_router import Brain11Router  # noqa: E402
from brain11_tokenizer import encode  # noqa: E402
from brain11_train import evaluate  # noqa: E402

NAMES = {"coder": "brain11_coder", "language": "brain11_language", "agentic": "brain11_agentic"}


def val_ppl(model_name: str, router: Brain11Router, device: str) -> float | None:
    path = HERE / f"{NAMES[model_name]}_val.npy"
    if not path.exists():
        return None
    val = np.load(path)
    model = router.get_model(model_name).model
    model.eval()
    try:
        return round(float(evaluate(model, val, min(320, Brain11Config().ctx), device, max_windows=24)), 4)
    except Exception as exc:
        return f"error: {exc}"


def runaway_repetition(text: str) -> bool:
    toks = text.split()
    if len(toks) < 8:
        return False
    for k in range(1, 5):
        for i in range(0, len(toks) - 2 * k, k):
            if toks[i:i + k] == toks[i + k:i + 2 * k] and toks[i:i + k] == toks[i + 2 * k:i + 3 * k]:
                return True
    return False


def test_language(router: Brain11Router) -> dict:
    prompts = [
        "Hello! How are you today?",
        "What is a function in Python?",
        "Tell me something about Lahore.",
        "آپ کیسے ہیں؟",
        "اردو میں ایک چھوٹا جملہ لکھیں۔",
        "Translate to Urdu: Good morning.",
    ]
    rows = []
    for p in prompts:
        torch.manual_seed(1234)
        out = router.chat(p, n=96)
        text = out["text"].strip()
        rows.append({
            "prompt": p, "output": text, "language": out["language"],
            "non_empty": bool(text), "no_runaway_repetition": not runaway_repetition(text),
            "within_length": len(encode(text, add_bos=False, add_eos=False)) <= Brain11Config().ctx,
        })
    return {
        "checks_passed": sum(1 for r in rows if r["non_empty"] and r["no_runaway_repetition"] and r["within_length"]),
        "total": len(rows), "rows": rows,
    }


AGENTIC_CASES = [
    ("What is the weather in Lahore?", "get_weather", "city", "Lahore"),
    ("Calculate 17 * 4.", "calculate", "expression", None),
    ("Read the file notes.txt.", "read_file", "path", "notes.txt"),
    ("What time is it in UTC?", "get_time", "timezone", "UTC"),
    ("Send a message to Ali saying hi.", "send_message", "recipient", "Ali"),
]


def test_agentic(router: Brain11Router) -> dict:
    rows = []
    for task, tool, arg, val in AGENTIC_CASES:
        torch.manual_seed(99)
        d = router.agentic(task, status="new").get("decision", {})
        action = d.get("action")
        ok = (action in ("call_tool", "finish", "ask_user", "check_result")
              and d.get("tool") == tool and isinstance(d.get("args", {}), dict))
        if val is not None:
            ok = ok and str(d.get("args", {}).get(arg, "")).lower().find(val.lower()) >= 0
        rows.append({"task": task, "decision": d, "ok": bool(ok)})
    return {"checks_passed": sum(1 for r in rows if r["ok"]), "total": len(rows), "rows": rows}


def test_max_production(router: Brain11Router, model_name: str = "coder") -> dict:
    cfg = Brain11Config()
    model = router.get_model(model_name).model
    model.eval()
    prompt = "Once upon a time"
    ids = encode(prompt, add_bos=True, add_eos=False)
    # huge request + disable EOS to see the true clamp
    out = model.generate(ids, n=100_000, temperature=0.0, top_k=1, stop_ids=(), repetition_penalty=1.0)
    produced = len(out)
    total_len = len(ids) + produced
    max_new = cfg.ctx - len(ids)
    # long-prompt case
    long_ids = (ids * 200)[:cfg.ctx + 50]
    out2 = model.generate(long_ids, n=100_000, temperature=0.0, top_k=1, stop_ids=(), repetition_penalty=1.0)
    long_total = min(len(long_ids), cfg.ctx - 1) + len(out2)
    n = Brain11Config().n_params()
    return {
        "model": model_name,
        "ctx_window_tokens": cfg.ctx,
        "prompt_tokens": len(ids),
        "max_total_context_tokens": cfg.ctx,
        "max_generated_tokens_this_prompt": max_new,
        "actually_produced_tokens": produced,
        "total_length_after_generation": total_len,
        "never_exceeds_ctx": total_len <= cfg.ctx,
        "long_prompt_input_tokens": len(long_ids),
        "long_prompt_total_length": long_total,
        "long_prompt_never_exceeds_ctx": long_total <= cfg.ctx,
        "n_params_per_model": n,
        "total_params_3_models": 3 * n,
        "param_cap_total": 33_000_000,
        "under_cap": 3 * n <= 33_000_000,
        "note": "Generation length is a hard clamp: len(prompt)+generated <= ctx=320. "
                "Quality/reliability does NOT scale with length; it stops at a safe bound. "
                "HTTP streaming can chunk a reply but cannot exceed ctx per request.",
    }


def evaluate_all(label: str, device: str = "cuda") -> dict:
    torch.manual_seed(20261125)
    router = Brain11Router(device=device, lazy=True)
    cortex = Cortex11(router)
    teacher = Teacher11(cortex)

    results = {"label": label, "device": device, "params": Brain11Config().n_params(),
               "val_ppl": {m: val_ppl(m, router, device) for m in NAMES}}

    results["code_no_memory"] = teacher.evaluate(limit=12, n=3, rounds=2, use_memory=False)
    results["code_with_memory"] = teacher.evaluate(limit=12, n=3, rounds=2, use_memory=True)
    results["language"] = test_language(router)
    results["agentic"] = test_agentic(router)
    results["max_production"] = test_max_production(router)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = evaluate_all(args.label, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {args.label} ===")
    print("val_ppl:", res["val_ppl"])
    print("code_no_memory:", {k: round(v, 4) for k, v in res["code_no_memory"].items()})
    print("code_with_memory:", {k: round(v, 4) for k, v in res["code_with_memory"].items()})
    print(f"language: {res['language']['checks_passed']}/{res['language']['total']}")
    print(f"agentic:  {res['agentic']['checks_passed']}/{res['agentic']['total']}")
    print("max_production:", {k: res["max_production"][k] for k in
          ("ctx_window_tokens", "max_generated_tokens_this_prompt", "actually_produced_tokens",
           "never_exceeds_ctx", "total_params_3_models", "under_cap")})
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
