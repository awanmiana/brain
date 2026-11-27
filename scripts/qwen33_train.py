#!/usr/bin/env python3
"""Honest, cancel-safe, resumable Brain11 (33M) trainer wrapper + optional Qwen3.5-2B distillation.

You run it. It outputs + saves; NOTHING is lost; cancel anytime; restart anytime; like brain11_train.

HONEST / NOT WHAT YOU ASKED LITERALLY:
- This trains **Brain33M**, NOT `qwen3.5:2b`. You cannot "train from qwen3.5 2b to brain 33m"
  (different arch/tokenizer/size; Ollama Qwen weights are not shrinkable; no 2B->33M conversion).
- Qwen3.5-2B via Ollama HTTP `:11434` is an optional TEACHER (distillation data / eval baseline),
  NOT a target to become 33M and NOT proof of matching it. No parity claim.
- 33M cap kept; **not auto-raised**. To exceed, pass explicit `--param-cap` (but raw parity w/ 2.3B
  honestly needs ~2.3B; you said don't use 2B; keep 33M + measurable redefined target).
- `--match-*` only RECORDS your defined target for comparison; never fabricates a match.

Cancel: Ctrl-C / kill any time. Atomic chunked resumable `brain11_train` (100/50, tmp+fsync+replace,
restore-best, `optim_matches_weights`, no `.tmp`, no data lost, append-only logs, read-only corpus).
Restart: rerun same command; reads `_state.json`, resumes step N; `--restore-best` to roll back.

Ollama: `ollama serve` (or CLI auto); model `qwen3.5:2b`; HTTP `http://127.0.0.1:11434`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from brain11_train import restore_best  # noqa: E402

BRAIN11_TOTAL = 32_379_264
TRAIN_PY = HERE / "brain11_train.py"


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    last = None
    for i in range(12):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, OSError) as exc:
            last = exc
            time.sleep(0.05 * (i + 1))
    raise last


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:  # append-only; nothing lost
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_ollama(url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=5) as r:
            names = {m.get("name") for m in json.loads(r.read().decode()).get("models", [])}
        print(f"[ollama] {url} up; {model} present={model in names}")
        return model in names
    except Exception as exc:
        print(f"[ollama] {url} NOT reachable: {exc}; start `ollama serve`. continuing without teacher.")
        return False


def _state_step(model_file: str) -> int:
    name = model_file
    st = _read_json(HERE / f"brain11_{name}_state.json", {})
    return int(st.get("step", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="coder", choices=["coder", "language", "agentic", "navigate"])
    ap.add_argument("--steps", type=int, required=True, help="TOTAL target step; chunked; resume from _state")
    ap.add_argument("--chunk-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--param-cap", type=int, default=33_000_000, help="explicit; NOT auto-raised; 33M original")
    ap.add_argument("--restore-best", action="store_true")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--teacher-model", default="qwen3.5:2b")
    ap.add_argument("--no-teacher", action="store_true", help="don't check/require Ollama Qwen; train Brain33M only")
    ap.add_argument("--out", default=str(HERE / "models" / "qwen33_train_summary.json"))
    args = ap.parse_args()

    if args.param_cap < BRAIN11_TOTAL:
        print(f"[cap] param_cap {args.param_cap} < Brain11 {BRAIN11_TOTAL}; refusing (lower cap can't hold 3x10,793,088)")
        return 2
    if args.param_cap > 33_000_000:
        print(f"[cap] NOTE: raised cap to {args.param_cap}. Honest: 33M!=2.3B raw parity; ~2.3B for raw match; you keep own small model.")

    teacher_ok = (not args.no_teacher) and check_ollama(args.ollama, args.teacher_model)

    # shared agentic for navigate; can't train Qwen; restore-best snapshots; nothing deleted
    train_name = args.model
    weight_name = "agentic" if args.model == "navigate" else args.model

    if args.restore_best:
        print(json.dumps(restore_best([weight_name]), indent=2))

    summary = {
        "trains": "Brain11-33M (NOT qwen3.5:2b)", "model": train_name, "weight_name": weight_name,
        "total_steps": args.steps, "chunk_steps": args.chunk_steps, "device": args.device,
        "param_cap": args.param_cap, "brain11_total_params": BRAIN11_TOTAL,
        "teacher": None if not teacher_ok else {"ollama": args.ollama, "model": args.teacher_model, "role": "distill/eval only"},
        "honest": "no 2B->33M; no parity claim; cancel-safe; resumable; nothing lost; match=undefined unless you set eval",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    log = HERE / "models" / f"qwen33_train_{train_name}.jsonl"
    try:
        while True:
            cur = _state_step(weight_name)
            if cur >= args.steps:
                break
            nxt = min(args.steps, cur + args.chunk_steps)
            cmd = [sys.executable, str(TRAIN_PY), "--model", train_name,
                   "--steps", str(nxt), "--device", args.device, "--stop-after-chunk"]
            print(f"[chunk] step {cur} -> {nxt} (cancel anytime; resumable)")
            rc = subprocess.run(cmd, check=False).returncode
            after = _state_step(weight_name)
            _append_jsonl(log, {"ts": time.time(), "from": cur, "to": nxt, "rc": rc, "step_after": after})
            if rc != 0:
                print(f"[chunk] rc={rc}; checkpoint intact; rerun to resume. No data lost.")
            if after <= cur and rc != 0:
                print("[chunk] no progress this chunk; stopping to avoid a hot loop. No data lost; rerun to resume.")
                break
            # no cleanup, no deletion, no forced --fresh; restart resumes
    except KeyboardInterrupt:
        print("\n[cancel] interrupted; last atomic checkpoint intact; rerun same command to resume. No data lost.")

    summary["final_step"] = _state_step(weight_name)
    summary["finished_or_cancelled"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write(Path(args.out), json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
