#!/usr/bin/env python3
"""Collect verified reactive-hop training data for variant-B `navigate` mode.

Runs `brain11_reactive.reactive_build` over a verified curriculum (optionally WITH
the real 10.8M model as proposer) and writes JSONL `{"text": ..., ...}`:

    input  = prompt + code-status + done + pending + history
    output = typed op + target_section + pending_before->after + test_delta

Only hops in `state.hops` (accepted, re-verified) are POSITIVE SFT examples.
Rejected/rolled-back hops in `state.rejected` are kept as NEGATIVES
(`accepted=false`, `negative=true`); `brain11_train.load_docs` does NOT train them
as positive actions. Nothing untested enters the pool. Nothing is deleted.

    python3 scripts/collect_hops.py --out brain11_navigate.jsonl --limit 50

Writes `brain11_navigate.jsonl`. By default also writes a NEW combined corpus
`brain11_agentic_navigate.jsonl` = original `brain11_agentic.jsonl` + verified
navigate positives (+ stored negatives for audit; not SFT-trained). Original
`brain11_agentic.jsonl` is read-only. No training data is lost.

Then (cap-compliant; shared 11M `agentic` model; NOT a 4th model):

    python3 brain11_train.py --model navigate --steps <new-total> --device cuda
    # == python3 brain11_train.py --model agentic --mode navigate --steps <new-total> ...
    python3 brain11_train.py --model agentic --restore-best        # restore shared agentic first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from brain11_loop import curriculum  # noqa: E402
from brain11_reactive import reactive_build, syntax_ok  # noqa: E402

MODE = "navigate"


def _section_kind(task: str) -> str:
    t = (task or "").lower()
    if "def " in t or "function" in t or "class " in t:
        return "function"
    return "section"


def _target_section(h) -> str:
    if h.pending_before:
        return h.pending_before[0]
    if h.pending_after:
        return h.pending_after[0]
    return "whole"


def _test_delta(h) -> dict:
    before = len(h.pending_before)
    after = len(h.pending_after)
    return {
        "pending_before": before,
        "pending_after": after,
        "delta": after - before,                 # negative = progress; 0 = no change
        "passing_before": before == 0,
        "passing_after": after == 0,
    }


def _navigate_text(inp: dict, out: dict, accepted: bool) -> str:
    """Deterministic `### Navigate` text target for next-token SFT (`### Mode: navigate`)."""
    hist = inp.get("history") or []
    return (
        "### Mode:\nnavigate\n"
        f"### Navigate Task:\n{inp.get('prompt','')}\n"
        f"### Code Status:\n{json.dumps(inp.get('code_status', {}), ensure_ascii=False, sort_keys=True)}\n"
        f"### Done:\n{json.dumps(inp.get('done', []), ensure_ascii=False, sort_keys=True)}\n"
        f"### Pending:\n{json.dumps(inp.get('pending', []), ensure_ascii=False, sort_keys=True)}\n"
        f"### History:\n{json.dumps(hist, ensure_ascii=False, sort_keys=True)}\n"
        "### Action:\n" + json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n"
        f"### Accepted:\n{str(accepted).lower()}\n"
    )


def _history(st, h=None) -> list[dict]:
    hist = []
    for p in st.hops:
        hist.append({
            "round": p.round, "op": p.action, "detail": p.detail,
            "pending_before": p.pending_before, "pending_after": p.pending_after,
        })
    return hist


def _input(task: str, st, h=None) -> dict:
    pending = (h.pending_before if h is not None else st.pending)
    return {
        "prompt": task,
        "section_kind": _section_kind(task),
        "code_status": {
            "output_len": len(st.output or ""),
            "syntax_ok": syntax_ok(st.output or ""),
            "emitted_sections": list(st.emitted_sections),
            "done_count": len(st.done),
            "pending_count": len(pending),
            "rejected_count": len(st.rejected),
        },
        "done": list(st.done),
        "pending": list(pending),
        "history_len": len(st.hops),
        "history": _history(st, h),
        "emitted_sections": list(st.emitted_sections),
        "section_kinds": ["function", "section"],
    }


def _output_action(action: str, detail: str, pending_before, pending_after) -> dict:
    h = type("H", (), {"pending_before": list(pending_before), "pending_after": list(pending_after)})()
    target = pending_before[0] if pending_before else (pending_after[0] if pending_after else "whole")
    return {
        "op": action,                            # forward | backward | verify | stop
        "target_section": target,
        "detail": detail,
        "pending_before": list(pending_before),
        "pending_after": list(pending_after),
        "test_passing": len(pending_after) == 0,
        "test_delta": _test_delta(h),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "brain11_navigate.jsonl"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--use-model", action="store_true", help="use trained coder as proposer")
    ap.add_argument("--device", default=None)
    ap.add_argument("--combined-out", default=str(HERE / "brain11_agentic_navigate.jsonl"))
    ap.add_argument("--agentic", default=str(HERE / "brain11_agentic.jsonl"))
    ap.add_argument("--no-combine-agentic", action="store_true", help="do not write combined agentic+navigate corpus")
    args = ap.parse_args()

    model = None
    if args.use_model:
        from brain11_router import Brain11Router
        model = Brain11Router(device=args.device).get_model("coder")

    out_path = Path(args.out)
    pos = neg = 0
    written = 0
    navigate_records: list[dict] = []

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (task, tests) in enumerate(curriculum()[:args.limit]):
            st = reactive_build(task, tests=tests, model=model, max_rounds=4)

            # accepted, re-verified hops -> positive examples (includes correct `stop`)
            for h in st.hops:
                inp = _input(task, st, h)
                out = _output_action(h.action, h.detail, h.pending_before, h.pending_after)
                rec = {"mode": MODE, "input": inp, "output": out, "accepted": True, "negative": False,
                       "text": _navigate_text(inp, out, True)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                navigate_records.append(rec)
                written += 1
            if any(h.action in ("forward", "backward") for h in st.hops):
                pos += 1

            # rejected/rolled-back hops -> NEGATIVES (not trained as positive targets)
            for rej in st.rejected:
                inp = _input(task, st, None)
                out = _output_action(rej.get("action", "?"), rej.get("detail", ""),
                                     rej.get("pending_before", []), rej.get("pending_after", []))
                out["reason"] = rej.get("reason", "")
                out["error"] = rej.get("error", "")
                rec = {"mode": MODE, "input": inp, "output": out, "accepted": False, "negative": True,
                       "text": _navigate_text(inp, out, False)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                navigate_records.append(rec)
                written += 1
                neg += 1

    print(f"wrote {out_path}: {written} records, {pos} solved tasks, {neg} rejected-negative hops")

    if not args.no_combine_agentic:
        combined_path = Path(args.combined_out)
        agentic_path = Path(args.agentic)
        # original agentic data is read-only; combined is a NEW file; nothing lost
        with open(combined_path, "w", encoding="utf-8") as out_f:
            if agentic_path.exists():
                with open(agentic_path, encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("accepted") is False or rec.get("negative") is True:
                            continue
                        rec.setdefault("mode", "agentic")
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for rec in navigate_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote combined {combined_path}: agentic + navigate (original agentic untouched)")

    print("next (cap-compliant; shared 11M agentic; NOT a 4th model): "
          "python3 brain11_train.py --model navigate --steps <new-total> --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
