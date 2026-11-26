#!/usr/bin/env python3
"""Collect reactive-hop training data for the `navigate` mode (variant B).

Runs brain11_reactive.reactive_build over a task set (optionally WITH the real
10.8M model as the proposer) and writes JSONL:

    input  = prompt + code-status + done + pending + history
    output = typed op + target_section + pending_before->after + test_delta

Only ACCEPTED hops are positive examples; rejected/rolled-back hops are kept as
negatives. Everything is re-verified; nothing untested enters the pool.

    python3 scripts/collect_hops.py --out brain11_navigate.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from brain11_loop import curriculum  # noqa: E402
from brain11_reactive import reactive_build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "brain11_navigate.jsonl"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--use-model", action="store_true", help="use trained coder as proposer")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = None
    if args.use_model:
        from brain11_router import Brain11Router
        model = Brain11Router(device=args.device).get_model("coder")

    pos = neg = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (task, tests) in enumerate(curriculum()[:args.limit]):
            st = reactive_build(task, tests=tests, model=model, max_rounds=4)
            for h in st.hops:
                rec = {
                    "input": {
                        "prompt": task,
                        "section_kind": "function",
                        "done": st.done,
                        "pending": h.pending_before,
                        "history_len": h.round,
                    },
                    "output": {
                        "op": h.action,
                        "detail": h.detail,
                        "pending_after": h.pending_after,
                        "test_passing": not h.pending_after or h.pending_after == [],
                    },
                    "accepted": h.action != "stop",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # rejected hops recorded on the state object are kept as negatives
            for rej in st.rejected:
                f.write(json.dumps({"input": {"prompt": task},
                                    "output": rej, "accepted": False},
                                   ensure_ascii=False) + "\n")
                neg += 1
            if any(h.action in ("forward", "backward") for h in st.hops):
                pos += 1

    print(f"wrote {args.out}: {pos} solved tasks, {neg} rejected-negative hops")
    print("next (cap-compliant): continue training brain11_agentic in a `navigate` mode "
          "on this + agentic data; verify with the deterministic checker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
