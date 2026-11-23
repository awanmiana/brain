#!/usr/bin/env python3
"""Generate a synthetic tool-calling and formatting corpus.

Writes `corpus_tools.txt`: JSON tool calls, API-style tool invocations,
YAML/XML tool calls, and formatting notes for the 1M toolcalling variant.
"""

import json
import random
import sys
from pathlib import Path

FILES = ["src/main.py", "notes/plan.txt", "data/rows.csv"]
TOOLS = ["read_file", "write_file", "search", "run_command"]


def main(out="corpus_tools.txt", n=400):
    rng = random.Random(20261131)
    chunks = []
    for _ in range(n):
        kind = rng.randrange(5)
        if kind == 0:
            chunks.append(json.dumps({"tool": rng.choice(TOOLS), "args": {"path": rng.choice(FILES)}}) + "\n")
        elif kind == 1:
            chunks.append(json.dumps({"action": "call_tool", "tool": rng.choice(TOOLS),
                                      "params": {"query": "TODO fix", "path": rng.choice(FILES)}}) + "\n")
        elif kind == 2:
            chunks.append('<tool name="search"><arg name="query">logout bug</arg></tool>\n')
        elif kind == 3:
            chunks.append('Format: one JSON object per line — {"tool": "...", "args": {...}}. '
                          'Never add prose outside the JSON.\n')
        else:
            chunks.append("tool: write_file\nargs:\n  path: notes/plan.txt\n  content: |\n    step one\n    step two\n\n")
    text = "\n".join(chunks)
    Path(out).write_text(text)
    print(f"wrote {out}: {len(text)} chars")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
