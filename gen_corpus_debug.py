#!/usr/bin/env python3
"""Generate a synthetic debugging/troubleshooting corpus.

Writes `corpus_debug.txt`: error messages, tracebacks, debugging steps,
fixes, and troubleshooting notes for the 1M debugging variant.
"""

import random
import sys
from pathlib import Path

VARS = ["items", "result", "total", "count", "rows", "name", "value", "ids"]
FILES = ["data.csv", "config.yml", "script.py", "log.txt", "db.sqlite"] + [
    "log/{}.log".format(d) for d in ["app", "db", "worker", "cron"]
]
FUNCS = ["run_job", "process", "fetch_data", "parse_input", "main"]
STEPS = [
    "reproduce the failure in a minimal script",
    "check the traceback line",
    "inspect the variables with a print or pdb breakpoint",
    "verify the input data for unexpected values",
    "compare with the expected output from the spec",
    "check the log file for earlier errors",
    "restart the worker and retry the job",
]


def main(out="corpus_debug.txt", n=450):
    rng = random.Random(20261129)
    chunks = []
    for _ in range(n):
        kind = rng.randrange(7)
        v = rng.choice(VARS)
        f = rng.choice(FILES)
        fn = rng.choice(FUNCS)
        if kind == 0:
            chunks.append(rng.choice(
                ["AttributeError", "TypeError", "ValueError", "KeyError", "IndexError"]
            ) + ": '" + v + "' object has no attribute '" +
                rng.choice(["size", "count", "total"]) + "'\n")
        elif kind == 1:
            chunks.append("Traceback (most recent call last):\n  File \"{}\", line {}, in {}\n    {}\n".format(
                f, rng.randrange(1, 90), fn,
                rng.choice(["total = sum(items)", "{} = parse_input()".format(v), "data = read(f)"])))
        elif kind == 2:
            steps = rng.sample(STEPS, rng.randrange(2, 5))
            chunks.append("Debugging steps:\n" + "\n".join(
                "{}. {}.".format(i + 1, s.capitalize()) for i, s in enumerate(steps)) + "\n")
        elif kind == 3:
            chunks.append("Fix: cast the value before use — x = int(value) or validate the input.\n")
        elif kind == 4:
            chunks.append("Troubleshooting: if the script exits silently, check the log file — errors logged before the traceback are not shown on stdout. If the output is empty, verify the data was read: print the row count after loading.\n")
        elif kind == 5:
            chunks.append("Error when running {}: check that the file exists and is readable.\n".format(f))
        else:
            chunks.append("When {} fails, look at the last traceback line first, then check the input file for the missing or unexpected value.\n".format(fn))
    text = "\n".join(chunks)
    Path(out).write_text(text)
    print(f"wrote {out}: {len(text)} chars")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
