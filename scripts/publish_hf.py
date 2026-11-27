#!/usr/bin/env python3
"""Publish MODEL artifacts to Hugging Face. NOTHING here goes to git.

Source code -> GitHub.  Model artifacts -> Hugging Face.  They are separate.

    pip install huggingface_hub
    huggingface-cli login            # or set HF_TOKEN
    brain11 publish-hf --repo-id YOUR_NAME/brain11

Uploads MODEL artifacts only (never source code, never optimizer/training state):
    brain11_tokenizer.json
    brain11_<name>_weights.pt, _best.pt
    brain11_<name>_train.npy / _val.npy / *.jsonl   (optional, with --include-data; NOT recommended for "model only")
Stages a clean folder so we never upload source or .git. Writes a model card.
Does NOT touch git. Never run `git add models/`. Source code -> GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# MODEL artifacts only. NOT source code. NOT optimizer/training state. NOT raw data.
ARTIFACT_GLOBS = [
    "brain11_tokenizer.json",
    "brain11_*_weights.pt", "brain11_*_best.pt",
]

CARD = """---
license: apache-2.0
tags: [edge-ai, small-language-model, code-generation, urdu, self-learning]
library_name: pytorch
pipeline_tag: text-generation
---

# Brain11

One local edge AI system. **3 independent ~10,793,088-param models = 32,379,264 total (cap 33M).**
Shared 4,096-token byte-level BPE. ~43.2 MB fp32 each. Runs on CPU or RTX 3050.

| Model | Role |
|---|---|
| `brain11_coder_weights.pt` | code + structured output + tool-calling assist |
| `brain11_language_weights.pt` | English + Urdu communication |
| `brain11_agentic_weights.pt` | task/status -> next-action JSON |

Load with `brain11` (source on GitHub):

```python
from brain11 import Brain11Router
r = Brain11Router(device="cuda")
r.code("write a function add(a, b) that returns a + b")
```

## Limits

10.8M per model. Narrow + verified tasks reliable; novel open-ended coding is not.
No open-domain parity with 8B models. Nothing here exceeds the 33M cap.
Source code is NOT in this repo; it is published separately on GitHub.
"""


def collect(roots: list[Path], include_data: bool) -> list[Path]:
    files: list[Path] = []
    pats = list(ARTIFACT_GLOBS)
    if include_data:
        pats += ["brain11_*_train.npy", "brain11_*_val.npy", "brain11_*.jsonl"]
    for root in roots:
        if not root.exists():
            continue
        for pat in pats:
            files.extend(root.glob(pat))
    return sorted(set(f for f in files if f.is_file()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Push MODEL artifacts to Hugging Face (never git).")
    ap.add_argument("--repo-id", required=True, help="e.g. your-name/brain11")
    ap.add_argument("--model-dir", default=str(HERE / "models"))
    ap.add_argument("--include-data", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would upload")
    args = ap.parse_args(argv)

    roots = [Path(args.model_dir), HERE]
    files = collect(roots, args.include_data)
    if not files:
        print("no model artifacts found. Train first, then publish.", file=sys.stderr)
        return 1

    print(f"found {len(files)} artifact(s):")
    for f in files:
        print(f"  {f.relative_to(HERE) if f.is_relative_to(HERE) else f} "
              f"({f.stat().st_size / 1e6:.2f} MB)")

    if args.dry_run:
        print("dry-run: nothing uploaded.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # stage into a clean folder so we never upload source or .git
    stage = HERE / ".hf_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, stage / f.name)
    (stage / "README.md").write_text(CARD, encoding="utf-8")

    # clean repo: drop remote files that are not part of this model upload
    keep = {f.name for f in stage.iterdir() if f.is_file()} | {"README.md", ".gitattributes"}
    for remote in api.list_repo_files(args.repo_id, repo_type="model"):
        if remote not in keep:
            api.delete_file(path_in_repo=remote, repo_id=args.repo_id, repo_type="model",
                            commit_message="remove useless/corrupt artifact before clean model upload")
            print(f"  removed remote: {remote}")

    api.upload_folder(folder_path=str(stage), repo_id=args.repo_id, repo_type="model",
                      commit_message="clean model upload: tokenizer + 3x ~10.8M weights (32.4M total)")
    shutil.rmtree(stage)
    print(f"uploaded to https://huggingface.co/{args.repo_id}")
    print("source code is NOT uploaded here; push source to GitHub separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
