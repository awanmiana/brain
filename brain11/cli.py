#!/usr/bin/env python3
"""Brain11 — THE single entry point.

    brain11 <command> [args...]          (after `pip install -e .`)
    python3 -m brain11 <command> [...]   (from the repo root, no install)

Commands
  prepare        build corpora + tokenizer            -> `python3 brain11_prepare.py`
  train          chunked atomic resumable training    -> `python3 brain11_train.py`
  serve          OpenAI-compatible inference API      -> `python3 brain_http_server.py`
  doctor         health checks (+--fix)               -> `python3 brain11_loop.py --doctor`
  eval           held-out eval (+--no-memory)         -> `python3 brain11_loop.py --eval`
  learn          self-learning from verified wins      -> `python3 brain11_loop.py --learn N`
  selflearn      alias of learn
  solve          one task + tests (+self-heal/repair) -> `python3 brain11_loop.py --solve`
  reactive       reactive section-graph construction  -> `python3 brain11_reactive.py`
  navigate       alias of reactive
  plan           prerequisite-first planner           -> `python3 brain11_planner.py`
  route          unified router CLI                   -> `python3 brain11_router.py`
  selftest       run all self-tests
  models         show params / model files
  infer          one-shot inference (embedded + CLI)
  chat           one-shot language inference
  code           one-shot coding inference
  agentic        one-shot tool-call inference
  opencode       print OpenCode provider config
  publish-hf     push MODEL artifacts to Hugging Face (never pushed to git)
  version
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# make top-level brain11_* modules importable regardless of install mode
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _dispatch(module_name: str, prog: str, rest: list[str]) -> int:
    mod = importlib.import_module(module_name)
    argv0 = sys.argv[0]
    saved = list(sys.argv)
    sys.argv = [prog] + list(rest)
    try:
        result = mod.main()
    finally:
        sys.argv = saved
    return 0 if result is None else int(result)


# ---------------------------------------------------------------- commands

def cmd_selftest(rest: list[str]) -> int:
    ok = True
    for prog, mod in (
        ("brain11-loop", "brain11_loop"),
        ("brain11-planner", "brain11_planner"),
        ("brain11-reactive", "brain11_reactive"),
    ):
        rc = _dispatch(mod, prog, ["--selftest"])
        ok = ok and (rc == 0)
    return 0 if ok else 1


def cmd_train(rest: list[str]) -> int:
    """Accept `brain11 train coder --steps N` OR `brain11 train --model coder ...`."""
    rest = list(rest)
    if rest and not rest[0].startswith("-") and rest[0] not in ("all", "coder", "language", "agentic"):
        print(f"warning: unknown model '{rest[0]}'", file=sys.stderr)
    if rest and not rest[0].startswith("-"):
        rest = ["--model", rest[0]] + rest[1:]
    return _dispatch("brain11_train", "brain11 train", rest)


def cmd_models(rest: list[str]) -> int:
    from brain11_model import Brain11Config
    from brain11_router import MODEL_FILES
    cfg = Brain11Config()
    n = cfg.n_params()
    out = {
        "name": "brain",
        "n_params_per_model": n,
        "total_params": 3 * n,
        "param_cap": 33_000_000,
        "config": cfg.__dict__,
        "models": {k: (str(HERE / v), (HERE / v).exists()) for k, v in MODEL_FILES.items()},
        "tokenizer": str(HERE / "brain11_tokenizer.json"),
        "model_dir": str(_model_dir()),
        "source_dir": str(HERE),
    }
    print(json.dumps(out, indent=2))
    return 0


def _model_dir() -> Path:
    import os
    env = os.environ.get("BRAIN11_MODEL_DIR")
    return Path(env) if env else (HERE / "models")


def cmd_infer(rest: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="brain11 infer")
    ap.add_argument("text")
    ap.add_argument("--kind", choices=["chat", "code", "agentic", "auto"], default="auto")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(rest)
    from brain11_router import Brain11Router, classify_task
    r = Brain11Router(device=args.device)
    kind = classify_task(args.text) if args.kind == "auto" else args.kind
    if kind == "coder":
        kind = "code"
    if kind == "code":
        out = r.code(args.text, n=args.n)
        text = out["text"]
        # honor the name the task asks for even in raw (non-verified) inference
        try:
            from brain11_loop import Healer
            text = Healer().normalize(text, args.text)
        except Exception:
            pass
    elif kind == "agentic":
        out = r.agentic(args.text)
        text = json.dumps(out["decision"], ensure_ascii=False, indent=2)
    else:
        out = r.chat(args.text, n=args.n)
        text = out["text"]
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


def cmd_opencode(rest: list[str]) -> int:
    snippet = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "brain": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Brain (local 33M edge)",
                "options": {"baseURL": "http://127.0.0.1:8765/v1", "apiKey": "not-needed"},
                "models": {
                    "brain": {"name": "Brain (3 x 10.8M)"},
                    "brain11-coder": {"name": "Brain Coder (10.8M)"},
                    "brain11-language": {"name": "Brain Language (10.8M, EN+UR)"},
                    "brain11-agentic": {"name": "Brain Agentic (10.8M)"},
                },
            }
        },
        "model": "brain/brain",
    }
    print(json.dumps(snippet, indent=2))
    print("\n# 1) start the API:   brain11 serve", file=sys.stderr)
    print("# 2) paste into ~/.config/opencode/opencode.jsonc", file=sys.stderr)
    print("# 3) select model 'brain' (or brain11-coder) as the coding agent", file=sys.stderr)
    return 0


def cmd_publish_hf(rest: list[str]) -> int:
    return _dispatch("scripts.publish_hf", "brain11 publish-hf", rest)


def cmd_learn(rest: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="brain11 learn")
    ap.add_argument("n", nargs="?", type=int, default=24)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--device", default=None)
    known, passthrough = ap.parse_known_args(rest)
    extra = ["--learn", str(known.n), "--n", str(known.n_samples),
             "--rounds", str(known.rounds)]
    if known.no_memory:
        extra.append("--no-memory")
    if known.device:
        extra += ["--device", known.device]
    return _dispatch("brain11_loop", "brain11 learn", extra + passthrough)


def cmd_solve(rest: list[str]) -> int:
    if not rest:
        print("usage: brain11 solve TASK [--tests 'assert ...'] [--n N] [--rounds N]",
              file=sys.stderr)
        return 2
    task = rest[0]
    return _dispatch("brain11_loop", "brain11 solve", ["--solve", task] + rest[1:])


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]

    if cmd in ("version", "--version"):
        from brain11 import __version__
        print(__version__)
        return 0

    # direct dispatch to the underlying module CLIs (passthrough args)
    dispatch = {
        "prepare": ("brain11_prepare", "brain11 prepare"),
        "serve": ("brain_http_server", "brain11 serve"),
        "route": ("brain11_router", "brain11 route"),
        "reactive": ("brain11_reactive", "brain11 reactive"),
        "navigate": ("brain11_reactive", "brain11 navigate"),
        "plan": ("brain11_planner", "brain11 plan"),
        "eval": ("brain11_loop", "brain11 eval"),
        "doctor": ("brain11_loop", "brain11 doctor"),
    }
    if cmd == "eval":
        rest = ["--eval"] + rest
    elif cmd == "doctor":
        rest = ["--doctor"] + rest
    if cmd in dispatch:
        mod, prog = dispatch[cmd]
        return _dispatch(mod, prog, rest)

    if cmd == "train":
        return cmd_train(rest)
    if cmd in ("selftest", "test"):
        return cmd_selftest(rest)
    if cmd == "models":
        return cmd_models(rest)
    if cmd == "infer":
        return cmd_infer(rest)
    if cmd in ("chat",):
        return cmd_infer(["--kind", "chat"] + rest)
    if cmd in ("code",):
        return cmd_infer(["--kind", "code"] + rest)
    if cmd in ("agentic",):
        return cmd_infer(["--kind", "agentic"] + rest)
    if cmd == "opencode":
        return cmd_opencode(rest)
    if cmd == "publish-hf":
        return cmd_publish_hf(rest)
    if cmd in ("learn", "selflearn"):
        return cmd_learn(rest)
    if cmd == "solve":
        return cmd_solve(rest)

    print(f"unknown command: {cmd}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
