#!/usr/bin/env python3
"""Brain11 Router — one edge-friendly entry point for the three 10.8M models.

    coder     -> code synthesis / formatting / structured output
    language  -> English + Urdu conversation
    agentic   -> task/status -> tool-call decision

The router is deliberately lazy: on an edge device only the model needed for
the current task is loaded. The coding path is meant to be used through
brain11_loop.py (retrieval + verifier + repair), not as a raw oracle.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

from brain11_model import Brain, Brain11Config, load_model
from brain11_tokenizer import decode, encode

MODEL_FILES = {
    "coder": "brain11_coder_weights.pt",
    "language": "brain11_language_weights.pt",
    "agentic": "brain11_agentic_weights.pt",
}

CODING_WORDS = (
    "code", "function", "def ", "class ", "python", "javascript", "script",
    "implement", "program", "algorithm", "debug", "syntax", "json", "yaml",
    "csv", "format", "regex", "api", "sql", "html", "css", "unit test",
)
AGENTIC_WORDS = (
    "task_type:", "status:", "next action", "agent", "tool call", "tool:",
    "predict", "plan", "workflow", "check_result", "ask_user", "call_tool",
)
URDU_RE = re.compile(r"[\u0600-\u06FF]")


def classify_task(text: str) -> str:
    low = text.lower()
    if any(w in low for w in AGENTIC_WORDS):
        return "agentic"
    if any(w in low for w in CODING_WORDS):
        return "coder"
    return "language"


class Model:
    def __init__(self, name: str, path: Path, device: str):
        self.name = name
        self.device = device
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path.name}; train it with: python3 brain11_train.py --model {name}"
            )
        self.model: Brain = load_model(path, device)
        self.model.eval()

    def generate(self, prompt: str, n: int = 128, temperature: float = 0.7,
                 top_k: int = 40, repetition_penalty: float = 1.1) -> str:
        ids = encode(prompt, add_bos=True, add_eos=False)
        out = self.model.generate(ids, n=n, temperature=temperature, top_k=top_k,
                                  repetition_penalty=repetition_penalty)
        return decode(out)

    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        return self.model.score_choices(prompt, choices)


class Brain11Router:
    def __init__(self, device: str | None = None, lazy: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lazy = lazy
        self._models: dict[str, Model] = {}
        self._stats = {"coder": 0, "language": 0, "agentic": 0}
        if not lazy:
            for name in MODEL_FILES:
                self._get(name)

    def _get(self, name: str) -> Model:
        if name not in self._models:
            self._models[name] = Model(name, HERE / MODEL_FILES[name], self.device)
            print(f"[brain11] loaded {name} ({self._models[name].model.n_params():,} params)",
                  flush=True)
        return self._models[name]

    def get_model(self, name: str) -> Model:
        """Public accessor used by the Cortex loop."""
        return self._get(name)

    # ---------------------------------------------------------------- language
    def chat(self, message: str, n: int = 120, temperature: float = 0.6) -> dict:
        urdu = bool(URDU_RE.search(message))
        if urdu:
            prompt = f"### صارف: {message}\n### معاون:"
        else:
            prompt = f"### User: {message}\n### Assistant:"
        stops = ("### User:", "### Assistant:", "### صارف:", "### معاون:",
                 "### Story:", "### Text:")
        text = self._get("language").generate(prompt, n=n, temperature=temperature)
        text = _cut(text, stops).strip()
        self._stats["language"] += 1
        return {"task_type": "language", "model": "brain11-language", "text": text,
                "language": "ur" if urdu else "en"}

    # ---------------------------------------------------------------- coding
    def code(self, task: str, n: int = 160, temperature: float = 0.2) -> dict:
        prompt = f"### Task:\n{task.strip()}\n### Code:\n"
        text = self._get("coder").generate(prompt, n=n, temperature=temperature)
        text = _cut(text, ("### Tests:", "### Task:", "### File:")).strip()
        self._stats["coder"] += 1
        return {"task_type": "coder", "model": "brain11-coder", "text": text}

    # ---------------------------------------------------------------- agentic
    def agentic(self, task: str, status: str | None = None,
                context: dict | None = None, score: bool = True) -> dict:
        spec = _infer_tool(task)
        ctx = dict(context or {})
        status = (status or _infer_status(task)).lower()
        tool = spec["tool"] if spec else None
        arg = spec["arg"] if spec else None
        if arg and arg not in ctx:
            value = _extract_arg(task, arg)
            if value:
                ctx[arg] = value

        if status == "blocked" or (status == "new" and arg and arg not in ctx):
            decision = {"action": "ask_user", "tool": None,
                        "args": {"question": f"Please provide the {arg or 'required information'}."}}
            source = "policy"
        elif status == "done":
            decision = {"action": "finish", "tool": None,
                        "args": {"summary": f"{tool or 'task'} completed."}}
            source = "policy"
        elif status == "in_progress":
            decision = {"action": "check_result", "tool": tool,
                        "args": {"request_id": "last"}}
            source = "policy"
        elif tool and arg in ctx:
            decision = {"action": "call_tool", "tool": tool, "args": ctx}
            source = "policy"
        else:
            decision = {"action": "ask_user", "tool": None,
                        "args": {"question": "What would you like me to do?"}}
            source = "policy"

        prompt = (
            "### Agent Task\n"
            f"task_type: {spec['task_type'] if spec else 'unknown'}\n"
            f"status: {status}\n"
            f"context: {json.dumps(ctx, ensure_ascii=False)}\n"
            "### Decision\n"
        )
        choices = _decision_choices(spec, status, ctx)
        scores = None
        if score:
            try:
                scores = self._get("agentic").score_choices(prompt, choices)
                best = choices[max(range(len(choices)), key=lambda i: scores[i])]
                # policy guardrails keep the output valid even if the tiny scorer is unsure
                if status in {"blocked", "done", "in_progress"}:
                    best = json.dumps(decision, ensure_ascii=False, separators=(", ", ": "))
                decision = json.loads(best)
                source = "neural+policy"
            except Exception as exc:
                print(f"[brain11] agentic scorer unavailable ({exc}); using policy", flush=True)
        self._stats["agentic"] += 1
        return {"task_type": "agentic", "model": "brain11-agentic",
                "decision": decision, "scores": scores, "source": source}

    def stats(self) -> dict:
        return dict(self._stats)


# ------------------------------------------------------------------ helpers

def _cut(text: str, stops: tuple[str, ...]) -> str:
    for stop in stops:
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text.strip()


TOOL_SPECS = {
    "weather": {"task_type": "weather", "tool": "get_weather", "arg": "city",
                "words": ("weather", "forecast", "temperature", "rain")},
    "time": {"task_type": "time", "tool": "get_time", "arg": "timezone",
             "words": ("time", "timezone", "clock")},
    "calculation": {"task_type": "calculation", "tool": "calculate", "arg": "expression",
                    "words": ("calculate", "compute", "solve", "what is", "math")},
    "search": {"task_type": "search", "tool": "web_search", "arg": "query",
               "words": ("search", "look up", "find information", "web")},
    "file": {"task_type": "file", "tool": "read_file", "arg": "path",
             "words": ("read file", "open file", "contents of", "show the file")},
    "message": {"task_type": "message", "tool": "send_message", "arg": "recipient",
                "words": ("send a message", "message", "tell ", "notify")},
    "translation": {"task_type": "translation", "tool": "translate", "arg": "text",
                    "words": ("translate", "how do i say")},
    "booking": {"task_type": "booking", "tool": "book_flight", "arg": "date",
                "words": ("book a flight", "reserve", "flight to")},
}


def _infer_tool(task: str) -> dict | None:
    low = task.lower()
    best, best_score = None, 0
    for spec in TOOL_SPECS.values():
        score = sum(1 for w in spec["words"] if w in low)
        if score > best_score:
            best, best_score = spec, score
    return best


def _infer_status(task: str) -> str:
    m = re.search(r"status\s*:\s*(new|in_progress|blocked|done|failed)", task, re.I)
    return m.group(1).lower() if m else "new"


def _extract_arg(task: str, arg: str) -> str | None:
    q = re.search(r"[\"“](.+?)[\"”]", task)
    if arg == "city":
        m = re.search(r"(?:in|for|at)\s+([A-Z][A-Za-z .-]{1,30}?)(?:\?|\.|,|$)", task)
        return m.group(1).strip() if m else None
    if arg == "timezone":
        m = re.search(r"(UTC|[A-Z][a-z]+/[A-Z][a-z_]+)", task)
        return m.group(1) if m else "UTC"
    if arg == "expression":
        m = re.search(r"[-+*/().\d\s]{3,}", task)
        return m.group(0).strip() if m else None
    if arg == "query":
        return q.group(1) if q else task
    if arg == "path":
        m = re.search(r"[\w./-]+\.(?:txt|json|md|csv|py|yaml|yml|log)", task)
        return m.group(0) if m else None
    if arg == "recipient":
        m = re.search(r"(?:to|tell|message|notify)\s+([A-Z][a-z]+)", task)
        return m.group(1) if m else None
    if arg == "text":
        return q.group(1) if q else task
    if arg == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}|tomorrow|today|next \w+|next week", task, re.I)
        return m.group(0) if m else None
    return q.group(1) if q else None


def _decision_choices(spec: dict | None, status: str, ctx: dict) -> list[str]:
    if status == "blocked":
        base = [{"action": "ask_user", "tool": None, "args": {"question": "Please provide the missing information."}},
                {"action": "call_tool", "tool": spec["tool"] if spec else None, "args": ctx}]
    elif status == "done":
        base = [{"action": "finish", "tool": None, "args": {"summary": "Task completed."}},
                {"action": "call_tool", "tool": spec["tool"] if spec else None, "args": ctx}]
    elif status == "in_progress":
        base = [{"action": "check_result", "tool": spec["tool"] if spec else None, "args": {"request_id": "last"}},
                {"action": "call_tool", "tool": spec["tool"] if spec else None, "args": ctx}]
    else:
        base = [{"action": "call_tool", "tool": spec["tool"] if spec else None, "args": ctx},
                {"action": "ask_user", "tool": None, "args": {"question": "Please provide the required information."}}]
    return [json.dumps(x, ensure_ascii=False, separators=(", ", ": ")) for x in base]


def main():
    ap = argparse.ArgumentParser(description="Brain11 unified router")
    ap.add_argument("--chat")
    ap.add_argument("--task")
    ap.add_argument("--status")
    ap.add_argument("--context")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    router = Brain11Router(device=args.device)
    if args.chat:
        print(json.dumps(router.chat(args.chat, n=args.n), ensure_ascii=False, indent=2))
    elif args.task:
        kind = classify_task(args.task)
        if kind == "coder":
            print(json.dumps(router.code(args.task, n=args.n), ensure_ascii=False, indent=2))
        elif kind == "agentic":
            ctx = json.loads(args.context) if args.context else {}
            print(json.dumps(router.agentic(args.task, args.status, ctx), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(router.chat(args.task, n=args.n), ensure_ascii=False, indent=2))
    elif args.stats:
        print(json.dumps(router.stats(), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
