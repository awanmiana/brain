#!/usr/bin/env python3
"""Brain11 Cortex — the self-learning / healing / testing loop for the 33M system.

The 430M Brain already had: memory -> propose -> verify -> repair -> remember.
At 10.8M per model that loop is not optional; it is what makes the system useful.

Loop:
    route -> retrieve verified analogy -> propose best-of-N
          -> sandboxed test -> deterministic heal -> error-driven repair
          -> remember only verified successes -> periodic chunked retraining

Self-learning is gated: a generated example enters the training pool only after
it passes real tests. The base corpora are never overwritten. The trainer reads
brain11_selflearn.jsonl on the next chunk, so the model learns from its own
verified wins without catastrophic forgetting.

Usage:
    python3 brain11_loop.py --selftest
    python3 brain11_loop.py --doctor
    python3 brain11_loop.py --doctor --fix
    python3 brain11_loop.py --solve "write add(a,b)" --tests "assert add(2,3)==5"
    python3 brain11_loop.py --learn 24
    python3 brain11_loop.py --eval --limit 12
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

from brain_cortex import BM25, Verifier, extract_code, requested_name
from brain11_model import Brain11Config, load_model
from brain11_planner import heal_code
from brain11_router import Brain11Router, classify_task
from brain11_tokenizer import TOKENIZER_PATH, decode, encode

CODE_CORPUS = HERE / "brain11_code.jsonl"
SELFLEARN = HERE / "brain11_selflearn.jsonl"
SELFLEARN_META = HERE / "brain11_selflearn_meta.jsonl"
REPAIRS = HERE / "brain11_selflearn_repairs.jsonl"
TRACES = HERE / "brain11_memory.json"


# ------------------------------------------------------------------ parsing

def parse_code_docs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            if not text.startswith("### Task:"):
                continue
            m = re.match(r"### Task:\n(.*?)\n### Code:\n(.*?)(?:\n### Tests:\n(.*))?$", text, re.S)
            if not m:
                continue
            task, code, tests = m.group(1).strip(), m.group(2).strip(), (m.group(3) or "").strip()
            if task and code:
                out.append({"task": task, "code": code, "tests": tests, "source": path.name})
    return out


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    last = None
    for i in range(12):  # Windows file-lock retry; no corrupt destination
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, OSError) as exc:
            last = exc
            time.sleep(0.05 * (i + 1))
    raise last


# ------------------------------------------------------------------ memory

class Memory11:
    """Verified code traces + task/code corpus, retrieved with BM25."""

    def __init__(self, max_docs: int = 8000, traces_path: Path = TRACES):
        self.traces_path = traces_path
        self.docs: list[dict] = []
        self.seen: set[str] = set()
        self.traces = []
        self._load_traces()
        for path in (SELFLEARN, CODE_CORPUS):
            for doc in parse_code_docs(path):
                self._add_doc(doc, rebuild=False)
        self.index = BM25(self.docs) if self.docs else None
        print(f"[memory] {len(self.docs)} code docs, {len(self.traces)} verified traces", flush=True)

    def _load_traces(self):
        if self.traces_path.exists():
            try:
                self.traces = json.loads(self.traces_path.read_text(encoding="utf-8"))
            except Exception:
                self.traces = []
        for t in self.traces:
            if t.get("task") and t.get("code"):
                self._add_doc({"task": t["task"], "code": t["code"],
                               "tests": t.get("tests", ""), "source": "trace"}, rebuild=False)

    def _add_doc(self, doc: dict, rebuild: bool = True):
        key = _hash(doc["task"], doc["code"])
        if key in self.seen:
            return False
        self.seen.add(key)
        doc = dict(doc)
        doc["key"] = key
        doc["text"] = f"### Task:\n{doc['task']}\n### Code:\n{doc['code']}"
        self.docs.append(doc)
        if rebuild:
            self.index = BM25(self.docs)
        return True

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if not self.index:
            return []
        return [doc for _, doc in self.index.top(query, k)]

    def add_trace(self, task: str, code: str, tests: str = "", source: str = "neural"):
        if not task or not code:
            return
        if not any(t.get("task") == task and t.get("code") == code for t in self.traces):
            self.traces.append({"task": task, "code": code, "tests": tests,
                                "source": source, "ts": time.time()})
            _atomic_write(self.traces_path, json.dumps(self.traces, indent=2, ensure_ascii=False))
        self._add_doc({"task": task, "code": code, "tests": tests, "source": "trace"})

    def memory_solve(self, task: str, tests: str, verifier: Verifier):
        if not tests:
            return None
        wanted = requested_name(task)
        for doc in self.retrieve(task, k=5):
            code = doc["code"]
            m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
            if wanted and m and m.group(1) != wanted:
                code = re.sub(rf"\b{re.escape(m.group(1))}\b", wanted, code)
            if verifier.verify(code, tests)["passed"]:
                return {"code": code, "source": f"memory:{doc['source']}",
                        "old_name": m.group(1) if m else None}
        return None


# ------------------------------------------------------------------ healer

class Healer:
    """Deterministic first aid before asking the neural model to try again."""

    def diagnose(self, error: str, code: str = "") -> str:
        e = (error or "").lower()
        if not code.strip():
            return "empty"
        if "forbidden" in e:
            return "safety"
        if "timeout" in e:
            return "timeout"
        if "indentation" in e or "expected an indented block" in e:
            return "indentation"
        if "syntax" in e or "invalid syntax" in e or "unexpected eof" in e:
            return "syntax"
        if "nameerror" in e or "not defined" in e:
            return "naming"
        if "typeerror" in e or "unsupported operand" in e:
            return "types"
        if "assertionerror" in e:
            return "logic"
        if "zerodivision" in e:
            return "logic"
        return "unknown"

    def normalize(self, code: str, task: str) -> str:
        """Rename the generated function to the name the task explicitly asks for.

        The task is the specification; if it says `multiply_v2`, a function
        called `multiply_vowels` does not satisfy it even when the body is
        correct. Only the identifier changes — logic is still test-verified.
        """
        fixed = self.auto_fix(code, task, "")
        return fixed["code"] if fixed else code

    def auto_fix(self, code: str, task: str, error: str) -> dict | None:
        """Small, safe repairs. Returns {'code', 'note'} or None."""
        if not code.strip():
            return None
        original = code
        code = extract_code(code)
        code = code.replace("### Code:", "").replace("### Tests:", "")
        code = textwrap.dedent(code).strip()
        notes = []

        # missing colon after a block opener
        lines = code.splitlines()
        changed = False
        for i, line in enumerate(lines):
            s = line.rstrip()
            if re.match(r"^\s*(def|class|if|elif|else|for|while|try|except|finally|with)\b", s):
                if not s.endswith(":") and i + 1 < len(lines) and lines[i + 1].startswith((" ", "\t")):
                    lines[i] = s + ":"
                    changed = True
        if changed:
            code = "\n".join(lines)
            notes.append("added missing colon")

        # function name mismatch
        wanted = requested_name(task)
        if wanted:
            defs = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", code)
            if defs and wanted not in defs:
                old = defs[0]
                code = re.sub(rf"\b{re.escape(old)}\b", wanted, code)
                notes.append(f"renamed {old}->{wanted}")

        # NameError: rename the undefined name to the requested function
        m = re.search(r"name '([^']+)' is not defined", error or "")
        if m and wanted and m.group(1) != wanted:
            code = re.sub(rf"\b{re.escape(m.group(1))}\b", wanted, code)
            notes.append(f"fixed undefined name {m.group(1)}")

        # strip obvious prose / markdown leftovers
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"```\s*$", "", code).strip()

        # a trailing incomplete line often causes unexpected EOF
        if "unexpected eof" in (error or "").lower():
            body = code.splitlines()
            while body:
                test = "\n".join(body)
                try:
                    ast.parse(test)
                    code = test
                    notes.append("dropped incomplete trailing line")
                    break
                except SyntaxError:
                    body.pop()

        if code == original or not code.strip():
            return None
        try:
            ast.parse(code)
        except SyntaxError:
            return None
        return {"code": code, "note": ", ".join(notes) or "normalized"}


# ------------------------------------------------------------------ generation

def build_prompt(task: str, docs: list[dict]) -> str:
    parts = []
    for doc in docs[:2]:
        parts.append(f"### Task:\n{doc['task']}\n### Code:\n{doc['code'].strip()}")
    parts.append(f"### Task:\n{task.strip()}\n### Code:\n")
    return "\n\n".join(parts)


def repair_prompt(task: str, code: str, error: str, docs: list[dict]) -> str:
    parts = []
    for doc in docs[:1]:
        parts.append(f"### Task:\n{doc['task']}\n### Code:\n{doc['code'].strip()}")
    parts.append(
        f"### Task:\n{task.strip()}\n"
        f"### Previous Code:\n{code.strip()}\n"
        f"### Error:\n{error.strip()}\n"
        f"### Corrected Code:\n"
    )
    return "\n\n".join(parts)


def extract_code11(text: str) -> str:
    for stop in ("### Tests:", "### Task:", "### File:", "### Previous Code:"):
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    text = text.replace("### Code:", "").replace("### Corrected Code:", "")
    return extract_code(text)


class Generator11:
    def __init__(self, router: Brain11Router, max_new: int = 160):
        self.model = router.get_model("coder")
        self.max_new = max_new

    def sample(self, prompt: str, n: int = 4, temperature: float = 0.8) -> list[str]:
        ids = encode(prompt, add_bos=True, add_eos=False)
        ctx = self.model.model.cfg.ctx
        reserve = min(self.max_new, ctx - 1)
        if len(ids) > ctx - reserve:
            ids = [ids[0]] + ids[-(ctx - reserve - 1):]
        out = []
        for _ in range(n):
            gen = self.model.model.generate(
                ids, n=self.max_new, temperature=temperature, top_k=40,
                repetition_penalty=1.05,
            )
            out.append(decode(gen))
        return out


# ------------------------------------------------------------------ cortex

def plan_for(task: str) -> dict:
    low = task.lower()
    if any(w in low for w in ("fix", "bug", "error", "debug", "fails", "repair")):
        return {"n": 6, "rounds": 3, "temperature": 0.8}
    if any(w in low for w in ("write", "implement", "create", "function", "def ", "returns")):
        return {"n": 6, "rounds": 2, "temperature": 0.75}
    return {"n": 3, "rounds": 1, "temperature": 0.6}


class Cortex11:
    def __init__(self, router: Brain11Router, memory: Memory11 | None = None,
                 verifier: Verifier | None = None, max_new: int = 160):
        self.router = router
        self.memory = memory or Memory11()
        self.verifier = verifier or Verifier()
        self.generator = Generator11(router, max_new=max_new)
        self.healer = Healer()

    def solve(self, task: str, tests: str | None = None, n: int | None = None,
              rounds: int | None = None, temperature: float | None = None,
              store: bool = True, use_memory: bool = True) -> dict:
        plan = plan_for(task)
        n = n or plan["n"]
        rounds = rounds or plan["rounds"]
        temperature = temperature if temperature is not None else plan["temperature"]

        if use_memory:
            solved = self.memory.memory_solve(task, tests or "", self.verifier)
            if solved:
                if store and tests:
                    self.memory.add_trace(task, solved["code"], tests, source="memory")
                return {
                    "ok": True, "code": solved["code"], "round": 0, "sample": 0,
                    "attempts": 1, "one_shot": None, "best_of_n": None,
                    "source": solved["source"], "history": [],
                }

        docs = self.memory.retrieve(task, k=3)
        prompt = build_prompt(task, docs)
        history = []
        one_shot = None
        best_of_n = False
        for round_idx in range(1, rounds + 1):
            samples = self.generator.sample(prompt, n=n, temperature=temperature)
            for sample_idx, text in enumerate(samples):
                code = extract_code11(text)
                # the task names the function; comply before testing the logic
                code = self.healer.normalize(code, task)
                result = self.verifier.verify(code, tests)
                rec = {"round": round_idx, "sample": sample_idx, "code": code,
                       "passed": result["passed"], "error": result["error"]}
                history.append(rec)
                if round_idx == 1:
                    if one_shot is None:
                        one_shot = result["passed"]
                    best_of_n = best_of_n or result["passed"]
                if result["passed"]:
                    if store and tests:
                        self.memory.add_trace(task, code, tests, source="neural")
                    return {"ok": True, "code": code, "round": round_idx,
                            "sample": sample_idx, "attempts": len(history),
                            "one_shot": one_shot, "best_of_n": best_of_n,
                            "source": "neural", "history": history}

                # prerequisite-first repair: add imports / constants / env setup
                planned = heal_code(code, task)
                if planned["code"] != code:
                    pres = self.verifier.verify(planned["code"], tests)
                    history.append({"round": round_idx, "sample": "planned",
                                    "code": planned["code"], "passed": pres["passed"],
                                    "error": pres["error"],
                                    "note": ", ".join(planned["notes"])})
                    if pres["passed"]:
                        if store and tests:
                            self.memory.add_trace(task, planned["code"], tests, source="planned")
                        return {"ok": True, "code": planned["code"], "round": round_idx,
                                "sample": "planned", "attempts": len(history),
                                "one_shot": one_shot, "best_of_n": True,
                                "source": "planned", "history": history}

            # deterministic healing of the most useful failure
            failures = [h for h in history if not h["passed"]]
            pick = next((h for h in reversed(failures)
                         if not h["error"].startswith("SyntaxError")), failures[-1] if failures else None)
            if pick:
                healed = self.healer.auto_fix(pick["code"], task, pick["error"])
                if healed:
                    res = self.verifier.verify(healed["code"], tests)
                    history.append({"round": round_idx, "sample": "heal",
                                    "code": healed["code"], "passed": res["passed"],
                                    "error": res["error"], "note": healed["note"]})
                    if res["passed"]:
                        if store and tests:
                            self.memory.add_trace(task, healed["code"], tests, source="healed")
                        return {"ok": True, "code": healed["code"], "round": round_idx,
                                "sample": "heal", "attempts": len(history),
                                "one_shot": one_shot, "best_of_n": True,
                                "source": "healed", "history": history}
                prompt = repair_prompt(task, pick["code"], pick["error"], docs)

        best = history[-1] if history else None
        return {"ok": False, "code": best["code"] if best else "",
                "attempts": len(history), "one_shot": one_shot,
                "best_of_n": best_of_n, "source": "neural", "history": history,
                "error": best["error"] if best else "no candidates"}


# ------------------------------------------------------------------ doctor

class Doctor11:
    def __init__(self):
        self.report: list[tuple[str, str]] = []

    def add(self, level: str, msg: str):
        self.report.append((level, msg))

    def check(self) -> list[tuple[str, str]]:
        self.report = []
        if TOKENIZER_PATH.exists():
            try:
                from brain11_tokenizer import get_tokenizer
                tok = get_tokenizer()
                self.add("ok" if tok.get_vocab_size() == 4096 else "warn",
                         f"tokenizer: {tok.get_vocab_size()} tokens")
            except Exception as exc:
                self.add("fail", f"tokenizer unreadable: {exc}")
        else:
            self.add("fail", "tokenizer missing: brain11_tokenizer.json")

        expected = Brain11Config().n_params()
        for name in ("coder", "language", "agentic"):
            path = HERE / f"brain11_{name}_weights.pt"
            if not path.exists():
                self.add("warn", f"{name}: not trained yet ({path.name} missing)")
                continue
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                cfg = Brain11Config(**ckpt["config"])
                n = cfg.n_params()
                self.add("ok" if n == expected else "warn",
                         f"{name}: {n:,} params, step {ckpt.get('step', '?')}")
            except Exception as exc:
                self.add("fail", f"{name}: checkpoint unreadable: {exc}")
            state = HERE / f"brain11_{name}_state.json"
            if state.exists():
                try:
                    s = json.loads(state.read_text(encoding="utf-8"))
                    self.add("ok", f"{name}: state step {s.get('step')}, "
                                   f"val {s.get('last_val_ppl')}, best {s.get('best_val')}, "
                                   f"mode {s.get('mode')}, optim_matches_weights {s.get('optim_matches_weights', 'n/a')}")
                except Exception as exc:
                    self.add("warn", f"{name}: state unreadable: {exc}")
            else:
                self.add("warn", f"{name}: state file missing")

        for label, path in (("code corpus", CODE_CORPUS),
                            ("selflearn", SELFLEARN)):
            if path.exists():
                size = path.stat().st_size / 1e6
                extra = "" if size > 0 else " (no verified self-learned examples yet)"
                self.add("ok", f"{label}: {size:.2f} MB{extra}")
            else:
                # self-learning is optional until first verified `brain11 learn`; code corpus is required
                self.add("warn" if label == "selflearn" else "fail",
                         f"{label}: missing {path.name}" +
                         (" (empty ok; `brain11 doctor --fix` creates it; `brain11 selflearn N` appends verified)" if label == "selflearn" else ""))

        free = shutil.disk_usage(HERE).free / (1024 ** 3)
        self.add("ok" if free > 2 else "warn", f"disk free: {free:.1f} GB")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            self.add("ok", f"GPU: {torch.cuda.get_device_name(0)} "
                           f"({alloc:.0f} MB in use by this process)")
        else:
            self.add("warn", "GPU unavailable; CPU mode")
        return self.report

    def fix(self) -> list[tuple[str, str]]:
        fixed = []

        # self-learning is optional, but doctor creates a clean empty pool; nothing is fabricated.
        for path in (SELFLEARN, SELFLEARN_META, REPAIRS):
            if not path.exists():
                _atomic_write(path, "")
                fixed.append(("ok", f"created empty {path.name}"))

        try:
            from brain11_train import restore_best  # properly restore shared agentic + sync `_state`
            for r in restore_best(["coder", "language", "agentic"]):
                name = r.get("model")
                if r.get("ok"):
                    fixed.append(("ok", f"{name}: restored `_weights.pt` from `_best.pt`, synced `_state`, "
                                        f"snapshotted `_optim`; no data deleted ({r.get('backup')})"))
                else:
                    # no `_best.pt`: at least rebuild missing `_weights.pt` from `_best.pt` isn't possible; rebuild state if needed
                    fixed.append(("warn", f"{name}: restore-best skipped: {r.get('error')}"))
        except Exception as exc:
            fixed.append(("warn", f"restore-best via trainer failed: {exc}"))

        for name in ("coder", "language", "agentic"):
            weights = HERE / f"brain11_{name}_weights.pt"
            best = HERE / f"brain11_{name}_best.pt"
            state = HERE / f"brain11_{name}_state.json"
            opt = HERE / f"brain11_{name}_optim.pt"
            if not weights.exists() and best.exists():
                shutil.copyfile(best, weights)
                fixed.append(("ok", f"{name}: restored missing weights from best checkpoint"))
            if weights.exists() and not state.exists():
                try:
                    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
                    payload = {
                        "model": name, "step": ckpt.get("step", 0),
                        "tokens_seen": 0, "best_val": float("inf"),
                        "config": ckpt.get("config", {}),
                        "n_params": Brain11Config().n_params(),
                        "data_sha256": None, "tokenizer_sha256": None,
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "rebuilt_by_doctor": True,
                    }
                    _atomic_write(state, json.dumps(payload, indent=2))
                    fixed.append(("ok", f"{name}: rebuilt missing state file"))
                except Exception as exc:
                    fixed.append(("fail", f"{name}: could not rebuild state: {exc}"))
            if weights.exists() and not opt.exists():
                fixed.append(("warn", f"{name}: optimizer state missing (weights intact; resume starts fresh optimizer)"))
        self.report.extend(fixed)
        return self.report


# ------------------------------------------------------------------ teacher / tester

def curriculum() -> list[tuple[str, str]]:
    """Verified task families used for self-learning and evaluation."""
    names = ["add", "subtract", "multiply", "divide", "square", "cube", "absolute",
             "max2", "min2", "is_even", "is_odd", "factorial", "fibonacci",
             "is_prime", "reverse_string", "count_vowels", "is_palindrome",
             "sum_list", "max_list", "celsius_to_fahrenheit"]
    tasks = []
    for i, name in enumerate(names):
        v = f"{name}_v{i}"
        if name == "add":
            tasks.append((f"write a function {v}(a, b) that returns a + b",
                          f"assert {v}(2, 3) == 5\nassert {v}(-1, 1) == 0"))
        elif name == "subtract":
            tasks.append((f"write a function {v}(a, b) that returns a - b",
                          f"assert {v}(5, 3) == 2\nassert {v}(2, 5) == -3"))
        elif name == "multiply":
            tasks.append((f"write a function {v}(a, b) that returns a * b",
                          f"assert {v}(3, 4) == 12\nassert {v}(-2, 3) == -6"))
        elif name == "divide":
            tasks.append((f"write a function {v}(a, b) that returns a / b, or None if b is zero",
                          f"assert {v}(10, 2) == 5.0\nassert {v}(7, 0) is None"))
        elif name == "square":
            tasks.append((f"write a function {v}(n) that returns n squared",
                          f"assert {v}(3) == 9\nassert {v}(-4) == 16"))
        elif name == "cube":
            tasks.append((f"write a function {v}(n) that returns n cubed",
                          f"assert {v}(2) == 8\nassert {v}(-3) == -27"))
        elif name == "absolute":
            tasks.append((f"write a function {v}(n) that returns the absolute value of n",
                          f"assert {v}(-5) == 5\nassert {v}(3) == 3"))
        elif name == "max2":
            tasks.append((f"write a function {v}(a, b) that returns the larger value",
                          f"assert {v}(2, 7) == 7\nassert {v}(9, 1) == 9"))
        elif name == "min2":
            tasks.append((f"write a function {v}(a, b) that returns the smaller value",
                          f"assert {v}(2, 7) == 2\nassert {v}(9, 1) == 1"))
        elif name == "is_even":
            tasks.append((f"write a function {v}(n) that returns True if n is even",
                          f"assert {v}(4) is True\nassert {v}(7) is False"))
        elif name == "is_odd":
            tasks.append((f"write a function {v}(n) that returns True if n is odd",
                          f"assert {v}(7) is True\nassert {v}(4) is False"))
        elif name == "factorial":
            tasks.append((f"write a function {v}(n) that returns n factorial",
                          f"assert {v}(0) == 1\nassert {v}(5) == 120"))
        elif name == "fibonacci":
            tasks.append((f"write a function {v}(n) that returns the nth Fibonacci number",
                          f"assert {v}(0) == 0\nassert {v}(1) == 1\nassert {v}(7) == 13"))
        elif name == "is_prime":
            tasks.append((f"write a function {v}(n) that returns True if n is prime",
                          f"assert {v}(2) is True\nassert {v}(9) is False\nassert {v}(13) is True"))
        elif name == "reverse_string":
            tasks.append((f"write a function {v}(s) that returns s reversed",
                          f"assert {v}('abc') == 'cba'\nassert {v}('') == ''"))
        elif name == "count_vowels":
            tasks.append((f"write a function {v}(s) that counts vowels in s",
                          f"assert {v}('hello') == 2\nassert {v}('xyz') == 0"))
        elif name == "is_palindrome":
            tasks.append((f"write a function {v}(s) that returns True if s is a palindrome",
                          f"assert {v}('madam') is True\nassert {v}('abc') is False"))
        elif name == "sum_list":
            tasks.append((f"write a function {v}(items) that returns the sum of a list",
                          f"assert {v}([1, 2, 3]) == 6\nassert {v}([]) == 0"))
        elif name == "max_list":
            tasks.append((f"write a function {v}(items) that returns the largest item",
                          f"assert {v}([1, 9, 2]) == 9\nassert {v}([-3, -1]) == -1"))
        elif name == "celsius_to_fahrenheit":
            tasks.append((f"write a function {v}(c) that converts Celsius to Fahrenheit",
                          f"assert {v}(0) == 32\nassert {v}(100) == 212"))
    return tasks


class Teacher11:
    def __init__(self, cortex: Cortex11):
        self.cortex = cortex
        self.tasks = curriculum()

    def append_selflearn(self, task: str, code: str, tests: str, source: str,
                         model_step: int | None = None):
        key = _hash(task, code)
        existing = set()
        if SELFLEARN_META.exists():
            for line in SELFLEARN_META.read_text(encoding="utf-8").splitlines():
                try:
                    existing.add(json.loads(line)["key"])
                except Exception:
                    pass
        if key in existing:
            return False
        doc = f"### Task:\n{task}\n### Code:\n{code}\n### Tests:\n{tests}\n"
        SELFLEARN.parent.mkdir(parents=True, exist_ok=True)
        with open(SELFLEARN, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        meta = {"key": key, "task": task, "source": source,
                "model_step": model_step, "ts": time.time(),
                "verified": True, "accepted": True}
        with open(SELFLEARN_META, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True

    def append_repair(self, task: str, bad_code: str, error: str, fixed: str):
        doc = (f"### Task:\n{task}\n### Previous Code:\n{bad_code}\n"
               f"### Error:\n{error}\n### Corrected Code:\n{fixed}\n")
        REPAIRS.parent.mkdir(parents=True, exist_ok=True)
        with open(REPAIRS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def learn(self, limit: int = 24, n: int = 4, rounds: int = 2,
              use_memory: bool = True) -> dict:
        solved = learned = 0
        for task, tests in self.tasks[:limit]:
            res = self.cortex.solve(task, tests, n=n, rounds=rounds, store=True,
                                    use_memory=use_memory)
            if res["ok"]:
                solved += 1
                if self.append_selflearn(task, res["code"], tests, res["source"]):
                    learned += 1
                # store an error->fix pair when the loop had to repair
                fails = [h for h in res.get("history", []) if not h["passed"] and h.get("code")]
                if fails and res["source"] in ("healed", "neural"):
                    self.append_repair(task, fails[-1]["code"], fails[-1]["error"], res["code"])
            print(f"[learn] {'ok ' if res['ok'] else 'no '} {task[:60]}", flush=True)
        print(f"[learn] solved {solved}/{min(limit, len(self.tasks))}, "
              f"new verified examples {learned}")
        print("[learn] next: python3 brain11_train.py --model coder --steps <new-total>")
        return {"solved": solved, "learned": learned, "total": min(limit, len(self.tasks))}

    def evaluate(self, limit: int | None = None, n: int = 4, rounds: int = 2,
                 use_memory: bool = True) -> dict:
        rows = []
        subset = self.tasks[:limit] if limit else self.tasks
        for i, (task, tests) in enumerate(subset):
            # reproducible eval: identical task -> identical samples
            torch.manual_seed(20261125 + i)
            res = self.cortex.solve(task, tests, n=n, rounds=rounds, store=False,
                                    use_memory=use_memory)
            rows.append({"task": task, "ok": res["ok"], "source": res["source"],
                         "attempts": res["attempts"], "one_shot": res.get("one_shot"),
                         "best_of_n": res.get("best_of_n")})
            print(f"[eval] {'ok ' if res['ok'] else 'no '} {res['source']:20s} {task[:55]}", flush=True)
        total = max(1, len(rows))
        neural = [r for r in rows if not str(r["source"]).startswith("memory")]
        summary = {
            "tasks": len(rows),
            "loop": sum(r["ok"] for r in rows) / total,
            "memory": sum(str(r["source"]).startswith("memory") for r in rows) / total,
            "one_shot": sum(bool(r["one_shot"]) for r in neural) / max(1, len(neural)),
            "best_of_n": sum(bool(r["best_of_n"]) for r in neural) / max(1, len(neural)),
        }
        print("[eval] " + json.dumps(summary, indent=2))
        report = {"summary": summary, "rows": rows,
                  "passed": [r["task"] for r in rows if r["ok"]],
                  "failed": [r["task"] for r in rows if not r["ok"]]}
        _atomic_write(HERE / "brain11_eval_last.json",
                      json.dumps(report, indent=2, ensure_ascii=False))
        return summary


# ------------------------------------------------------------------ selftest

def selftest() -> bool:
    checks = []
    verifier = Verifier()
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    tests = "assert add(2, 3) == 5"
    checks.append(("verifier accepts correct code", verifier.verify(good, tests)["passed"]))
    checks.append(("verifier rejects wrong code", not verifier.verify(bad, tests)["passed"]))
    checks.append(("verifier rejects forbidden import",
                   not verifier.verify("import os\ndef add(a,b):\n    return a+b", tests)["passed"]))

    healer = Healer()
    fixed = healer.auto_fix("def add(a, b)\n    return a + b", "write add(a,b)", "SyntaxError: expected ':'")
    checks.append(("healer fixes missing colon", bool(fixed and verifier.verify(fixed["code"], tests)["passed"])))
    renamed = healer.auto_fix("def sum2(a, b):\n    return a + b", "write add(a,b)", "NameError: name 'add' is not defined")
    checks.append(("healer renames function", bool(renamed and "def add(" in renamed["code"])))

    docs = parse_code_docs(CODE_CORPUS) if CODE_CORPUS.exists() else []
    if docs:
        mem = Memory11()
        hits = mem.retrieve("write a function add_pair(a, b) that returns a + b", k=3)
        checks.append(("BM25 retrieval works", bool(hits)))
    else:
        checks.append(("BM25 retrieval works (skipped: no corpus)", True))

    # agentic policy validation (no model needed)
    from brain11_router import _decision_choices, _infer_tool
    spec = _infer_tool("What is the weather in Lahore?")
    choices = _decision_choices(spec, "new", {"city": "Lahore"})
    ok = False
    for c in choices:
        d = json.loads(c)
        if d["action"] == "call_tool" and d["tool"] == "get_weather" and d["args"]["city"] == "Lahore":
            ok = True
    checks.append(("agentic decision is valid JSON + correct tool", ok))
    checks.append(("classifier routes code", classify_task("write a Python function") == "coder"))
    checks.append(("classifier routes agentic", classify_task("status: new, predict next action") == "agentic"))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"[selftest] {passed}/{len(checks)} passed")
    return passed == len(checks)


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description="Brain11 self-learning/healing loop")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doctor", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--solve")
    ap.add_argument("--tests", default="")
    ap.add_argument("--learn", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-memory", action="store_true",
                    help="disable BM25 retrieval so results measure the neural model only")
    ap.add_argument("--show-history", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    if args.doctor:
        doc = Doctor11()
        doc.check()
        if args.fix:
            doc.fix()
        for level, msg in doc.report:
            print(f"[{level:4s}] {msg}")
        return

    if not (args.solve or args.learn or args.eval):
        ap.print_help()
        return

    router = Brain11Router(device=args.device, lazy=True)
    cortex = Cortex11(router)
    if args.solve:
        res = cortex.solve(args.solve, args.tests, n=args.n, rounds=args.rounds,
                           store=True)
        print(json.dumps({k: v for k, v in res.items() if k != "history"},
                         ensure_ascii=False, indent=2))
        if args.show_history:
            for h in res.get("history", []):
                print(f"  round={h['round']} sample={h['sample']} passed={h['passed']} "
                      f"error={h['error'][:80]}")
        raise SystemExit(0 if res["ok"] else 1)

    teacher = Teacher11(cortex)
    if args.learn:
        teacher.learn(limit=args.learn, n=args.n, rounds=args.rounds,
                      use_memory=not args.no_memory)
    if args.eval:
        teacher.evaluate(limit=args.limit, n=args.n, rounds=args.rounds,
                         use_memory=not args.no_memory)


if __name__ == "__main__":
    main()
