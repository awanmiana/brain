#!/usr/bin/env python3
"""BrainCortex — verifier-gated, memory-augmented, plastic MoE agent.

The loop: route -> retrieve -> propose (best-of-N) -> verify -> repair -> remember.

This is the first implemented slice of the BrainCortex design: the neural core
(Brain) proposes, a memory of verified solutions supplies analogies, and a
sandboxed verifier decides what is true. Nothing is accepted without a check.

Usage:
    python3 brain_cortex.py --selftest
    python3 brain_cortex.py --task "write a function add(a, b) that returns a + b" \
        --tests "assert add(2, 3) == 5" --n 4 --rounds 2
    python3 brain_cortex.py --eval-file brain_cortex_eval.json --limit 12 --n 2 --rounds 2
"""

import argparse
import ast
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "brain_weights.json"
DEFAULT_MEMORY = HERE / "brain_cortex_memory.json"
TEACHER_CORPUS = HERE / "corpus_teacher.txt"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    """A small, dependency-free BM25 index."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d["text"]) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf = [Counter(t) for t in self.doc_tokens]
        self.df = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1
        self.n = len(docs)

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _score(self, query, idx):
        q = tokenize(query)
        tf = self.tf[idx]
        dl = self.doc_len[idx]
        score = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avgdl))
            score += self._idf(term) * f * (self.k1 + 1) / denom
        return score

    def top(self, query, k=3):
        scored = [(self._score(query, i), i) for i in range(self.n)]
        scored.sort(reverse=True)
        return [(s, self.docs[i]) for s, i in scored[:k] if s > 0]


class Memory:
    """Hippocampus: verified traces + corpus examples, retrieved by BM25."""

    def __init__(self, corpus_paths=None, traces_path=DEFAULT_MEMORY, repo_paths=None, limit_files=200):
        self.docs = []
        self.traces = []
        self.traces_path = Path(traces_path)
        self._load_traces()
        for path in corpus_paths or []:
            self._load_text_file(Path(path))
        if repo_paths:
            self._load_repos([Path(p) for p in repo_paths], limit_files)
        self.index = BM25(self.docs)
        print(f"[memory] {len(self.docs)} documents, {len(self.traces)} verified traces", flush=True)

    def _load_traces(self):
        if self.traces_path.exists():
            try:
                self.traces = json.loads(self.traces_path.read_text(encoding="utf-8"))
            except Exception:
                self.traces = []
        for trace in self.traces:
            self.docs.append({"text": f"# task: {trace['task']}\n{trace['code']}", "source": "trace"})

    def _load_text_file(self, path):
        if not path.exists():
            print(f"[memory] missing: {path}", flush=True)
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        for chunk in re.split(r"\n\s*\n(?=# task:)", text):
            chunk = chunk.strip()
            if chunk:
                self.docs.append({"text": chunk, "source": path.name})

    def _load_repos(self, paths, limit_files):
        files = []
        for path in paths:
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(path.rglob("*.py")))
        files = files[:limit_files]
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for start in range(0, len(text), 1200):
                chunk = text[start:start + 1200]
                if chunk.strip():
                    self.docs.append({"text": chunk, "source": f"{path.name}:{start}"})
        print(f"[memory] indexed {len(files)} repo files", flush=True)

    def retrieve(self, query, k=3):
        return [doc for _, doc in self.index.top(query, k)]

    def add_trace(self, task, code, tests):
        self.traces.append({"task": task, "code": code, "tests": tests, "ts": time.time()})
        self.docs.append({"text": f"# task: {task}\n{code}", "source": "trace"})
        self.traces_path.write_text(json.dumps(self.traces, indent=2), encoding="utf-8")
        self.index = BM25(self.docs)


class Verifier:
    """Critic: AST safety + syntax + sandboxed execution of unit tests."""

    FORBIDDEN_MODULES = {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
        "ctypes", "multiprocessing", "threading", "builtins",
    }
    FORBIDDEN_CALLS = {
        "eval", "exec", "open", "compile", "__import__", "input", "globals",
        "locals", "vars",
    }

    def check_safety(self, code):
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.FORBIDDEN_MODULES:
                        return False, f"forbidden import: {root}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in self.FORBIDDEN_MODULES:
                    return False, f"forbidden import: {root}"
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else ""
                )
                if name in self.FORBIDDEN_CALLS:
                    return False, f"forbidden call: {name}"
        return True, "ok"

    def verify(self, code, tests=None):
        if not code.strip():
            return {"passed": False, "syntax_ok": False, "error": "empty candidate"}
        safe, message = self.check_safety(code)
        if not safe:
            return {"passed": False, "syntax_ok": True, "error": message}
        if not tests:
            return {"passed": True, "syntax_ok": True, "error": ""}
        program = code + "\n\n" + tests + "\n\nprint('__PASS__')\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solution.py"
            path.write_text(program, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    capture_output=True, text=True, timeout=6, cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return {"passed": False, "syntax_ok": True, "error": "timeout"}
        stdout = proc.stdout or ""
        if proc.returncode == 0 and "__PASS__" in stdout:
            return {"passed": True, "syntax_ok": True, "error": ""}
        lines = (proc.stderr or stdout or "").strip().splitlines()
        return {"passed": False, "syntax_ok": True, "error": lines[-1] if lines else f"exit {proc.returncode}"}


def extract_code(text):
    """Pull the first complete Python definition out of a raw generation."""
    if not text:
        return ""
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "@")):
            start = i
            break
    if start is None:
        return text.strip()
    body = []
    for line in lines[start:]:
        if line.startswith(("# task:", "# context:", "# error:", "# corrected")):
            break
        body.append(line)
    while body and not body[-1].strip():
        body.pop()
    for end in range(len(body), 0, -1):
        candidate = "\n".join(body[:end]).rstrip()
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return "\n".join(body).rstrip()


def route(task):
    text = task.lower()
    if any(word in text for word in ("fix", "bug", "error", "debug", "fails")):
        return {"kind": "debug", "n": 8, "rounds": 3, "temperature": 0.9}
    if any(word in text for word in ("write", "implement", "create", "function", "def ", "returns")):
        return {"kind": "code", "n": 8, "rounds": 2, "temperature": 0.9}
    return {"kind": "general", "n": 4, "rounds": 1, "temperature": 0.8}


def build_prompt(task, docs):
    parts = []
    for doc in docs:
        text = doc["text"].strip()
        if text.startswith("# task:"):
            parts.append(text)
    parts.append(f"# task: {task}")
    return "\n\n".join(parts) + "\n"


def repair_prompt(task, code, error):
    return (
        f"# task: {task}\n"
        f"# previous attempt:\n{code}\n"
        f"# error: {error}\n"
        f"# corrected solution:\n"
    )


def requested_name(task):
    for pattern in (
        r"function\s+([A-Za-z_]\w*)\s*\(",
        r"function\s+([A-Za-z_]\w*)",
        r"implement\s+([A-Za-z_]\w*)",
        r"create\s+([A-Za-z_]\w*)",
    ):
        match = re.search(pattern, task)
        if match:
            return match.group(1)
    match = NAME_RE.search(task)
    return match.group(1) if match else None


def memory_solve(memory, verifier, task, tests):
    """Case-based reasoning: retrieve an analogous verified solution, rename it."""
    if not tests:
        return None
    wanted = requested_name(task)
    if not wanted:
        return None
    for _, doc in memory.index.top(task, k=5):
        text = doc["text"]
        match = re.search(r"# task:[^\n]*\n(def\s+([A-Za-z_]\w*)\s*\()", text)
        if not match:
            continue
        old = match.group(2)
        code = text[match.start(1):].strip()
        candidate = code if old == wanted else re.sub(rf"\b{re.escape(old)}\b", wanted, code)
        if verifier.verify(candidate, tests)["passed"]:
            return {"code": candidate, "source": f"memory:{old}->{wanted}"}
    return None


class Generator:
    def __init__(self, model, vocab, max_new=160):
        self.model = model
        self.vocab = vocab
        self.max_new = max_new

    def sample(self, prompt, n=1, temperature=0.9, top_k=40):
        ids = self.vocab.encode(prompt)
        if temperature is not None and temperature <= 0:
            outs = [self.model.generate(ids, n=self.max_new)]
        else:
            outs = self.model.sample(ids, n=n, max_new=self.max_new, temperature=temperature, top_k=top_k)
        return [self.vocab.decode(out) for out in outs]


class Cortex:
    """The loop: memory-first, then neural proposals, verification and repair."""

    def __init__(self, generator, memory, verifier, max_new=160):
        self.generator = generator
        self.memory = memory
        self.verifier = verifier
        self.max_new = max_new

    def solve(self, task, tests=None, n=None, rounds=None, temperature=None, store=True, use_memory=True):
        plan = route(task)
        n = n or plan["n"]
        rounds = rounds or plan["rounds"]
        temperature = temperature if temperature is not None else plan["temperature"]

        if use_memory:
            solved = memory_solve(self.memory, self.verifier, task, tests)
            if solved:
                if store:
                    self.memory.add_trace(task, solved["code"], tests)
                return {
                    "ok": True, "code": solved["code"], "round": 0, "sample": 0,
                    "attempts": 1, "one_shot": None, "best_of_n": None,
                    "source": solved["source"],
                    "history": [{"round": 0, "sample": 0, "code": solved["code"],
                                 "passed": True, "error": "", "source": solved["source"]}],
                }

        docs = self.memory.retrieve(task, k=3)
        prompt = build_prompt(task, docs)
        history = []
        one_shot = None
        best_of_n = False
        for round_idx in range(1, rounds + 1):
            samples = self.generator.sample(prompt, n=n, temperature=temperature)
            for sample_idx, text in enumerate(samples):
                code = extract_code(text)
                result = self.verifier.verify(code, tests)
                record = {
                    "round": round_idx, "sample": sample_idx, "code": code,
                    "passed": result["passed"], "error": result["error"], "text": text,
                }
                history.append(record)
                if round_idx == 1:
                    if one_shot is None:
                        one_shot = result["passed"]
                    best_of_n = best_of_n or result["passed"]
                if result["passed"]:
                    if store and tests:
                        self.memory.add_trace(task, code, tests)
                    return {
                        "ok": True, "code": code, "round": round_idx, "sample": sample_idx,
                        "attempts": len(history), "one_shot": one_shot, "best_of_n": best_of_n,
                        "source": "neural", "history": history,
                    }
            failures = [h for h in history if not h["passed"]]
            pick = None
            for record in reversed(failures):
                if not record["error"].startswith("SyntaxError"):
                    pick = record
                    break
            if pick is None and failures:
                pick = failures[-1]
            if pick:
                prompt = repair_prompt(task, pick["code"], pick["error"])

        best = history[-1] if history else None
        return {
            "ok": False, "code": best["code"] if best else "",
            "attempts": len(history), "one_shot": one_shot, "best_of_n": best_of_n,
            "source": "neural", "history": history,
            "error": best["error"] if best else "no candidates",
        }


def evaluate(cortex, items, n, rounds, limit=None):
    rows = []
    subset = items[:limit] if limit else items
    for idx, item in enumerate(subset):
        task, tests, split = item["task"], item["tests"], item.get("split", "?")
        result = cortex.solve(task, tests, n=n, rounds=rounds, store=False)
        row = {
            "task": task, "split": split,
            "memory": result.get("source", "").startswith("memory"),
            "one_shot": result.get("one_shot"),
            "best_of_n": result.get("best_of_n"),
            "loop": result["ok"],
            "attempts": result["attempts"],
        }
        rows.append(row)
        mark = "mem" if row["memory"] else ("ok " if row["loop"] else "no ")
        print(f"[{idx + 1}/{len(subset)}] {split:12s} {mark} attempts={row['attempts']} {task[:64]}", flush=True)

    def rate(key, subset_rows):
        return sum(1 for r in subset_rows if r[key]) / max(1, len(subset_rows))

    summary = {}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        neural = [r for r in rs if not r["memory"]]
        summary[split] = {
            "tasks": len(rs),
            "memory": rate("memory", rs),
            "total": rate("loop", rs),
            "neural_tasks": len(neural),
            "one_shot": sum(1 for r in neural if r["one_shot"]) / max(1, len(neural)),
            "best_of_n": sum(1 for r in neural if r["best_of_n"]) / max(1, len(neural)),
            "loop": sum(1 for r in neural if r["loop"]) / max(1, len(neural)),
        }
    return rows, summary


def selftest():
    checks = []
    verifier = Verifier()
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    tests = "assert add(2, 3) == 5"
    checks.append(("correct solution passes", verifier.verify(good, tests)["passed"]))
    checks.append(("wrong solution fails", not verifier.verify(bad, tests)["passed"]))
    checks.append(("syntax error rejected", not verifier.verify("def add(a, b)", tests)["passed"]))
    checks.append(("forbidden import rejected", not verifier.verify(
        "import os\ndef add(a, b):\n    return a + b", tests)["passed"]))
    checks.append(("extract_code finds def", extract_code(
        "noise\n```python\ndef add(a, b):\n    return a + b\n```\nmore").startswith("def add")))
    memory = Memory(corpus_paths=[TEACHER_CORPUS] if TEACHER_CORPUS.exists() else None)
    hits = memory.retrieve("write a function add_func(a, b) that returns the sum of a and b", k=3)
    checks.append(("BM25 retrieves an add example", any("return a + b" in h["text"] for h in hits)))
    solved = memory_solve(
        memory, verifier,
        "write a function add_func(a, b) that returns the sum of a and b",
        "assert add_func(2, 3) == 5",
    )
    checks.append(("memory solver renames an analogy", bool(solved)))
    ok = True
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="BrainCortex — verified, memory-augmented MoE agent.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--tests", help="inline test assertions")
    parser.add_argument("--tests-file")
    parser.add_argument("--eval-file")
    parser.add_argument("--split", help="only evaluate tasks with this split (e.g. seen_family, novel)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--vocab", default="brain_vocab.json")
    parser.add_argument("--preset", default="FIT_8GB", choices=["FIT_8GB", "QWEN2B"])
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--index", default=str(TEACHER_CORPUS))
    parser.add_argument("--repo", help="optional repo file/dir to index")
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY))
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--no-analogy", action="store_true")
    parser.add_argument("--memory-only", action="store_true", help="skip the model; test the memory/analogy path")
    parser.add_argument("--dump")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    tests = None
    if args.tests_file:
        tests = Path(args.tests_file).read_text(encoding="utf-8")
    elif args.tests:
        tests = args.tests

    if args.eval_file:
        items = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
        if args.split:
            items = [item for item in items if item.get("split") == args.split]
            if not items:
                parser.error(f"no tasks with split={args.split!r}")
    elif args.task:
        items = None
    else:
        parser.error("--task, --eval-file or --selftest required")

    import torch
    from brain import FIT_8GB, QWEN2B, Brain
    from bpe_tokenizer import Vocab, load_tokens

    corpus_paths = [p.strip() for p in args.index.split(",") if p.strip()]
    repo_paths = [p.strip() for p in args.repo.split(",") if p.strip()] if args.repo else None
    memory = Memory(corpus_paths=corpus_paths, traces_path=args.memory, repo_paths=repo_paths)

    if args.memory_only:
        verifier = Verifier()
        if items is not None:
            subset = items[:args.limit] if args.limit else items
            rows = []
            for idx, item in enumerate(subset):
                solved = memory_solve(memory, verifier, item["task"], item["tests"])
                rows.append({"split": item.get("split", "?"), "ok": bool(solved)})
                print(f"[{idx + 1}/{len(subset)}] {'mem' if solved else 'no '} "
                      f"{item.get('split', '?'):12s} {item['task'][:60]}", flush=True)
            for split in sorted({r["split"] for r in rows}):
                rs = [r for r in rows if r["split"] == split]
                print(f"{split}: {sum(1 for r in rs if r['ok'])}/{len(rs)} solved by memory")
            return
        if not args.task:
            parser.error("--memory-only requires --task or --eval-file")
        solved = memory_solve(memory, verifier, args.task, tests)
        if solved:
            print(f"source: {solved['source']}")
            print(solved["code"])
        else:
            print("memory: no verified analogy found")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FIT_8GB if args.preset == "FIT_8GB" else QWEN2B
    tokens = load_tokens(args.vocab)
    model = Brain(tokens, use_rope=args.rope, **cfg).to(device)
    print(f"[cortex] Brain({args.preset}): {model.n_params():,} params, rope={args.rope}, device={device}", flush=True)
    if not args.fresh:
        path = Path(args.weights)
        if path.exists():
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            print(f"[cortex] loaded <- {path}", flush=True)
        else:
            print(f"[cortex] WARNING: weights not found: {path} (random init)", flush=True)
    model.eval()

    vocab = Vocab(tokens)
    generator = Generator(model, vocab, max_new=args.max_new)
    cortex = Cortex(generator, memory, Verifier(), max_new=args.max_new)

    if items is not None:
        rows, summary = evaluate(cortex, items, args.n or 4, args.rounds or 2, limit=args.limit)
        print(json.dumps(summary, indent=2))
        if args.dump:
            Path(args.dump).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
        return

    result = cortex.solve(args.task, tests, n=args.n, rounds=args.rounds,
                          temperature=args.temperature, store=not args.no_store,
                          use_memory=not args.no_analogy)
    print()
    print(f"source: {result.get('source')}  ok: {result['ok']}  attempts: {result['attempts']}")
    if result.get("error"):
        print(f"error: {result['error']}")
    print("--- solution ---")
    print(result["code"])
    if args.dump:
        Path(args.dump).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[dump] {args.dump}")


if __name__ == "__main__":
    main()
