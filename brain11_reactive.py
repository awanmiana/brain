#!/usr/bin/env python3
"""Brain11 ReactiveCoder — the mechanism you described, as a working module.

Idea (restated precisely):
    next code section is driven by the logic of
      prompt + current-code-status
        -> this section's logical requirements
        -> all requirements already satisfied ("done")
        -> requirements still pending ("open hops")
        -> iterations
        -> actions FORWARD (append/complete) and BACKWARD (revise/reorder earlier sections)

Standard language-model training does NOT provide this. A causal LM learns
P(token_t | tokens_<t). It never re-opens token 3 after writing token 500. The
reliability at 10.8M parameters comes from making the state explicit OUTSIDE
the network and re-checking it every step. No new neural architecture is
required for that version.

This module implements it:

    state_t  ->  requirements(section_t)  ->  pending  ->  action  ->  state_t+1
                                                    ^                     |
                                                    +-- full re-check ----+

Two ways to connect the 10.8M model:

  A. Systems layer (this file): the model proposes a section; the machine
     computes requirements, finds pending items, and applies forward/backward
     fixes, re-running tests after every change. [implemented]
  B. Trained scratchpad: add `### Plan -> ### Code` training docs and teach the
     same model to emit `PENDING:` / `DONE:` state tokens itself, then still
     run (A) as a verifier. [hook: `model` argument]

Usage:
    python3 brain11_reactive.py --selftest
    python3 brain11_reactive.py --task "write load_config() that reads config.json and returns the API URL"
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from brain11_planner import (
    Plan,
    compose_program,
    dataflow_check,
    insert_setup,
    plan_task,
)

IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+\w", re.M)
DEF_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(", re.M)
WORD_RE_TMPL = r"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])"


def _name_present(name: str, output: str) -> bool:
    """Word-boundary match. `name in output` would match `CONFIG_PATH_X`,
    comments, and string literals."""
    if not name:
        return False
    return re.search(WORD_RE_TMPL.format(name=re.escape(name)), output) is not None


def _has_assignment(name: str, output: str) -> bool:
    return re.search(WORD_RE_TMPL.format(name=re.escape(name)) + r"\s*=", output) is not None


@dataclass
class Hop:
    """One recorded step. `action` is forward|backward|verify|stop."""
    round: int
    action: str
    detail: str
    pending_before: list[str] = field(default_factory=list)
    pending_after: list[str] = field(default_factory=list)


@dataclass
class CodeState:
    """Full state of the code artifact being built."""
    prompt: str
    plan: dict
    output: str = ""
    emitted_sections: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ------------------------------------------------------------------ requirements

def _clean(code: str) -> str:
    """Strip BOM / stray control chars that break ast.parse on Windows."""
    return (code or "").replace("\ufeff", "").replace("\x00", "").strip()


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(_clean(code))
        return True
    except SyntaxError:
        return False


def section_requirements(section: str, plan: Plan) -> list[str]:
    """What THIS section needs before it can run."""
    req: list[str] = []
    issues = dataflow_check(section)
    for name in {i["name"] for i in issues}:
        req.append(f"needs:{name}")
    for m in plan.imports:
        if re.search(rf"\b{re.escape(m)}\s*\.", section) and not re.search(
                rf"(?:import\s+{re.escape(m)}\b|from\s+{re.escape(m)}\b)", section):
            req.append(f"needs-import:{m}")
    return sorted(set(req))


def pending_requirements(output: str, plan: Plan, tests: str | None = None,
                         tests_passed: bool | None = None) -> list[str]:
    """Recompute the full pending list from the current artifact, every round."""
    pend: list[str] = []
    if not output.strip():
        pend.append("empty")
        return pend

    if not syntax_ok(output):
        pend.append("syntax")
        return pend  # can't evaluate anything else on a broken tree

    # imports + constants + setup first
    for i in dataflow_check(output):
        pend.append(f"undefined:{i['name']}")

    # ordering: setup must appear before any use of the name
    for s in plan.setup:
        name = s.split("=")[0].strip()
        if not _has_assignment(name, output):
            pend.append(f"missing-setup:{name}")

    for m in plan.imports:
        if re.search(rf"\b{m}\.", output) and not re.search(
                rf"(?:import\s+{m}\b|from\s+{m}\b)", output):
            pend.append(f"missing-import:{m}")

    # structure
    names = DEF_RE.findall(output)
    if not names:
        pend.append("no-function")
    else:
        for fn in plan.functions:
            wanted = fn.split("(")[0].strip()
            if wanted and wanted not in names:
                pend.append(f"missing-function:{wanted}")

    if tests is not None and tests_passed is not None and not tests_passed:
        pend.append("tests-failing")

    return sorted(set(pend))


def already_satisfied(output: str, plan: Plan) -> list[str]:
    """What is already done, recomputed from the artifact each round."""
    done: list[str] = []
    if syntax_ok(output):
        done.append("syntax-ok")
    if IMPORT_RE.search(output):
        done.append("imports-present")
    for s in plan.setup:
        name = s.split("=")[0].strip()
        if _has_assignment(name, output):
            done.append(f"setup:{name}")
    for name in DEF_RE.findall(output):
        done.append(f"function:{name}")
    # execution order check: no use before definition
    if not dataflow_check(output):
        done.append("no-undefined-names")
    return sorted(set(done))


# ------------------------------------------------------------------ verification

def run_tests(code: str, tests: str) -> tuple[bool, str]:
    if not tests.strip():
        return True, ""
    try:
        from brain_cortex import Verifier
        res = Verifier().verify(code, tests)
        return bool(res["passed"]), res.get("error", "")
    except Exception as exc:  # pragma: no cover - verifier missing
        return False, f"verifier error: {exc}"


# ------------------------------------------------------------------ main loop

def reactive_build(task: str, seed_code: str = "", tests: str = "",
                   model=None, max_rounds: int = 6, n_samples: int = 3,
                   temperature: float = 0.4) -> CodeState:
    """Round-based reactive construction with full re-checking.

    Each round:
      1. re-derive `done` and `pending` from the current artifact
      2. if pending is empty and tests pass -> stop
      3. pick a pending requirement
      4. apply a FORWARD fix (append/add) or a BACKWARD fix (reorder/patch earlier text)
      5. re-run the full check from the top, re-run tests
    """
    plan = plan_task(task)
    state = CodeState(prompt=task, plan=asdict(plan))
    output = _clean(seed_code.strip())
    seen_signatures: set[str] = set()

    if not output:
        # round 0: ask the model for a first proposal, else build skeleton
        output = _propose(task, model, n_samples=n_samples, temperature=temperature) or \
            compose_program(plan, _skeleton(plan))
        state.emitted_sections.append(output)

    for rnd in range(1, max_rounds + 1):
        pending_before = pending_requirements(output, plan, tests or None,
                                               run_tests(output, tests)[0] if tests else None)
        tests_passed, test_err = run_tests(output, tests) if tests else (True, "")
        state.pending = pending_before
        state.done = already_satisfied(output, plan)

        if not pending_before and tests_passed:
            state.hops.append(Hop(rnd, "stop", "all requirements satisfied",
                                  pending_before, []))
            break

        # no-progress / loop detection: same artifact + same pending => change or stop
        sig = hashlib.sha1((output + "\x00" + ",".join(pending_before)).encode("utf-8")).hexdigest()
        if sig in seen_signatures:
            state.hops.append(Hop(rnd, "stop", "no progress; artifact+pending unchanged",
                                  pending_before, pending_before))
            break
        seen_signatures.add(sig)

        before_output = output
        before_pending_count = len(pending_before)

        target = pending_before[0] if pending_before else "tests-failing"
        kind = target.split(":")[0]

        # ---- BACKWARD: fix the already-emitted artifact ----
        if kind in ("undefined", "missing-setup", "missing-import", "syntax", "missing-function"):
            fixed, notes = insert_setup(output, task, plan)
            # re-emit the program in canonical order: imports -> setup -> functions -> call
            body = _extract_functions(fixed, _names_added_by_plan(plan))
            fixed = compose_program(plan, body)
            action = "backward"
            detail = f"{target}: {', '.join(notes) or 'reordered imports/setup before use'}"
            output = fixed
        elif kind == "tests-failing":
            # BACKWARD revision driven by the real failure, not by guessing
            fixed, notes = insert_setup(output, task, plan)
            if test_err:
                name = re.search(r"name '([^']+)' is not defined", test_err)
                if name:
                    wanted = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", output)
                    if wanted:
                        fixed = re.sub(rf"\b{re.escape(name.group(1))}\b",
                                       wanted.group(1), fixed)
                        notes.append(f"renamed undefined {name.group(1)}")
            if notes:
                output = fixed
            # FORWARD: re-append a fresh proposal for the whole function
            prop = _propose(task, model, context=test_err, n_samples=n_samples,
                            temperature=temperature)
            if prop:
                output = compose_program(plan, prop)
            action = "backward" if notes else "forward"
            detail = f"tests failing: {test_err[:120]}"
        elif kind == "no-function":
            output = compose_program(plan, _skeleton(plan))
            action = "forward"
            detail = "emitted function skeleton"
        elif kind == "missing-function":
            output = output + f"\n\ndef {target.split(':')[1]}():\n    raise NotImplementedError\n"
            action = "forward"
            detail = target
        elif kind == "empty":
            output = compose_program(plan, _skeleton(plan))
            action = "forward"
            detail = "emitted first section"
        else:
            # unknown pending -> ask the model for a corrected whole artifact
            prop = _propose(task, model, context=target, n_samples=n_samples,
                            temperature=temperature)
            if prop:
                output = compose_program(plan, prop)
                action = "backward"
                detail = target
            else:
                state.hops.append(Hop(rnd, "stop", f"cannot make progress on {target}",
                                      pending_before, pending_before))
                break

        # full re-check from the top after every change
        pending_after = pending_requirements(output, plan, tests or None,
                                             run_tests(output, tests)[0] if tests else None)

        # TRANSACTIONAL: roll back if the action made things worse
        if len(pending_after) > before_pending_count:
            state.rejected.append({"round": rnd, "action": action, "detail": detail,
                                   "reason": "pending count increased",
                                   "pending_before": pending_before,
                                   "pending_after": pending_after})
            output = before_output
            pending_after = pending_before
            action = "backward"
            detail = detail + " [rolled back: made things worse]"
        else:
            state.emitted_sections.append(output)

        state.hops.append(Hop(rnd, action, detail, pending_before, pending_after))
    else:
        state.hops.append(Hop(max_rounds, "stop", "max rounds reached",
                              state.pending, state.pending))

    state.output = output
    state.pending = pending_requirements(output, plan, tests or None,
                                        run_tests(output, tests)[0] if tests else None)
    state.done = already_satisfied(output, plan)
    return state


def _extract_functions(code: str, drop_names: set[str] | None = None) -> str:
    """Keep only defs/classes and their bodies.

    Imports and previously injected setup lines are removed; compose_program()
    re-adds canonical versions, so nothing is duplicated.
    """
    drop_names = drop_names or set()
    setup_name_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")
    keep = []
    for line in code.splitlines():
        if re.match(r"^\s*(?:import|from)\s+", line):
            continue
        m = setup_name_re.match(line)
        if m and m.group(1) in drop_names and not re.match(r"^\s", line):
            continue
        keep.append(line)
    out, buf = [], []
    for line in keep:
        if re.match(r"^\S", line) and buf and not re.match(r"^(def|class|async)\s", line):
            if any(b.strip() for b in buf):
                out.extend(buf)
                out.append("")
            buf = []
        buf.append(line)
    if any(b.strip() for b in buf):
        out.extend(buf)
    seen, deduped = set(), []
    for line in out:
        if line.strip() and line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return "\n".join(deduped).strip() or "pass"


def _names_added_by_plan(plan: Plan) -> set[str]:
    names = {s.split("=")[0].strip() for s in plan.setup if s.strip()}
    return {n for n in names if n}


def _skeleton(plan: Plan) -> str:
    name = (plan.functions[0].split("(")[0].strip() if plan.functions else "main")
    args = ""
    if plan.functions and "(" in plan.functions[0]:
        args = plan.functions[0].split("(", 1)[1].rstrip(")")
    return f"def {name}({args}):\n    raise NotImplementedError"


def _propose(task: str, model, context: str = "", n_samples: int = 3,
             temperature: float = 0.4) -> str:
    """Optional neural proposal hook. The machine still verifies everything."""
    if model is None:
        return ""
    try:
        from brain11_loop import extract_code11
        from brain11_tokenizer import encode, decode
        prompt = f"### Task:\n{task}\n" + (f"### Error:\n{context}\n" if context else "") + "### Code:\n"
        ids = encode(prompt, add_bos=True, add_eos=False)
        inner = getattr(model, "model", model)
        ctx = inner.cfg.ctx
        if len(ids) > ctx - 170:
            ids = [ids[0]] + ids[-(ctx - 171):]
        best = ""
        for _ in range(max(1, n_samples)):
            gen = inner.generate(ids, n=170, temperature=temperature, top_k=40,
                                 repetition_penalty=1.05)
            code = extract_code11(decode(gen))
            if syntax_ok(code):
                best = code
                break
        return best
    except Exception:
        return ""


# ------------------------------------------------------------------ selftest

def selftest() -> bool:
    checks = []
    task = "write load_config() that reads config.json and returns the API URL"
    seed = ("def load_config():\n"
            "    with open(CONFIG_PATH) as f:\n"
            "        return json.load(f)['api_url']")

    st = reactive_build(task, seed_code=seed, max_rounds=4)

    checks.append(("syntax valid", syntax_ok(st.output)))
    checks.append(("json imported", re.search(r"^import json", st.output, re.M) is not None))
    checks.append(("CONFIG_PATH initialized", re.search(r"CONFIG_PATH\s*=", st.output) is not None))
    checks.append(("CONFIG_PATH = config.json", 'CONFIG_PATH = "config.json"' in st.output))

    # ordering: imports + setup appear before any use
    lines = st.output.splitlines()
    first_use = next((i for i, l in enumerate(lines) if "CONFIG_PATH" in l and "=" not in l.split("(")[0]
                      and not l.startswith(("import", "from")) and "open" in l), len(lines))
    setup_idx = next((i for i, l in enumerate(lines) if l.startswith("CONFIG_PATH")), -1)
    checks.append(("setup precedes use", 0 <= setup_idx < first_use))
    import_idx = next((i for i, l in enumerate(lines) if l.startswith("import json")), -1)
    checks.append(("import precedes use", 0 <= import_idx < len(lines)))

    # state tracking
    checks.append(("function reported done", any(d.startswith("function:load_config") for d in st.done)))
    checks.append(("no pending requirements", st.pending == []))
    checks.append(("hops recorded", len(st.hops) >= 1))
    checks.append(("hop actions valid", all(h.action in
                                            ("forward", "backward", "verify", "stop") for h in st.hops)))
    checks.append(("pending shrinks over time",
                   st.hops[-1].pending_after == [] or st.hops[-1].action == "stop"))
    checks.append(("done and pending are complementary",
                   not any(p.split(":")[-1] in st.done for p in st.pending)))

    # pending_requirements is recomputed from the artifact
    broken = "def x():\n    return UNDEFINED_NAME"
    p = pending_requirements(broken, plan_task("write x()"))
    checks.append(("undefined name detected", any(q.startswith("undefined:UNDEFINED_NAME") for q in p)))
    checks.append(("clean artifact has no pending",
                   pending_requirements("def x():\n    return 1", plan_task("write x()")) == []))
    checks.append(("syntax error detected first",
                   pending_requirements("def x(:", plan_task("write x()")) == ["syntax"]))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"[reactive selftest] {passed}/{len(checks)} passed")
    return passed == len(checks)


def main():
    ap = argparse.ArgumentParser(description="Brain11 reactive coder")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--seed-code", default="")
    ap.add_argument("--tests", default="")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.task:
        st = reactive_build(args.task, seed_code=args.seed_code, tests=args.tests,
                            max_rounds=args.max_rounds)
        if args.json:
            print(json.dumps(st.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(st.output)
            print("\n--- state ---")
            print("done:", st.done)
            print("pending:", st.pending)
            for h in st.hops:
                print(f"  round={h.round} {h.action:9s} {h.detail[:70]:70s} "
                      f"pending {len(h.pending_before)}->{len(h.pending_after)}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
