#!/usr/bin/env python3
"""Brain11 Planner — dependency-aware, prerequisite-first code generation.

The idea: before emitting code, derive what must exist first (imports, constants,
environment variables, helper functions, initialization), order those
dependencies, and only then emit the function and its call site.

This is not a new neural architecture. It is a plan -> generate -> dataflow
check -> repair loop around the 10.8M coder. It catches the small-model failure
mode where a function references CONFIG_PATH / API_KEY / json before anything
defines or imports it.

Usage:
    python3 brain11_planner.py --plan "write load_config() that reads config.json and returns the API URL"
    python3 brain11_planner.py --heal "write load_config() that reads config.json" --code "def load_config():\n    return json.load(open(CONFIG_PATH))"
    python3 brain11_planner.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUILTINS = set(dir(builtins))
ENV_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
PATH_RE = re.compile(r"[\w./-]+\.(?:json|csv|txt|md|yaml|yml|log|ini|toml|py)")
FUNC_RE = re.compile(
    r"(?:write|create|implement|define|make)\s+(?:a\s+)?(?:function|method)\s+"
    r"([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?", re.I,
)
FUNC_RE2 = re.compile(
    r"(?:write|create|implement|define|make)\s+(?:a\s+)?([A-Za-z_]\w*)\s*\(([^)]*)\)",
    re.I,
)
DEF_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")


@dataclass
class Plan:
    goal: str
    language: str = "python"
    imports: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


def plan_task(task: str) -> Plan:
    low = task.lower()
    imports, setup, risks = [], [], []

    if ".json" in low or "json" in low:
        imports.append("json")
    if ".csv" in low or "csv" in low:
        imports.append("csv")
    if "environment" in low or "env var" in low or "api key" in low or "secret" in low:
        if "os" not in imports:
            imports.append("os")
    if "regex" in low or "regular expression" in low:
        imports.append("re")
    if "math" in low:
        imports.append("math")
    if "date" in low or "time" in low:
        imports.append("datetime")

    paths = PATH_RE.findall(task)
    for path in dict.fromkeys(paths):
        base = path.split("/")[-1].upper().replace(".", "_").replace("-", "_")
        if "config" in path.lower():
            name = "CONFIG_PATH"
        elif "data" in path.lower() or path.endswith(".csv"):
            name = "DATA_PATH"
        else:
            name = f"{base}_PATH"
        setup.append(f'{name} = "{path}"')

    for env in ENV_RE.findall(task):
        if env not in {"API", "URL", "JSON", "CSV", "PYTHON"} and env not in setup:
            if "os" not in imports:
                imports.append("os")
            setup.append(f'{env} = os.environ.get("{env}")')
            risks.append(f"{env} may be unset at runtime")

    m = FUNC_RE.search(task) or FUNC_RE2.search(task) or DEF_RE.search(task)
    functions = []
    if m:
        name = m.group(1)
        args = [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
        functions.append(f"{name}({', '.join(args)})")
    else:
        if any(w in low for w in ("print", "run", "execute")):
            functions.append("main()")

    steps = []
    if imports:
        steps.append("import " + ", ".join(imports))
    if setup:
        steps.append("initialize constants before any function call")
    if functions:
        steps.append("define " + ", ".join(functions))
    if any(w in low for w in ("print", "run", "execute", "call")):
        steps.append("invoke main() under if __name__ == '__main__'")
    steps.append("verify with tests / run the program")

    if any(".json" in p for p in paths):
        risks.append("file may be missing or contain invalid JSON")
    if "divide" in low or "/" in low:
        risks.append("division by zero")
    return Plan(goal=task, imports=list(dict.fromkeys(imports)), setup=setup,
                functions=functions, steps=steps, risks=risks)


def render_plan(plan: Plan) -> str:
    lines = ["### Plan", f"goal: {plan.goal}", f"language: {plan.language}"]
    lines.append("imports: " + (", ".join(plan.imports) if plan.imports else "none"))
    lines.append("setup:")
    for s in plan.setup or ["none"]:
        lines.append(f"- {s}")
    lines.append("functions: " + (", ".join(plan.functions) if plan.functions else "none"))
    lines.append("order:")
    for i, s in enumerate(plan.steps, 1):
        lines.append(f"{i}. {s}")
    if plan.risks:
        lines.append("risks: " + "; ".join(plan.risks))
    return "\n".join(lines)


# ------------------------------------------------------------------ dataflow

def _target_names(node) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    return names


def _module_defs(node) -> set[str]:
    names = set()
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            names.add(a.asname or a.name.split(".")[0])
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        names |= _target_names(node)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        names |= _target_names(node.target)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars:
                names |= _target_names(item.optional_vars)
    return names


class _Loads(ast.NodeVisitor):
    def __init__(self, skip_function_bodies: bool = True):
        self.names: list[tuple[str, int]] = []
        self.skip = skip_function_bodies

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.append((node.id, getattr(node, "lineno", 0)))

    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults:
            self.visit(d)
        if not self.skip:
            for stmt in node.body:
                self.visit(stmt)

    visit_AsyncFunctionDef = visit_FunctionDef


def dataflow_check(code: str) -> list[dict]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [{"name": "<syntax>", "line": 0, "kind": "syntax"}]
    issues = []
    defined = set(BUILTINS)
    # module-level sequential use-before-def
    for node in tree.body:
        visitor = _Loads(skip_function_bodies=True)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                visitor.visit(d)
            for d in node.args.defaults:
                visitor.visit(d)
            for d in node.args.kw_defaults:
                if d is not None:
                    visitor.visit(d)
            if node.returns:
                visitor.visit(node.returns)
        elif isinstance(node, ast.ClassDef):
            for d in node.decorator_list:
                visitor.visit(d)
            for b in node.bases:
                visitor.visit(b)
            for k in node.keywords:
                visitor.visit(k.value)
        else:
            visitor.visit(node)
        for name, line in visitor.names:
            if name not in defined:
                issues.append({"name": name, "line": line, "kind": "use-before-def"})
        defined |= _module_defs(node)

    # globals referenced inside functions but defined nowhere
    all_defined = defined | set(BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local = {a.arg for a in node.args.args + node.args.kwonlyargs}
            if node.args.vararg:
                local.add(node.args.vararg.arg)
            if node.args.kwarg:
                local.add(node.args.kwarg.arg)
            for stmt in node.body:
                local |= _module_defs(stmt)
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        local.add(sub.id)
            visitor = _Loads(skip_function_bodies=False)
            for stmt in node.body:
                visitor.visit(stmt)
            for name, line in visitor.names:
                if name not in local and name not in all_defined:
                    issues.append({"name": name, "line": line, "kind": "undefined-global"})
    # dedupe
    seen, out = set(), []
    for i in issues:
        key = (i["name"], i["kind"])
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def insert_setup(code: str, task: str, plan: Plan | None = None) -> tuple[str, list[str]]:
    plan = plan or plan_task(task)
    issues = dataflow_check(code)
    if not issues:
        return code, []
    notes = []
    imports = list(plan.imports)
    setup = list(plan.setup)

    for issue in issues:
        name = issue["name"]
        if name in {"<syntax>"}:
            continue
        if name in {"json", "csv", "os", "re", "math", "datetime", "sys", "pathlib"}:
            if name not in imports:
                imports.append(name)
            notes.append(f"added import {name}")
            continue
        if name.isupper() and (name.endswith("_PATH") or name.endswith("_URL") or name.endswith("_KEY")):
            if name.endswith("_PATH"):
                path = PATH_RE.search(task)
                value = f'"{path.group(0)}"' if path else '"config.json"'
                setup.append(f"{name} = {value}")
            elif name.endswith("_KEY"):
                if "os" not in imports:
                    imports.append("os")
                setup.append(f'{name} = os.environ.get("{name}")')
            else:
                setup.append(f'{name} = ""')
            notes.append(f"initialized {name}")
            continue
        # unknown global: use a safe placeholder, never invent a dangerous value
        setup.append(f"{name} = None  # TODO: initialize before use")
        notes.append(f"placeholder for {name}")

    lines = code.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import|from)\s+", line):
            insert_at = i + 1
    prefix = [f"import {m}" for m in dict.fromkeys(imports)]
    prefix += [s for s in dict.fromkeys(setup) if s not in lines]
    lines = lines[:insert_at] + prefix + lines[insert_at:]
    return "\n".join(lines).strip() + "\n", notes


def compose_program(plan: Plan, function_code: str, call_code: str = "") -> str:
    """Order the program: imports -> constants/setup -> functions -> call."""
    function_code = textwrap.dedent(function_code).strip()
    lines = [f"import {m}" for m in plan.imports]
    if lines:
        lines.append("")
    lines += plan.setup
    if plan.setup:
        lines.append("")
    lines.append(function_code)
    if call_code:
        lines += ["", textwrap.dedent(call_code).strip()]
    elif plan.functions and any("main()" in f for f in plan.functions):
        lines += ["", "", "if __name__ == '__main__':", "    main()"]
    return "\n".join(lines).strip() + "\n"


def heal_code(code: str, task: str) -> dict:
    plan = plan_task(task)
    healed, notes = insert_setup(code, task, plan)
    return {"code": healed, "notes": notes, "plan": asdict(plan),
            "issues": dataflow_check(code)}


# ------------------------------------------------------------------ selftest

def selftest() -> bool:
    checks = []
    task = "write load_config() that reads config.json and returns the API URL"
    plan = plan_task(task)
    checks.append(("plan imports json", "json" in plan.imports))
    checks.append(("plan finds CONFIG_PATH", any("CONFIG_PATH" in s for s in plan.setup)))
    checks.append(("plan finds load_config", any(f.startswith("load_config(") for f in plan.functions)))

    code = "def load_config():\n    with open(CONFIG_PATH) as f:\n        return json.load(f)[\"api_url\"]"
    issues = dataflow_check(code)
    checks.append(("dataflow catches CONFIG_PATH", any(i["name"] == "CONFIG_PATH" for i in issues)))
    checks.append(("dataflow catches json", any(i["name"] == "json" for i in issues)))

    healed = heal_code(code, task)
    checks.append(("heal adds json import", "import json" in healed["code"]))
    checks.append(("heal initializes CONFIG_PATH", "CONFIG_PATH =" in healed["code"]))
    try:
        ast.parse(healed["code"])
        checks.append(("healed code parses", True))
    except SyntaxError:
        checks.append(("healed code parses", False))

    ordered = compose_program(plan, "def load_config():\n    return {}", "print(load_config())")
    checks.append(("compose keeps imports first", ordered.startswith("import json")))
    checks.append(("compose puts call last", ordered.rstrip().endswith("print(load_config())")))

    env = heal_code("def send():\n    return API_KEY", "send a message using API_KEY")
    checks.append(("heal initializes API_KEY from env", "API_KEY = os.environ.get" in env["code"]))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"[planner selftest] {passed}/{len(checks)} passed")
    return passed == len(checks)


def main():
    ap = argparse.ArgumentParser(description="Brain11 dependency-aware planner")
    ap.add_argument("--plan", metavar="TASK")
    ap.add_argument("--heal", metavar="TASK")
    ap.add_argument("--code", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.plan:
        p = plan_task(args.plan)
        print(json.dumps(asdict(p), indent=2, ensure_ascii=False))
        print("\n" + render_plan(p))
        return
    if args.heal:
        print(json.dumps(heal_code(args.code, args.heal), indent=2, ensure_ascii=False))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
