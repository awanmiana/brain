#!/usr/bin/env python3
"""Generate language-specific coding corpora.

Writes `corpus_lang_python.txt`, `corpus_lang_javascript.txt`,
`corpus_lang_cplusplus.txt` and `corpus_lang.txt` (all combined):
syntax patterns, idioms, and language-specific concepts for the
1M language-specific coding variant.
"""

import random
import sys
from pathlib import Path

PYTHON = [
    "Python idioms: use list comprehensions instead of loops when building lists — "
    "`[f(x) for x in items]`; use `for` + `enumerate` when you need the index; use "
    "`with open(path) as f` so files close automatically.",
    "Python type hints: annotate functions — `def sum_nums(nums: list[int]) -> int:` — improves clarity "
    "and tooling support.",
    "Python dataclasses: `@dataclass` generates `__init__`, `__repr__` and `__eq__` for you — declare the "
    "fields as class attributes.",
    "Python context managers: the `with` statement guarantees cleanup on success or exception — for files, "
    "locks, and connections.",
    "Python generators: `yield` turns a function into an iterator — lazy evaluation for large sequences.",
    "Python conventions: PEP 8 — snake_case names, four-space indent, docstrings in triple quotes, "
    "`if __name__ == '__main__':` entry point.",
    "Python specifics: `dict` preserves insertion order; `set` for membership tests; "
    "`try/except/finally` for error handling with cleanup.",
    "Python async: `async def` and `await` for I/O-bound code — cooperative concurrency.",
    "Python standard library: `json`, `re`, `collections` (deque, Counter), `functools` (lru_cache, "
    "partial), `itertools` (islice, chain) cover most everyday needs.",
]

JAVASCRIPT = [
    "JavaScript async/await: `async function` + `await` makes async code read like sync — the foundation "
    "of modern Node.js code.",
    "JavaScript closures: a function that captures variables from its enclosing scope — the classic "
    "pattern for event handlers and callbacks.",
    "JavaScript modules: `export`/`import` for module systems; `const` for values, `let` for reassignment; "
    "template literals with `${}` for interpolation.",
    "JavaScript DOM: query elements with `document.querySelector`, attach listeners with `addEventListener` "
    "— the basis of browser apps.",
    "JavaScript typing: dynamic typing — check types with `typeof`/`Array.isArray`; use TypeScript where "
    "static types help.",
    "JavaScript conventions: prefer `const` over `let`; avoid global variables; use arrow functions for "
    "short callbacks; prefer `for...of` over index loops when the index is not needed.",
    "JavaScript specifics: `Promise.all` for concurrent awaits; spread syntax `{...a}` and `[...a]`; "
    "`Array.prototype.map/filter/reduce` for data pipelines.",
]

CPP = [
    "C++ memory: `new`/`delete` manual memory — or prefer `std::vector`, `std::unique_ptr`, "
    "`std::shared_ptr` for automatic management.",
    "C++ RAII: resource acquisition is initialization — constructors acquire, destructors release — e.g., "
    "`std::lock_guard<std::mutex>` for mutex locking.",
    "C++ templates: generic programming with `template <typename T>` — used by the STL containers; "
    "concepts and SFINAE are advanced topics.",
    "C++ move semantics: `std::move` transfers ownership of a resource from a temporary to avoid copies.",
    "C++ conventions: pass heavy arguments by const reference; use `const` for values that don't change; "
    "compile-time checks with `static_assert`.",
    "C++ specifics: `std::vector` and `std::string` are the workhorses; ranges (`std::views`) simplify "
    "pipelines; structured bindings (`auto [a, b] = ...`) unpack tuples and pairs.",
    "C++ 11+ concurrency: `std::thread`, `std::mutex`, `std::atomic`; `std::async`/`std::future` for task "
    "results.",
    "C++ build: `CMakeLists.txt` is the standard build file; targets with `add_executable` and "
    "`add_library`; link libraries with `target_link_libraries`.",
]

LANGS = [("python", PYTHON), ("javascript", JAVASCRIPT), ("cplusplus", CPP)]


def main():
    rng = random.Random(20261133)
    all_text = []
    for lang, patterns in LANGS:
        chunks = []
        for pat in patterns:
            chunks.append(pat + "\n\n")
        text = "\n".join(chunks)
        out = f"corpus_lang_{lang}.txt"
        Path(out).write_text(text)
        print(f"wrote {out}: {len(text)} chars")
        all_text.append(text)
    combined = "\n\n".join(all_text)
    Path("corpus_lang.txt").write_text(combined)
    print(f"wrote corpus_lang.txt: {len(combined)} chars")


if __name__ == "__main__":
    main()
