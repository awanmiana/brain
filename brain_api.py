#!/usr/bin/env python3
"""
Brain API — the single, unified entry point for the Brain AI model.

This is the ONE file you import or run to use the Brain. It loads the
430M MoE (`brain.py`) and provides a simple generate/chat interface.
No server required — use it directly from Python or OpenCode.

================================================================================
HOW TO USE IT
================================================================================

Direct Python (OpenCode can import this):

    from brain_api import BrainAPI

    brain = BrainAPI()  # auto-loads brain_distill_weights.json

    # Raw generation
    text = brain.generate("# task: write a function add(a, b) that returns a + b\n")
    print(text)

    # Code solving with full loop: lattice → memory → neural → verifier → repair
    result = brain.solve_code(
        task="write a function add_func(a, b) that returns the sum of a and b",
        tests="assert add_func(2, 3) == 5",
        n=8, rounds=3,
    )
    print(result["ok"], result["source"], result["code"])

CLI (no server needed):

    python3 brain_api.py --task "write a function add(a, b)" --tests "assert add(2,3)==5"
    python3 brain_api.py --prompt "# task: write a function add(a, b)\n"
    python3 brain_api.py --chat "hello"   # best-effort, not chat-tuned

No HTTP server required — it loads weights directly into memory.
If you later want an HTTP server, brain_api.py is the single file to wrap.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ------------------------------------------------------------------ Lattice manager
# Multi-domain symbolic co-processor router. Loads domain lattices
# independently from seed files. Does NOT modify pathway_lattice.py or
# pathway_seed.py.

class LatticeManagerWrapper:
    """Thin wrapper around LatticeManager for brain_api integration."""

    def __init__(self):
        from lattice_manager import get_manager
        self.mgr = get_manager()

    def check(self, task: str) -> dict | None:
        """Return an exact lattice answer dict if any domain knows it, else None."""
        return self.mgr.check(task)


# ------------------------------------------------------------------ Brain API

class BrainAPI:
    """Unified Brain interface. Loads the latest Brain checkpoint automatically."""

    def __init__(self, weights=None, device=None, use_rope=True):
        import torch
        from brain import FIT_8GB, Brain
        from bpe_tokenizer import Vocab, load_tokens

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = Path(weights) if weights else (HERE / "brain_distill_weights.json")

        self.tokens = load_tokens("brain_vocab.json")
        self.vocab = Vocab(self.tokens)
        self.model = Brain(self.tokens, use_rope=use_rope, **FIT_8GB).to(self.device)

        if self.weights.exists():
            state = torch.load(self.weights, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[brain_api] loaded <- {self.weights.name} ({self.model.n_params():,} params)", flush=True)
        else:
            self.model.eval()
            print(f"[brain_api] WARNING: weights not found at {self.weights} — using random init", flush=True)

        self.model_max_new = 160
        self.lattice = LatticeManagerWrapper()

    # ------------------------------------------------------------------ generate

    def generate(self, prompt: str, n: int = 32, temperature: float = 0.9, top_k: int = 40) -> str:
        """Generate text from a prompt string. Returns the completion text."""
        import torch
        with torch.no_grad():
            ids = self.vocab.encode(prompt)
            outs = self.model.sample(
                ids, n=1, max_new=self.model_max_new,
                temperature=temperature, top_k=top_k,
            )
            return self.vocab.decode(outs[0])

    def generate_n(self, prompt: str, n: int = 8, temperature: float = 0.9,
                   top_k: int = 40) -> list[str]:
        """Generate n candidate completions from a prompt."""
        import torch
        with torch.no_grad():
            ids = self.vocab.encode(prompt)
            outs = self.model.sample(
                ids, n=n, max_new=self.model_max_new,
                temperature=temperature, top_k=top_k,
            )
            return [self.vocab.decode(o) for o in outs]

    # ------------------------------------------------------------------ chat (best-effort)

    def chat(self, message: str, system: str | None = None) -> str:
        """Conversational chat. NOTE: Brain is a code model, not chat-tuned."""
        prompt = ""
        if system:
            prompt += f"system: {system}\n"
        prompt += f"user: {message}\nassistant: "
        return self.generate(prompt, n=48, temperature=0.8)

    # ------------------------------------------------------------------ code solving (with verifier + lattice)

    def solve_code(self, task: str, tests: str | None = None,
                   n: int = 8, rounds: int = 3, use_memory: bool = True,
                   use_lattice: bool = True) -> dict:
        """
        Solve a coding task using the full Brain loop:

          lattice (exact) -> memory/analogy -> neural generation
          -> verifier -> repair -> memory store

        The lattice is checked first for exact arithmetic/facts. If it answers,
        we return immediately with source="lattice:*". Otherwise we fall through
        to memory/analogy, then neural generation with verifier + repair.

        Returns {"ok": bool, "code": str, "source": str, "attempts": int, ...}
        """
        from brain_cortex import (
            Memory, Verifier, Generator, Cortex,
            build_prompt, extract_code, memory_solve, requested_name,
        )

        # --- Step 1: Lattice exact check (symbolic co-processor) ---
        if use_lattice:
            lattice_result = self.lattice.check(task)
            if lattice_result is not None:
                # Lattice answered exactly — verify it if tests are given
                if tests:
                    v = Verifier()
                    vr = v.verify(lattice_result["code"], tests)
                    if vr["passed"]:
                        return {**lattice_result, "attempts": 1, "one_shot": True,
                                "best_of_n": True, "history": []}
                    # Lattice answer failed tests — fall through to neural
                else:
                    return {**lattice_result, "attempts": 1, "one_shot": True,
                            "best_of_n": True, "history": []}

        # --- Step 2: Memory/analogy -> neural -> verifier -> repair ---
        memory = Memory(corpus_paths=[HERE / "corpus_teacher.txt"])
        verifier = Verifier()
        generator = Generator(self.model, self.vocab, max_new=self.model_max_new)
        cortex = Cortex(generator, memory, verifier, max_new=self.model_max_new)

        return cortex.solve(
            task, tests=tests, n=n, rounds=rounds,
            store=False, use_memory=use_memory,
        )


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description="Brain API — unified entry point.")
    ap.add_argument("--task", help="coding task description")
    ap.add_argument("--tests", help="inline test assertions")
    ap.add_argument("--chat", help="conversational message (best-effort)")
    ap.add_argument("--prompt", help="raw generation prompt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-memory", action="store_true", help="disable memory/analogy path")
    ap.add_argument("--no-lattice", action="store_true", help="disable lattice exact check")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    brain = BrainAPI(weights=args.weights)

    if args.chat:
        print(brain.chat(args.chat))
        return

    if args.prompt:
        print(brain.generate(args.prompt, n=args.n))
        return

    if args.task:
        result = brain.solve_code(
            args.task, tests=args.tests, n=args.n, rounds=args.rounds,
            use_memory=not args.no_memory,
            use_lattice=not args.no_lattice,
        )
        print(json.dumps({
            "ok": result["ok"],
            "source": result.get("source"),
            "attempts": result.get("attempts"),
            "code": result.get("code", ""),
            "error": result.get("error", ""),
        }, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
