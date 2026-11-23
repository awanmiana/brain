# Brain AI — Multi-Layer Agentic Architecture (design)

Design for a multi-layer agentic model using the Brain AI model
(`coder_1m_moe.py` + `coder_1m_moe_weights.json`): the per-purpose 1M variants are
the layer specialists, and the Brain AI model (Mixture of Experts) is the shared
orchestrator brain that each layer consults.

> Status: design document. The Brain AI model is trained (2,891,052 params); the
> SaaS app (`saas_backend.py`, `dashboard.html`, `test_saas.py`) is written and
> tested. The layers below are not yet individually implemented — next step is
> `agentic_layers.py`.

## The layers

1. **Perception / parsing layer** (`coder_conversation_weights.json`)
   - Input parsing: user text → tokens; intent detection ("fix", "explain",
     "deploy"); entity extraction; language detection; tone.
   - Safety screening (deny / allow / ask).

2. **Planning layer** (`coder_logic_weights.json`)
   - Break the goal into steps; choose tools and their order; revise the plan on
     failure; human-in-the-loop checkpoints when the goal is ambiguous or risky.
   - Deep-analysis orientation: uses the logic/programming concepts corpus
     (`corpus_logic.txt`) — proof techniques, invariants, verification reasoning.

3. **Execution / tool-use layer** (`coder_tools_weights.json`, `coder_lang_weights.json`)
   - Structured tool-call formats — JSON / API-style / YAML / XML
     (`corpus_tools.txt` has examples).
   - Calls the real tools: files (read/write), shell (`shell`), the Brain AI model
     (`BrainMoE.generate`), the SaaS backend (`POST /generate`), OpenCode sessions
     (`opencode_server.py`).
   - Parses tool results back into the perception step.

4. **Verification / self-check layer** (`coder_debug_weights.json`)
   - Re-check the goal against the output; run tests (`test_saas.py`, project tests).
   - Debug on failure; retry with backoff; report discrepancies; update state.

5. **Memory / state layer**
   - Session state, facts, preferences, artifacts — backed by files/notes.
   - State handoff between sessions (pattern: `PROGRESS.md` / `NEXT.md`).

## The Brain AI model — the shared brain

`BrainMoE` (`coder_1m_moe.py`): 2,891,052 params, 4 experts per block, top-2
mixture of experts, load-balancing aux loss (NLL + 0.01 * load-balance term).
Trained on the combined corpora: `combined_corpus()` = repo code +
`corpus_debug.txt` + `corpus_conversation.txt` + `corpus_tools.txt` +
`corpus_logic.txt` + `corpus_lang.txt`.

Used by: the SaaS backend (`saas_backend.py` — `/health`, `/models`, `/generate`),
the dashboard, and the orchestration brain of every layer.

## Data flow

```
user input
  -> 1. Perception
  -> 2. Planning
  -> 3. Execution  [tool calls; BrainMoE as the brain of each tool]
  -> 4. Verification
  -> 5. Memory
  -> reply (1)
```

## Integration points (how the layers touch the real world)

- **Tools:** files (read/write), shell (`shell`), the Brain AI model, the SaaS
  backend endpoints, OpenCode sessions.
- **State:** JSON/notes files in this repo.
- **Testing:** `test_saas.py` — backend health/models/generate + dashboard + 404;
  each layer gets its own small test plan (e.g., the planning layer test: given
  goal X, the plan must contain steps 1..N).

## How the variants map to the layers (the "multi layer approach")

Each specialist is a dedicated 1M model, trained on purpose:
- `coder_debug_weights.json` → Verification (debugging/troubleshooting).
- `coder_conversation_weights.json` → Perception (dialogue/parsing).
- `coder_tools_weights.json` → Execution (tool calling/formatting).
- `coder_logic_weights.json` → Planning (logic/programming concepts).
- `coder_lang_weights.json` → Execution (language-specific coding — Python, JS, C++).
- `coder_1m_weights.json` ("coderbrain") → the original general coding LM.
- `BrainMoE` (`coder_1m_moe_weights.json`) → the orchestrator brain.

## Honesty notes

- Layers are a design; the orchestrator (`agentic_layers.py`) is the next
  implementation step. Each layer is a byte-level 1M model — modest outputs today;
  the verification layer is the guardrail for that.
- The "complete backend" is the SaaS backend (`saas_backend.py`) — health, models,
  generate, dashboard — the extension point for more endpoints.
