# TinyCoder-1M — using it with coding agents

TinyCoder-1M is a ~1M-parameter byte-level coding LM: `coder_1m.py` +
`coder_1m_weights.json` (canonical: the resumed 160-step checkpoint, val PPL ~10).
`opencode_server.py` exposes it over a local OpenAI-compatible endpoint so
OpenCode can use it as a custom provider.

Honest limits: it is a 1M-param model — expect weak, snippet-level output
(repo micro-bench: 0/3 exact matches, avg char accuracy 0.306 vs held-out text).
It works well for drafting small snippets and teaching snippets. For real
coding work use a stronger model, or combine this one with an agent layer
that applies edits.

## Files

- `coder_1m.py` — the model: 3-layer pre-LN GPT, d=162, 2 heads, tied byte head.
- `coder_1m_weights.json` — canonical weights (resumed 160-step run).
- `opencode_server.py` — the local HTTP shim (OpenAI-compatible).
- `coder_1m_report.txt`, `PROGRESS.md` — eval report and progress log.

## 1. Give it a prompt (CLI)

```sh
python3 coder_1m.py --generate --prompt "def fizzbuzz(n):" --n 30
```

- `--prompt` — starting text; `--n` — number of tokens to generate (default 20).
- Greedy generation at the byte level; run from the repo dir.
- Today, on real prompts it mostly emits whitespace (e.g. `def fizzbuzz(n):` -> blanks).

Python API:

```python
from coder_1m import Model, BOS
m = Model()
ids = [BOS] + list(b"def fizzbuzz(n):")
print(bytes(t for t in m.generate(ids, n=30, greedy=True) if t < 256).decode("utf-8", "replace"))
```

## 2. Use it in OpenCode

Start the server:

```sh
python3 -u opencode_server.py            # serves http://127.0.0.1:8765/v1
```

Options: `--port 8765` (default), `--max-tokens 64` (server-side cap on
generated tokens), `--weights coder_1m_weights.json` (default).

Then configure OpenCode — `opencode.jsonc` in the project or global
`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "coderbrain": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8765/v1" },
      "models": { "coderbrain": {} }
    }
  },
  "model": "coderbrain/coderbrain"
}
```

- Start the server before opening OpenCode. Pick the model with `/models`, or
  make it the default via `"model"`.
- If OpenCode complains about missing credentials for the custom provider:
  set a dummy value and add `"env": ["TINYCODER_API_KEY"]` to the provider
  (the local server does not check credentials).
- Performance: pure Python, 1M params — keep `max_tokens` modest (server
  default 64).

## 3. Test the endpoint (no OpenCode needed)

```sh
curl -s http://127.0.0.1:8765/v1/models
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"coderbrain","messages":[{"role":"user","content":"def fizzbuzz(n):"}],"max_tokens":16}'
```

Expect a `chat.completion` JSON with the continuation in
`choices[0].message.content` (`finish_reason`: "stop"/"length").

## 4. Honest notes for using it as an agent

- Text-only: TinyCoder-1M generates code text. OpenCode's tools (read/edit/
  write/shell) are the agent layer; the model cannot call them. Expect: it
  produces a snippet — you or the agent apply it.
- Context: the model has a 64-token context; the server truncates prompts to
  the last 64 tokens.
- Speed: every generated token is a forward pass in pure Python — small
  `max_tokens` keeps requests snappy.
- To improve the model further: `python3 coder_1m.py --train --steps 160 --resume`
  (see `PROGRESS.md`).
