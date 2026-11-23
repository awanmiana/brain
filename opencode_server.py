#!/usr/bin/env python3
"""TinyCoder-1M as a local OpenAI-compatible chat endpoint for OpenCode.

Serves the ~1M-parameter byte-level coding LM from `coder_1m.py` behind a
minimal `/v1/chat/completions` + `/v1/models` implementation, so OpenCode
can use it as a custom provider:

    providers.<id>.package = @opencode/ai/providers/openai-compatible
    providers.<id>.settings.baseURL = http://127.0.0.1:8765/v1

Usage:
    python3 opencode_server.py [--port 8765] [--max-tokens 64]
                               [--weights coder_coding_weights.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coder_1m import BOS, EOS, Model, load_weights

MODEL_ID = "coderbrain"
DEFAULT_PORT = 8765
DEFAULT_MAX_TOKENS = 64

_args = None
_model = None
_model_lock = threading.Lock()
_gen_lock = threading.Lock()


def load_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model = Model()
                load_weights(model, str(_args.weights))
                _model = model
    return _model


def flatten_messages(messages):
    """Flatten an OpenAI chat message list into one text prompt."""
    parts = []
    for m in messages or []:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            continue
        role = m.get("role", "user")
        parts.append(content if role == "assistant" else f"{role}: {content}")
    return "\n\n".join(parts)


def start_ids(prompt_text):
    """Prompt ids: BOS + prompt bytes, truncated to the model's context."""
    model = load_model()
    ids = [BOS] + list(prompt_text.encode("utf-8", "replace"))
    if len(ids) > model.t:
        ids = ids[-model.t:]
    return ids


def gen_text(prompt_text, n):
    """(text, finish_reason) for the greedy byte-level continuation."""
    model = load_model()
    ids = start_ids(prompt_text)
    out = []
    finish = "length"
    with _gen_lock:
        for _ in range(n):
            logits, _ = model.forward(ids)
            last = logits[-1]
            nxt = max(range(model.v), key=lambda i: last[i])
            if nxt == EOS:
                finish = "stop"
                break
            ids.append(nxt)
            if nxt < 256:
                out.append(nxt)
    return bytes(out).decode("utf-8", "replace"), finish


def gen_stream(prompt_text, n, state):
    """Yield decoded byte pieces while generating; set state['finish']."""
    model = load_model()
    ids = start_ids(prompt_text)
    state["finish"] = "length"
    with _gen_lock:
        for _ in range(n):
            logits, _ = model.forward(ids)
            last = logits[-1]
            nxt = max(range(model.v), key=lambda i: last[i])
            if nxt == EOS:
                state["finish"] = "stop"
                break
            ids.append(nxt)
            if nxt < 256:
                yield bytes([nxt]).decode("utf-8", "replace")


def _chunk(model_id, delta, finish=None):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[server] " + (fmt % args) + "\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tinycoder-local",
                }],
            })
        else:
            self._send_json(
                {"error": {"message": "not found", "type": "invalid_request_error"}},
                status=404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(
                {"error": {"message": "not found", "type": "invalid_request_error"}},
                status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}},
                status=400)
            return
        messages = req.get("messages") or []
        if not messages:
            self._send_json(
                {"error": {"message": "messages are required", "type": "invalid_request_error"}},
                status=400)
            return
        prompt_text = flatten_messages(messages)
        req_max = req.get("max_tokens") or req.get("max_completion_tokens") or 0
        if not isinstance(req_max, int):
            req_max = 0
        n = min(req_max, _args.max_tokens) if req_max else _args.max_tokens
        req_model = req.get("model") or MODEL_ID

        if req.get("stream"):
            self._send_stream(req_model, prompt_text, n)
        else:
            try:
                text, finish = gen_text(prompt_text, n)
            except Exception as exc:
                self._send_json(
                    {"error": {"message": f"generation failed: {exc}", "type": "server_error"}},
                    status=500)
                return
            completion_bytes = len(text.encode("utf-8", "replace"))
            self._send_json({
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": len(start_ids(prompt_text)),
                    "completion_tokens": completion_bytes,
                    "total_tokens": len(start_ids(prompt_text)) + completion_bytes,
                },
            })

    def _send_stream(self, req_model, prompt_text, n):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        state = {}
        self.wfile.write(_sse(_chunk(req_model, {"role": "assistant"})))
        self.wfile.flush()
        for piece in gen_stream(prompt_text, n, state):
            self.wfile.write(_sse(_chunk(req_model, {"content": piece})))
            self.wfile.flush()
        self.wfile.write(_sse(_chunk(req_model, {}, state.get("finish", "length"))))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global _args
    ap = argparse.ArgumentParser(
        description="Serve TinyCoder-1M as an OpenAI-compatible local endpoint.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port to listen on (default {DEFAULT_PORT})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"server-side cap on generated tokens (default {DEFAULT_MAX_TOKENS})")
    ap.add_argument("--weights", default=str(HERE / "coder_coding_weights.json"),
                    help="path to the canonical weights JSON")
    args = ap.parse_args()
    _args = args
    if not Path(args.weights).exists():
        print(f"error: weights not found: {args.weights}", file=sys.stderr)
        sys.exit(1)
    load_model()  # fail fast on a bad weights file
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"TinyCoder-1M server on http://127.0.0.1:{args.port}/v1 "
          f"(model {MODEL_ID}, max-tokens {args.max_tokens})")
    print(f"weights: {args.weights}")
    server.serve_forever()


if __name__ == "__main__":
    main()
