#!/usr/bin/env python3
"""SaaS backend — the Brain AI model (Mixture of Experts) behind HTTP endpoints.

Loads `BrainMoE` (coder_1m_moe.py) + weights from `coder_1m_moe_weights.json`.

Endpoints:
    GET  /health               -> {"status": "ok", "model": "brain"}
    GET  /models               -> BrainMoE info (params, experts, topk, layers, ...)
    GET  /v1/models            -> OpenAI-compatible: [{"id": "brain", ...}]
    POST /generate             -> {"prompt": "...", "n": 20} -> {"text": ..., ...}
    POST /v1/chat/completions  -> OpenAI-compatible (non-stream + stream)
    GET  /                     -> serves the professional dashboard (dashboard.html)

Usage:
    python3 saas_backend.py            # serves http://127.0.0.1:8123
"""

import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coder_1m_moe import BrainMoE, load_weights, BOS, EOS

WEIGHTS = HERE / "coder_1m_moe_weights.json"
DASHBOARD = HERE / "dashboard.html"
HOST, PORT = "127.0.0.1", 8123
MODEL_ID = "brain"
MAX_TOKENS = 48  # server-side cap on generation length (Brain MoE is small/slow-ish on CPU)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = BrainMoE()
        if WEIGHTS.exists():
            load_weights(_model, str(WEIGHTS))
        _model.eval()
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


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _chunk(model_id, delta, finish=None):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[saas] " + (fmt % args) + "\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            self._send_json({"status": "ok", "model": MODEL_ID})
        elif path == "/models":
            m = get_model()
            blk = m.blocks[0]
            self._send_json({
                "model": MODEL_ID,
                "params": sum(p.numel() for p in m.parameters()),
                "experts": m.n_experts,
                "topk": blk.topk,
                "layers": m.L,
                "dim": m.d,
                "vocab": m.v,
                "weights_file": str(WEIGHTS),
            })
        elif path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "brain-ai",
                }],
            })
        elif path in ("/", "/dashboard"):
            body = DASHBOARD.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/generate":
            prompt = str(req.get("prompt") or "")
            n = int(req.get("n") or 20)
            m = get_model()
            ids = [BOS] + list(prompt.encode("utf-8", "replace"))
            gen = m.generate(ids, n=n, greedy=True)
            text = bytes(t for t in gen if t < 256).decode("utf-8", "replace")
            finish = "length" if (gen and len(gen) >= n and gen[-1] != EOS) else "stop"
            self._send_json({"text": text, "finish_reason": finish, "model": MODEL_ID})
            return

        if path == "/v1/chat/completions":
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
            n = min(req_max, MAX_TOKENS) if req_max else MAX_TOKENS
            req_model = req.get("model") or MODEL_ID
            m = get_model()
            ids = [BOS] + list(prompt_text.encode("utf-8", "replace"))
            if len(ids) > m.t:
                ids = ids[-m.t:]

            if req.get("stream"):
                self._send_stream(m, req_model, ids, n)
            else:
                gen = m.generate(ids, n=n, greedy=True)
                text = bytes(t for t in gen if t < 256).decode("utf-8", "replace")
                finish = "length" if (gen and len(gen) >= n and gen[-1] != EOS) else "stop"
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
                        "prompt_tokens": len(ids),
                        "completion_tokens": len(gen),
                        "total_tokens": len(ids) + len(gen),
                    },
                })
            return

        self._send_json({"error": "not found"}, status=404)

    def _send_stream(self, model, req_model, ids, n):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with torch.no_grad():
            self.wfile.write(_sse(_chunk(req_model, {"role": "assistant"})))
            self.wfile.flush()
            out = []
            for _ in range(n):
                logits = model.forward(ids)
                nxt = int(torch.argmax(logits[-1]).item())
                if nxt == EOS:
                    break
                ids.append(nxt)
                if nxt < 256:
                    piece = bytes([nxt]).decode("utf-8", "replace")
                    out.append(nxt)
                    self.wfile.write(_sse(_chunk(req_model, {"content": piece})))
                    self.wfile.flush()
            self.wfile.write(_sse(_chunk(req_model, {}, "stop")))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


def main():
    get_model()  # fail fast on bad weights
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SaaS backend on http://{HOST}:{PORT} (Brain AI model: coder_1m_moe.py)")
    server.serve_forever()


if __name__ == "__main__":
    main()
