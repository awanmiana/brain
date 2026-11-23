#!/usr/bin/env python3
"""SaaS backend — the Brain AI model (Mixture of Experts) behind HTTP endpoints.

Loads `BrainMoE` (coder_1m_moe.py) + weights from `coder_1m_moe_weights.json`.

Endpoints:
    GET  /health     -> {"status": "ok", "model": "brain-ai-moe"}
    GET  /models     -> BrainMoE model info (params, experts, topk, layers, ...)
    POST /generate   -> {"prompt": "...", "n": 20} -> {"text": ..., "finish_reason": ...}
    GET  /           -> serves the professional dashboard (dashboard.html)

Usage:
    python3 saas_backend.py            # serves http://127.0.0.1:8123
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coder_1m_moe import BrainMoE, load_weights, BOS

WEIGHTS = HERE / "coder_1m_moe_weights.json"
DASHBOARD = HERE / "dashboard.html"
HOST, PORT = "127.0.0.1", 8123

_model = None


def get_model():
    global _model
    if _model is None:
        _model = BrainMoE()
        if WEIGHTS.exists():
            load_weights(_model, str(WEIGHTS))
        _model.eval()
    return _model


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
            self._send_json({"status": "ok", "model": "brain-ai-moe"})
        elif path == "/models":
            m = get_model()
            blk = m.blocks[0]
            self._send_json({
                "model": "brain-ai-moe",
                "params": sum(p.numel() for p in m.parameters()),
                "experts": m.n_experts,
                "topk": blk.topk,
                "layers": m.L,
                "dim": m.d,
                "vocab": m.v,
                "weights_file": str(WEIGHTS),
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
        if self.path != "/generate":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return
        prompt = str(req.get("prompt") or "")
        n = int(req.get("n") or 20)
        m = get_model()
        ids = [BOS] + list(prompt.encode("utf-8", "replace"))
        gen = m.generate(ids, n=n, greedy=True)
        text = bytes(t for t in gen if t < 256).decode("utf-8", "replace")
        finish = "length" if (gen and len(gen) >= n and gen[-1] != 257) else "stop"
        self._send_json({"text": text, "finish_reason": finish, "model": "brain-ai-moe"})


def main():
    get_model()  # fail fast on bad weights
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SaaS backend on http://{HOST}:{PORT} (Brain AI model: coder_1m_moe.py)")
    server.serve_forever()


if __name__ == "__main__":
    main()
