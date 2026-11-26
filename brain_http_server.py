#!/usr/bin/env python3
"""Brain HTTP server — OpenAI-compatible endpoint for the edge Brain system.

Default backend is Brain11: three ~10.8M specialists behind one router.
    /v1/models              brain, brain11-coder, brain11-language, brain11-agentic
    /v1/chat/completions    routes by model id or task type
    /health

Use --backend brain430 to serve the older 430M MoE instead.

Usage:
    python3 brain_http_server.py --port 8765
    python3 brain_http_server.py --backend brain430 --weights brain_distill_weights.json --no-rope

OpenCode provider config (~/.config/opencode/opencode.jsonc) points at:
    http://127.0.0.1:8765/v1
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from brain11_router import Brain11Router, MODEL_FILES, classify_task
from brain11_tokenizer import encode


def flatten_messages(messages: list) -> str:
    parts = []
    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict) and p.get("type") == "text")
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n".join(parts).strip()


class Brain11ServerModel:
    def __init__(self, device=None, lazy=True):
        self.router = Brain11Router(device=device, lazy=lazy)
        self.model_id = "brain"
        self.display_name = "Brain (3 x 10.8M edge)"

    def list_models(self) -> dict:
        names = {
            "brain": self.display_name,
            "brain11-coder": "Brain Coder (10.8M)",
            "brain11-language": "Brain Language (10.8M, English+Urdu)",
            "brain11-agentic": "Brain Agentic (10.8M)",
        }
        return {"object": "list", "data": [
            {"id": mid, "object": "model", "created": int(time.time()),
             "owned_by": "brain", "name": name}
            for mid, name in names.items()
        ]}

    def chat_completion(self, body: dict, stream: bool = False) -> dict:
        messages = body.get("messages", [])
        model_id = body.get("model", "brain")
        max_tokens = int(body.get("max_tokens", 160))
        temperature = float(body.get("temperature", 0.6))
        prompt = flatten_messages(messages)
        if not prompt:
            prompt = "Hello"

        kind = None
        if "coder" in model_id or "code" in model_id:
            kind = "coder"
        elif "language" in model_id:
            kind = "language"
        elif "agentic" in model_id:
            kind = "agentic"
        else:
            kind = classify_task(prompt)

        if kind == "coder":
            res = self.router.code(prompt, n=max_tokens, temperature=min(temperature, 0.4))
            text = res["text"].strip()
            try:
                from brain11_loop import Healer
                text = Healer().normalize(text, prompt)
            except Exception:
                pass
            if text and not text.startswith("```"):
                text = "```python\n" + text + "\n```"
        elif kind == "agentic":
            res = self.router.agentic(prompt)
            text = json.dumps(res["decision"], ensure_ascii=False, indent=2)
        else:
            res = self.router.chat(prompt, n=max_tokens, temperature=temperature)
            text = res["text"].strip()

        prompt_tokens = len(encode(prompt, add_bos=False, add_eos=False))
        completion_tokens = len(encode(text, add_bos=False, add_eos=False))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class Brain430ServerModel:
    """Legacy 430M backend, kept behind --backend brain430."""

    def __init__(self, weights=None, device=None, use_rope=True):
        from brain_api import BrainAPI
        self.brain = BrainAPI(weights=weights, device=device, use_rope=use_rope)
        self.model_id = "brain"
        self.display_name = "Brain (430M MoE)"

    def list_models(self) -> dict:
        return {"object": "list", "data": [{
            "id": self.model_id, "object": "model", "created": int(time.time()),
            "owned_by": "brain", "name": self.display_name,
        }]}

    def chat_completion(self, body: dict, stream: bool = False) -> dict:
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 160))
        temperature = float(body.get("temperature", 0.8))
        prompt = flatten_messages(messages) + "\nAssistant:"
        text = self.brain.generate(prompt, n=max_tokens, temperature=temperature, top_k=40)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", self.model_id),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt.split()),
                      "completion_tokens": len(text.split()),
                      "total_tokens": len(prompt.split()) + len(text.split())},
        }


class BrainHandler(BaseHTTPRequestHandler):
    model = None

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks):
        body = "".join(chunks).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(self.model.list_models())
        elif self.path in ("/health", "/v1/health"):
            self._send_json({"status": "ok", "model": self.model.model_id,
                             "backend": type(self.model).__name__})
        elif self.path in ("/v1", "/v1/"):
            self._send_json({"status": "ok", "model": self.model.model_id,
                             "endpoints": ["/v1/models", "/v1/chat/completions", "/health", "/v1/health"]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": {"message": "invalid JSON"}}, status=400)
            return
        try:
            result = self.model.chat_completion(body, stream=bool(body.get("stream")))
        except Exception as exc:
            self._send_json({"error": {"message": str(exc), "type": "server_error"}}, status=500)
            return
        if body.get("stream"):
            model = body.get("model", self.model.model_id)
            cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            content = result["choices"][0]["message"]["content"]
            chunk1 = {"id": cid, "object": "chat.completion.chunk", "created": created,
                      "model": model,
                      "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                   "finish_reason": None}]}
            chunk2 = {"id": cid, "object": "chat.completion.chunk", "created": created,
                      "model": model,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self._send_sse([f"data: {json.dumps(chunk1, ensure_ascii=False)}\n\n",
                            f"data: {json.dumps(chunk2, ensure_ascii=False)}\n\n",
                            "data: [DONE]\n\n"])
        else:
            self._send_json(result)


class ThreadedHTTPServer(HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Brain OpenAI-compatible server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", choices=["brain11", "brain430"], default="brain11")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-rope", action="store_true")
    args = ap.parse_args()

    if args.backend == "brain430":
        model = Brain430ServerModel(weights=args.weights, device=args.device,
                                    use_rope=not args.no_rope)
    else:
        model = Brain11ServerModel(device=args.device)
    BrainHandler.model = model

    server = ThreadedHTTPServer((args.host, args.port), BrainHandler)
    print(f"[brain_http_server] http://{args.host}:{args.port}", flush=True)
    print(f"[brain_http_server] backend={args.backend} model={model.model_id} "
          f"({model.display_name})", flush=True)
    print("[brain_http_server] endpoints: /v1/models, /v1/chat/completions, /health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[brain_http_server] shutting down", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
