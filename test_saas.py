#!/usr/bin/env python3
"""Tests for the SaaS application (Brain AI MoE model) — saas_backend.py.

Runs the backend in a subprocess and exercises:
    GET  /health
    GET  /models
    POST /generate
    GET  / (dashboard)

Usage:
    python3 test_saas.py
"""

import json
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


class SaaSBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = subprocess.Popen(
            [sys.executable, str(HERE / "saas_backend.py")],
            cwd=str(HERE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(120):
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("SaaS backend did not start; output so far:\n"
                           + cls.proc.stdout.read().decode(errors="replace"))

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def _get(self, path):
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            return r.status, r.read()

    def _get_json(self, path):
        status, body = self._get(path)
        return status, json.loads(body)

    def _post(self, path, payload):
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())

    def test_health(self):
        status, body = self._get_json("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "ok")

    def test_models(self):
        status, body = self._get_json("/models")
        self.assertEqual(status, 200)
        self.assertGreater(body["params"], 0)
        self.assertGreaterEqual(body["experts"], 2)
        self.assertGreaterEqual(body["layers"], 1)

    def test_generate(self):
        status, body = self._post("/generate", {"prompt": "def fizzbuzz(n):", "n": 6})
        self.assertEqual(status, 200)
        self.assertIn("text", body)
        self.assertIn("finish_reason", body)
        self.assertEqual(body["model"], "brain")

    def test_dashboard(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        html = body.decode(errors="replace")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Brain AI", html)

    def test_not_found(self):
        with self.assertRaises(urllib.error.HTTPError):
            self._get_json("/nope")


if __name__ == "__main__":
    unittest.main()
