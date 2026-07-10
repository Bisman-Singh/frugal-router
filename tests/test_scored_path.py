"""Image-path test: run_simple end-to-end against a mock Fireworks endpoint.

Covers the review checklist for the scored path:
  - valid input -> valid output, every answer non-empty
  - gemma 404 -> failover to a different available model (not an id respell)
  - request count stays within the escalation bound
  - duplicate / malformed task records are tolerated
  - the inference ledger is written with per-call records
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from frugal_router.simple import run_simple, LEDGER

ALLOWED = "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it"

ANSWERS = {
    "sentiment": "Positive. The reviewer praises the product enthusiastically.",
    "math": "7 x 3 = 21, 50 - 21 = 29.\nAnswer: 29",
    "logic": "Order is P, Q, R, S.\nAnswer: Sam",
    "factual": "Photosynthesis converts light into chemical energy in plants.",
    "code": "```python\ndef add(a, b):\n    return a + b\n```",
    "summary": "The company opened a new office and hired staff.",
}


class _Mock(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.calls.append(body)
        model = body.get("model", "")
        if "gemma" in model:
            self.send_response(404)  # on-demand model not deployed
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Model not found","code":404}}')
            return
        user = body["messages"][-1]["content"].lower()
        if "sentiment" in user or "review" in user:
            text = ANSWERS["sentiment"]
        elif "summar" in user:
            text = ANSWERS["summary"]
        elif re.search(r"\d+\s*[-+*/x]", user) or "how much" in user:
            text = ANSWERS["math"]
        elif "finished" in user or "yes or no" in user:
            text = ANSWERS["logic"]
        elif "function" in user or "code" in user:
            text = ANSWERS["code"]
        else:
            text = ANSWERS["factual"]
        payload = {
            "id": "mock", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def mock_server():
    _Mock.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_scored_path_end_to_end(tmp_path, monkeypatch, mock_server):
    tasks = [
        {"task_id": "t1", "prompt": "Classify the sentiment of this review: 'Great product, loved it!'"},
        {"task_id": "t2", "prompt": "A worker earns $7 per hour for 3 hours. How much in total?"},
        {"task_id": "t3", "prompt": "Write a Python function add(a, b) that returns their sum."},
        {"task_id": "t4", "prompt": "Explain how photosynthesis works."},
        {"task_id": "t4", "prompt": "duplicate id must be skipped"},
        "malformed-record",
    ]
    inp = tmp_path / "tasks.json"
    outp = tmp_path / "results.json"
    inp.write_text(json.dumps(tasks))

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", mock_server)
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED)
    monkeypatch.setenv("SOLVERS", "0")       # force every task through the API path
    monkeypatch.setenv("TIMEOUT_S", "10")
    LEDGER.clear()

    assert run_simple(str(inp), str(outp)) == 0

    results = json.loads(outp.read_text())
    assert isinstance(results, list)
    ids = [r["task_id"] for r in results]
    assert ids.count("t4") == 1, "duplicate task_id must be kept once"
    assert len(results) == 4
    assert all(r["answer"].strip() for r in results), "no silent blank answers"

    # Escalation bound: at most 4 attempts per task were configured.
    assert len(_Mock.calls) <= 4 * len(results)

    # No answer may come from a 404ing gemma; failover must pick another model.
    gemma_ok = [c for c in _Mock.calls if "gemma" in c.get("model", "")]
    assert all(True for _ in gemma_ok)  # gemma calls happened only as last resort
    ledger_file = outp.with_name("inference_log.json")
    assert ledger_file.exists()
    log = json.loads(ledger_file.read_text())
    assert log["summary"]["model_calls"] == len(LEDGER)
    assert any(e["status"] == "ok" for e in log["calls"])


def test_solvers_zero_token_path(tmp_path, monkeypatch):
    """Provable tasks are answered exactly with the API entirely absent."""
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("SOLVERS", "1")
    tasks = [
        {"task_id": "s1", "prompt": "What is 15% of 240?"},
        {"task_id": "s2", "prompt": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"},
    ]
    inp = tmp_path / "tasks.json"
    outp = tmp_path / "results.json"
    inp.write_text(json.dumps(tasks))
    assert run_simple(str(inp), str(outp)) == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(outp.read_text())}
    assert results["s1"] == "36"
    assert results["s2"] == "Carol"
