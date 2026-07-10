"""Scored-path tests against a scriptable mock endpoint.

Each test drives run_simple end-to-end over HTTP and scripts the mock's
behavior per request, so the failure paths are actually exercised:
  - primary-model outage -> a DIFFERENT model answers (real failover)
  - wrong-but-well-shaped math -> reasoning confirmation -> tiebreak majority
  - finish_reason == "length" -> one doubled-budget retry
  - all models failing -> bounded attempts, blank preserved, exit 0
  - non-chat entries in ALLOWED_MODELS are never called
  - a failed first answer is not replaced by a worse later one
  - duplicate/malformed input records, ledger integrity, zero-token solvers

NOTE: this is a worktree-level test of the scored entrypoint function. The
Docker image itself is smoke-tested by scripts/smoke_image.sh (build + run
against the same mock), which CI/humans run before pushing a tag.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from frugal_router.simple import LEDGER, run_simple

ALLOWED = "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it"


class _Mock(BaseHTTPRequestHandler):
    """Scriptable mock: tests assign a handler(body) -> (status, payload)."""

    script = staticmethod(lambda body: (200, "ok", "stop"))
    calls: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.calls.append(body)
        status, text, finish = _Mock.script(body)
        if status != 200:
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"scripted failure"}}')
            return
        payload = {
            "id": "mock", "object": "chat.completion", "model": body.get("model", ""),
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    _Mock.calls = []
    LEDGER.clear()
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED)
    monkeypatch.setenv("SOLVERS", "0")
    monkeypatch.setenv("TIMEOUT_S", "10")

    def run(tasks):
        inp, outp = tmp_path / "tasks.json", tmp_path / "results.json"
        inp.write_text(json.dumps(tasks))
        assert run_simple(str(inp), str(outp)) == 0
        return {r["task_id"]: r["answer"] for r in json.loads(outp.read_text())}, outp

    yield run
    srv.shutdown()


def test_primary_outage_fails_over_to_other_model(env):
    """minimax 404s; the answer must come from a different model family."""
    def script(body):
        if "minimax" in body["model"]:
            return 404, None, None
        return 200, "Photosynthesis converts light into chemical energy in plants.", "stop"
    _Mock.script = staticmethod(script)

    results, _ = env([{"task_id": "f1", "prompt": "Explain how photosynthesis works."}])
    assert results["f1"].startswith("Photosynthesis")
    models = [c["model"] for c in _Mock.calls]
    assert any("minimax" in m for m in models), "primary was tried first"
    assert any("minimax" not in m for m in models), "a different model answered"


def test_wrong_math_disagreement_goes_to_tiebreak(env):
    """Well-shaped but wrong primary math answer: the reasoning confirmation
    disagrees, the cross-model tiebreak sides with it, and the majority's
    answer is emitted."""
    def script(body):
        model = body["model"]
        suppressed = "reasoning_effort" in json.dumps(body)
        if "minimax" in model and suppressed:
            return 200, "7 x 3 = 21. 50 - 21 = 31.\nAnswer: 31", "stop"   # wrong
        if "minimax" in model:  # reasoning confirmation
            return 200, "Careful: 7x3=21, 50-21=29.\nAnswer: 29", "stop"  # right
        return 200, "Total 21, change 29.\nAnswer: 29", "stop"            # tiebreak
    _Mock.script = staticmethod(script)

    results, _ = env([{"task_id": "m1",
                       "prompt": "Pens cost $3 each. Buying 7 with a $50 note, how much change?"}])
    assert "29" in results["m1"] and "31" not in results["m1"].split("Answer:")[-1]


def test_agreeing_confirmation_keeps_primary(env):
    def script(body):
        return 200, "21 spent, 29 back.\nAnswer: 29", "stop"
    _Mock.script = staticmethod(script)
    results, _ = env([{"task_id": "m2",
                       "prompt": "Pens cost $3 each. Buying 7 with a $50 note, how much change?"}])
    assert "29" in results["m2"]
    # primary + one confirmation call only
    assert len(_Mock.calls) == 2


def test_length_finish_reason_retries_with_bigger_budget(env):
    state = {"n": 0}

    def script(body):
        state["n"] += 1
        if state["n"] == 1:
            return 200, "The council met and", "length"
        return 200, "The council approved the modified transit plan in one sentence.", "stop"
    _Mock.script = staticmethod(script)

    results, _ = env([{"task_id": "z1", "prompt": "Summarize in one sentence: the council met..."}])
    assert results["z1"].endswith("sentence.")
    assert _Mock.calls[1]["max_tokens"] == _Mock.calls[0]["max_tokens"] * 2


def test_total_outage_is_bounded_and_blank_preserved(env):
    _Mock.script = staticmethod(lambda body: (500, None, None))
    results, outp = env([{"task_id": "x1", "prompt": "Explain gravity."}])
    assert results["x1"] == ""            # honest blank, exit 0, schema valid
    # chain = [minimax, kimi, gemma-31b] + corrective slot; SDK retries once
    # per request, so the logical-attempt bound is 4 (<=8 wire calls).
    assert len(_Mock.calls) <= 8
    log = json.loads(outp.with_name("inference_log.json").read_text())
    assert all(e["status"].startswith("error") for e in log["calls"])


def test_non_chat_models_never_called(env):
    _Mock.script = staticmethod(
        lambda body: (200, "A clear factual explanation of gravity and mass.", "stop"))
    # flux/whisper first in the list must be filtered, not used as fallback
    import os
    os.environ["ALLOWED_MODELS"] = "flux-1-schnell,whisper-v3,minimax-m3,kimi-k2p7-code"
    try:
        results, _ = env([{"task_id": "g1", "prompt": "Explain gravity."}])
    finally:
        os.environ["ALLOWED_MODELS"] = ALLOWED
    assert results["g1"]
    assert all("flux" not in c["model"] and "whisper" not in c["model"]
               for c in _Mock.calls)


def test_first_answer_not_replaced_by_worse_retry(env):
    """Both attempts fail validation; the FIRST non-empty answer must win."""
    state = {"n": 0}

    def script(body):
        state["n"] += 1
        if state["n"] == 1:
            return 200, "It is unclear whether yes or no applies here at all.", "stop"
        return 200, "no idea", "stop"   # later and worse
    _Mock.script = staticmethod(script)

    results, _ = env([{"task_id": "l1",
                       "prompt": "All wumps are glorks. Answer yes or no: are all wumps glorks?"}])
    assert results["l1"].startswith("It is unclear")


def test_duplicates_malformed_and_ledger(env):
    _Mock.script = staticmethod(
        lambda body: (200, "Positive. The reviewer clearly loved the product.", "stop"))
    results, outp = env([
        {"task_id": "t1", "prompt": "Classify the sentiment of: 'Great product!'"},
        {"task_id": "t1", "prompt": "duplicate must be skipped"},
        "malformed-record",
    ])
    assert list(results) == ["t1"]
    log = json.loads(outp.with_name("inference_log.json").read_text())
    assert log["summary"]["model_calls"] == len(log["calls"]) == len(_Mock.calls)
    assert all({"task_id", "category", "model", "status", "attempt",
                "finish_reason", "duration_ms"} <= set(e) for e in log["calls"])


def test_solvers_zero_token_path(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("SOLVERS", "1")
    tasks = [
        {"task_id": "s1", "prompt": "What is 15% of 240?"},
        {"task_id": "s2", "prompt": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"},
    ]
    inp, outp = tmp_path / "tasks.json", tmp_path / "results.json"
    inp.write_text(json.dumps(tasks))
    assert run_simple(str(inp), str(outp)) == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(outp.read_text())}
    assert results["s1"] == "36"
    assert results["s2"] == "Carol"
