"""BATCH=1: grouped same-category calls with per-item validation + solo fallback."""
import json

from test_scored_path import _Mock, env  # noqa: F401  (fixture reuse)


def _batch_reply(body):
    """Answer every '### ITEM n' in the batched user message; sentiment shape."""
    user = body["messages"][-1]["content"]
    n = user.count("### ITEM")
    lines = []
    for i in range(1, n + 1):
        lines.append(f"### ITEM {i}\nPositive. The reviewer clearly liked it.")
    return (200, "\n".join(lines), "stop")


def test_batch_groups_into_one_call(env, monkeypatch):
    monkeypatch.setenv("BATCH", "1")
    _Mock.script = staticmethod(_batch_reply)
    tasks = [{"task_id": f"s{i}", "prompt":
              f"Classify the sentiment of this review: 'Great product number {i}, love it!'"}
             for i in range(5)]
    results, _ = env(tasks)
    assert all("Positive" in results[f"s{i}"] for i in range(5))
    # ONE grouped call answered all five (no solo calls needed)
    assert len(_Mock.calls) == 1, [c["model"] for c in _Mock.calls]


def test_batch_missing_item_falls_solo(env, monkeypatch):
    monkeypatch.setenv("BATCH", "1")
    state = {"n": 0}

    def script(body):
        state["n"] += 1
        user = body["messages"][-1]["content"]
        if "### ITEM" in user and state["n"] == 1:
            # answer only items 1-2; item 3 missing -> must fall to solo path
            return (200, "### ITEM 1\nPositive. The reviewer praises it strongly.\n"
                         "### ITEM 2\nNegative. The reviewer clearly hated it.", "stop")
        return (200, "Neutral. Mixed signals in the text.", "stop")

    _Mock.script = staticmethod(script)
    tasks = [{"task_id": "a", "prompt": "Classify the sentiment of this review: 'excellent!'"},
             {"task_id": "b", "prompt": "Classify the sentiment of this review: 'terrible!'"},
             {"task_id": "c", "prompt": "Classify the sentiment of this review: 'it is fine.'"}]
    results, _ = env(tasks)
    assert "Positive" in results["a"] and "Negative" in results["b"]
    assert "Neutral" in results["c"]          # recovered via solo fallback
    assert state["n"] >= 2                    # batch call + at least one solo


def test_batch_off_by_default(env):
    _Mock.script = staticmethod(
        lambda body: (200, "Positive. The reviewer praises the product.", "stop"))
    tasks = [{"task_id": f"s{i}", "prompt":
              f"Classify the sentiment of this review: 'nice thing {i}!'"} for i in range(3)]
    results, _ = env(tasks)
    assert len(results) == 3
    assert len(_Mock.calls) >= 3              # no grouping without BATCH=1
