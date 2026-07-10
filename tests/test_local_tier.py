"""Local-tier gating: every doubt escalates; only fully-gated answers stay."""
import time

from frugal_router import local_tier, simple


def _wall():
    return time.monotonic() + 9999


def test_kept_when_all_gates_pass(monkeypatch):
    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate",
                        lambda s, p, c, temperature=0.0:
                        "Positive. The reviewer praises the build quality.")
    monkeypatch.setattr(local_tier, "verify", lambda p, a: True)
    out = simple._try_local("t", "sentiment", "Classify the sentiment: 'Great!'", _wall())
    assert out and out.startswith("Positive")


def test_escalates_when_self_verify_says_no(monkeypatch):
    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate",
                        lambda s, p, c, temperature=0.0:
                        "Positive. The reviewer praises the build quality.")
    monkeypatch.setattr(local_tier, "verify", lambda p, a: False)
    assert simple._try_local("t", "sentiment", "Classify the sentiment: 'Great!'", _wall()) is None


def test_escalates_on_sentiment_disagreement(monkeypatch):
    answers = iter(["Positive. Praise throughout.", "Negative. Hidden complaints."])
    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate",
                        lambda s, p, c, temperature=0.0: next(answers))
    monkeypatch.setattr(local_tier, "verify", lambda p, a: True)
    assert simple._try_local("t", "sentiment", "Classify the sentiment: 'Great!'", _wall()) is None


def test_escalates_on_invalid_format(monkeypatch):
    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate",
                        lambda s, p, c, temperature=0.0: "person and organization are labels")
    monkeypatch.setattr(local_tier, "verify", lambda p, a: True)
    assert simple._try_local("t", "ner", "Extract entities from: 'Bob at Acme.'", _wall()) is None


def test_ineligible_category_and_deadline(monkeypatch):
    monkeypatch.setattr(local_tier, "available", lambda: True)
    assert simple._try_local("t", "math", "2+2?", _wall()) is None
    assert simple._try_local("t", "factual", "Explain x.", time.monotonic() + 100) is None
