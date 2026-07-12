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


def test_full_local_never_emits_unsure(monkeypatch):
    from frugal_router import local_tier, simple
    answers = iter(["UNSURE", "The capital of France is Paris.", ""])
    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate",
                        lambda s, p, c, temperature=0.0: next(answers, ""))
    out = simple._solve_local_only("t", "factual", "What is the capital of France?",
                                   time.monotonic() + 9999)
    assert out and "UNSURE" not in out.upper()


def test_full_local_forced_retry_appends_commitment(monkeypatch):
    from frugal_router import local_tier, simple
    seen = []

    def fake_gen(s, p, c, temperature=0.0):
        seen.append(s)
        return "UNSURE" if len(seen) == 1 else "Answer: 42"
    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate", fake_gen)
    out = simple._solve_local_only("t", "math", "What is 6 x 7?",
                                   time.monotonic() + 9999)
    assert "42" in out
    assert any("Never reply UNSURE" in s for s in seen[1:])
