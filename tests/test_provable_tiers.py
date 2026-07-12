"""Provable zero-token tiers: spaCy-verbatim NER and executed-agreement math."""
import json
import os
import stat
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frugal_router import ner_local, simple  # noqa: E402

PROMPT = ('Extract the named entities (person, organization, location, date) '
          'from this text, one per line as \'label: value\': "On June 3, 2024, '
          'Alice Meier of Helios Motors spoke in Jakarta."')


def _stub_spacy(tmp_path, ents):
    """A fake SPACY_PY interpreter: ignores input, prints canned JSON."""
    script = tmp_path / "stub.py"
    script.write_text("import json,sys\nsys.stdin.read()\n"
                      f"print(json.dumps({json.dumps(ents)}))\n")
    interp = tmp_path / "fakepython"
    interp.write_text(f"#!/bin/sh\nexec python3 {script} \"$@\"\n")
    interp.chmod(interp.stat().st_mode | stat.S_IEXEC)
    return str(interp)


def test_ner_verbatim_spans_accepted(tmp_path, monkeypatch):
    ents = [{"label": "person", "text": "Alice Meier"},
            {"label": "organization", "text": "Helios Motors"},
            {"label": "location", "text": "Jakarta"},
            {"label": "date", "text": "June 3, 2024"}]
    monkeypatch.setenv("SPACY_PY", _stub_spacy(tmp_path, ents))
    monkeypatch.setenv("SPACY_SCRIPT", str(tmp_path / "stub.py"))
    out = ner_local.extract(PROMPT)
    assert out is not None
    assert "person: Alice Meier" in out and "date: June 3, 2024" in out


def test_ner_hallucinated_span_rejected(tmp_path, monkeypatch):
    ents = [{"label": "person", "text": "Bob Petrov"},          # NOT in text
            {"label": "organization", "text": "Acme Corp"}]     # NOT in text
    monkeypatch.setenv("SPACY_PY", _stub_spacy(tmp_path, ents))
    monkeypatch.setenv("SPACY_SCRIPT", str(tmp_path / "stub.py"))
    assert ner_local.extract(PROMPT) is None                    # defers


def test_ner_thin_extraction_defers(tmp_path, monkeypatch):
    ents = [{"label": "person", "text": "Alice Meier"}]         # only one
    monkeypatch.setenv("SPACY_PY", _stub_spacy(tmp_path, ents))
    monkeypatch.setenv("SPACY_SCRIPT", str(tmp_path / "stub.py"))
    assert ner_local.extract(PROMPT) is None


def test_ner_missing_interp_defers(monkeypatch):
    monkeypatch.setenv("SPACY_PY", "/nonexistent/python")
    assert ner_local.extract(PROMPT) is None


def test_run_pot_executes_and_parses():
    assert simple._run_pot("print(40 * 8.4 / 100 * 10)") == "33.6"
    assert simple._run_pot("while True: pass", timeout=1.5) is None
    assert simple._run_pot("raise SystemExit(1)") is None


def test_math_pot_agreement_gate(monkeypatch):
    from frugal_router import local_tier

    calls = {"n": 0}

    def fake_generate(system, prompt, cap, temperature=0.0):
        calls["n"] += 1
        if "program" in system.lower():
            return "```python\nprint(0.15 * 240)\n```"
        return "Computing.\nAnswer: 36"

    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate", fake_generate)
    simple._LOCAL_SPENT["s"] = 0.0
    out = simple._try_math_pot("t1", "What is 15% of 240?", time.monotonic() + 600)
    assert out is not None and "Answer: 36" in out


def test_math_pot_disagreement_defers(monkeypatch):
    from frugal_router import local_tier

    def fake_generate(system, prompt, cap, temperature=0.0):
        if "program" in system.lower():
            return "```python\nprint(0.15 * 240)\n```"      # 36
        return "Computing.\nAnswer: 42"                      # disagrees

    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate", fake_generate)
    simple._LOCAL_SPENT["s"] = 0.0
    assert simple._try_math_pot("t1", "What is 15% of 240?",
                                time.monotonic() + 600) is None
