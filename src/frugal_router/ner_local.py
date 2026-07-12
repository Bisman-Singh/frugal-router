"""Zero-token NER tier: spaCy in an isolated venv, verified before trusted.

The extractor runs via subprocess (SPACY_PY interpreter) so the application
environment keeps its exact dependency pins. An extraction is kept ONLY when
every span appears VERBATIM in the source text and at least two typed
entities were found — anything else defers to the next tier. A drifted or
wrong spaCy can therefore never emit an unverifiable answer.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

_QUOTED = re.compile(r'"([^"]{20,})"')

_SCRIPT = os.environ.get(
    "SPACY_SCRIPT", os.path.join(os.path.dirname(__file__), "..", "..",
                                 "scripts", "spacy_ner.py"))


def _source_text(prompt: str) -> str:
    """The text to analyse: the quoted passage when present, else the prompt."""
    m = _QUOTED.search(prompt)
    return m.group(1) if m else prompt


def extract(prompt: str, timeout: float = 20.0) -> str | None:
    """Typed 'label: value' lines, or None to defer. Never raises."""
    interp = os.environ.get("SPACY_PY")
    if not interp or not os.path.exists(interp):
        return None
    script = os.environ.get("SPACY_SCRIPT", "/app/scripts/spacy_ner.py")
    if not os.path.exists(script):
        return None
    text = _source_text(prompt)
    try:
        out = subprocess.run([interp, script], input=text, text=True,
                             capture_output=True, timeout=timeout)
        ents = json.loads(out.stdout or "[]")
    except Exception:
        return None

    lines, seen = [], set()
    for e in ents:
        label, span = e.get("label"), (e.get("text") or "").strip()
        if not label or not span:
            continue
        if span not in text:          # verbatim-span gate: no hallucinations
            continue
        key = (label, span.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{label}: {span}")
    if len(lines) < 2:                # too thin to trust; defer
        return None
    return "\n".join(lines)
