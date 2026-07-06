"""Answer extraction, normalization, and format validation.

A correct answer the harness cannot parse scores as wrong, so this module is
deliberately strict about what it accepts and forgiving about what it reads.
"""
from __future__ import annotations

import re

_ANSWER_LINE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MCQ_LETTER = re.compile(r"\b([A-E])\b")


def extract_answer(text: str | None) -> str | None:
    """Pull the answer out of a model response. Prefers the last 'Answer:' line."""
    if not text:
        return None
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def normalize_number(raw: str) -> str | None:
    m = _NUMBER.search(raw.replace("$", "").replace("%", ""))
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return str(int(value)) if value.is_integer() else str(value)


def normalize(raw: str | None, task_type: str) -> str | None:
    if raw is None:
        return None
    raw = raw.strip().strip('"').strip("'").strip()
    if not raw:
        return None
    if task_type == "math":
        return normalize_number(raw)
    if task_type == "mcq":
        m = _MCQ_LETTER.search(raw.upper())
        return m.group(1) if m else None
    return re.sub(r"\s+", " ", raw).rstrip(".").casefold() or None


def is_valid(normalized: str | None, task_type: str) -> bool:
    if not normalized:
        return False
    if task_type == "math":
        try:
            float(normalized)
        except ValueError:
            return False
        return True
    if task_type == "mcq":
        return bool(re.fullmatch(r"[A-E]", normalized))
    return True
