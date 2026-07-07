"""Answer extraction, vote keys, and validity checks.

Two distinct jobs, never confused: vote_key() produces an aggressively
normalized key so self-consistency samples can be compared, while
final_answer() produces the intent-complete text actually emitted for the
LLM judge. Normalizing the emitted answer would strip the justification and
formatting the judge is looking for.
"""
from __future__ import annotations

import re

from .contracts import STYLE_LINE, style_of

_ANSWER_LINE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SENTIMENT_LABEL = re.compile(r"(?i)\b(positive|negative|neutral|mixed)\b")
_CODE_SHAPE = re.compile(r"```|\bdef\s+\w|\bclass\s+\w|\bfunction\b|\breturn\b")


def extract_answer(text: str | None) -> str | None:
    """Pull the answer line out of a response. Prefers the last 'Answer:' line."""
    if not text:
        return None
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return strip_md(matches[-1]).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return strip_md(lines[-1]).strip() if lines else None


def normalize_number(raw: str | None) -> str | None:
    if not raw:
        return None
    matches = _NUMBER.findall(raw.replace("$", "").replace("%", ""))
    if not matches:
        return None
    try:
        # The last number is the answer far more often than the first
        # ("Step 3: 15 + 6 = 21" must read as 21, not 3).
        value = float(matches[-1].replace(",", ""))
    except ValueError:
        return None
    return str(int(value)) if value.is_integer() else str(value)


_MD = re.compile(r"[*_`#]+")


def strip_md(raw: str) -> str:
    return _MD.sub("", raw)


def text_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    # Normalize unicode spaces (models emit narrow/no-break spaces) to ASCII.
    raw = raw.replace(" ", " ").replace(" ", " ")
    key = strip_md(re.sub(r"\s+", " ", raw)).strip().strip('"').strip("'").strip().rstrip(".").casefold()
    return key or None


def vote_key(text: str | None, category: str) -> str | None:
    """Normalized comparison key for self-consistency voting."""
    if not text:
        return None
    if category == "math":
        return normalize_number(extract_answer(text))
    if category == "sentiment":
        m = _SENTIMENT_LABEL.search(extract_answer(text) or text)
        return m.group(1).casefold() if m else None
    if style_of(category) == STYLE_LINE:
        return text_key(extract_answer(text))
    return text_key(text)


def final_answer(text: str | None, category: str) -> str | None:
    """The answer actually emitted, shaped for an intent judge."""
    if not text or not text.strip():
        return None
    if category == "math":
        raw = extract_answer(text)
        return normalize_number(raw) or (raw.strip() if raw else None)
    if style_of(category) == STYLE_LINE:
        return (extract_answer(text) or text).strip()
    return text.strip()


def is_valid_answer(answer: str | None, category: str) -> bool:
    """Cheap structural check that the answer can plausibly satisfy the judge."""
    if not answer or not answer.strip():
        return False
    if category == "math":
        return normalize_number(answer) is not None
    if category == "sentiment":
        # A label alone is not intent-complete; the justification must be there.
        return bool(_SENTIMENT_LABEL.search(answer)) and len(answer.split()) >= 4
    if category in ("code_debug", "code_gen"):
        return bool(_CODE_SHAPE.search(answer))
    return True
