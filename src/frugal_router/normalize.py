"""Category-aware answer normalization.

The judge scores answers against the category's expected intent, so an answer
that is *correct but mis-shaped* (label buried mid-sentence, missing final
Answer line, code wrapped in prose) reads as a miss. This layer repairs shape
only — it never invents content, never truncates reasoning the judge may want,
and always returns non-empty text (falling back to the raw answer when a rule
does not confidently apply).
"""
from __future__ import annotations

import json
import re

_PREAMBLE = re.compile(
    r"^(sure[,!.]?|okay[,!.]?|certainly[,!.]?|of course[,!.]?|"
    r"here(?:'s| is)(?: the)?(?: corrected| final)?(?: answer| code| summary)?[:,]?|"
    r"the (?:correct )?answer is[:,]?|final answer[:,]?)\s*",
    re.IGNORECASE,
)
_SENTIMENT_LABEL = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*[:=]")
_LAST_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_FENCED = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
_SUMMARY_PREFIX = re.compile(r"^(summary|tl;?dr)\s*[:\-]\s*", re.IGNORECASE)


def _strip_preamble(text: str) -> str:
    prev = None
    out = text.strip()
    while prev != out:
        prev = out
        out = _PREAMBLE.sub("", out).strip()
    return out


def _lead_with_label(text: str) -> str:
    """Sentiment: make sure the label opens the answer (judge reads label first).
    The justification is part of the category intent, so the rest is kept."""
    labels = {m.group(1).lower() for m in _SENTIMENT_LABEL.finditer(text)}
    if len(labels) != 1:
        return text  # zero or conflicting labels: not confident, leave as-is
    label = next(iter(labels))
    if text.strip().lower().startswith(label):
        return text
    return f"{label.capitalize()}. {text.strip()}"


def _ensure_answer_line(text: str, candidate: str | None) -> str:
    """Math/logic: guarantee a final 'Answer: <value>' line when we can extract
    the value unambiguously; the working stays untouched above it."""
    if _ANSWER_LINE.search(text) or not candidate:
        return text
    return f"{text.rstrip()}\nAnswer: {candidate}"


def _math_candidate(text: str) -> str | None:
    nums = _LAST_NUMBER.findall(text)
    return nums[-1].rstrip(",") if nums else None


def _logic_candidate(text: str) -> str | None:
    # Only yes/no is safe to extract blindly; names/values stay with the model.
    tail = text[-200:]
    hits = {m.group(1).lower() for m in _YES_NO.finditer(tail)}
    if len(hits) == 1:
        return next(iter(hits)).capitalize()
    return None


def _only_fenced_code(text: str) -> str:
    """code_gen asks for code only: when a fenced block exists, hand over just
    the first block (fences kept) instead of prose + code."""
    m = _FENCED.search(text)
    if not m:
        return text
    lang = re.match(r"```([a-zA-Z0-9]*)", text[m.start():])
    tag = lang.group(1) if lang else ""
    return f"```{tag}\n{m.group(1).rstrip()}\n```"


def _ner_lines(text: str) -> str:
    """NER wants 'label: value' lines. Convert a JSON payload if the model sent
    one; anything else passes through untouched."""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return text
    try:
        data = json.loads(stripped)
    except Exception:
        return text
    lines: list[str] = []
    if isinstance(data, dict):
        for label, values in data.items():
            for v in values if isinstance(values, list) else [values]:
                lines.append(f"{label}: {v}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                label = item.get("label") or item.get("type") or item.get("entity")
                value = item.get("value") or item.get("text") or item.get("name")
                if label and value:
                    lines.append(f"{label}: {value}")
    return "\n".join(lines) if lines else text


def normalize(category: str, text: str) -> str:
    """Repair the shape of a model answer for its category. Non-destructive:
    on any doubt the original text wins, and the result is never empty."""
    raw = (text or "").strip()
    if not raw:
        return raw
    out = _strip_preamble(raw)

    if category == "sentiment":
        out = _lead_with_label(out)
    elif category == "math":
        out = _ensure_answer_line(out, _math_candidate(out))
    elif category == "logic":
        out = _ensure_answer_line(out, _logic_candidate(out))
    elif category == "code_gen":
        out = _only_fenced_code(out)
    elif category == "ner":
        out = _ner_lines(out)
    elif category == "summarization":
        out = _SUMMARY_PREFIX.sub("", out).strip()
    # factual + code_debug: preamble strip only; their full body is the answer.

    return out if out.strip() else raw
