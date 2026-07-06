"""Heuristic task-type classification. Free, local, and deliberately simple."""
from __future__ import annotations

import re

from .tasks import Task

_MATH_HINTS = re.compile(
    r"(?i)\b(how many|how much|calculate|compute|sum of|total|percent|average|"
    r"remainder|product of|difference between|per (hour|day|week))\b"
    r"|[0-9]+\s*[+\-*/^]\s*[0-9]+"
)
_MCQ_HINTS = re.compile(r"(?im)^\s*\(?[A-E][.):]\s+\S|which of the following")
_CLASSIFY_HINTS = re.compile(r"(?i)\b(classify|sentiment|label|categor)")
_SUMMARY_HINTS = re.compile(r"(?i)\b(summari[sz]e|tl;?dr|in one sentence)\b")


def classify(task: Task) -> str:
    if task.type:
        return task.type
    if task.choices:
        return "mcq"
    text = task.input
    if _MCQ_HINTS.search(text):
        return "mcq"
    if _SUMMARY_HINTS.search(text):
        return "summarization"
    if _CLASSIFY_HINTS.search(text):
        return "classification"
    if task.context:
        return "extraction"
    if _MATH_HINTS.search(text) and re.search(r"\d", text):
        return "math"
    return "general"
