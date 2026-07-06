"""Heuristic category classification.

The judging harness supplies only {task_id, prompt}, so the category must be
read from the prompt text. The published category definitions are distinctive
enough that keyword heuristics cover them; 'factual' is the fallback.
"""
from __future__ import annotations

import re

from .tasks import Task

_SENTIMENT = re.compile(r"(?i)\bsentiment\b|\b(positive or negative)\b")
_SUMMARY = re.compile(r"(?i)\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b")
_NER = re.compile(
    r"(?i)\bnamed entit|\bentities\b|\bextract\b.{0,40}\b(people|persons?|organi[sz]ations?|locations?|dates?)\b"
)
_CODE = re.compile(
    r"```|\bdef\s+\w|\bclass\s+\w|\bfunction\b|\bimport\s+\w|\bprint\(|=>|\breturn\b|\bconsole\.log"
)
_DEBUG = re.compile(r"(?i)\b(bug|debug|fix|error|crash|incorrect|wrong output|doesn'?t work|fails?)\b")
_CODE_GEN = re.compile(r"(?i)\b(write|implement|create|generate)\b.{0,60}\b(function|program|script|class|code|method)\b")
_LOGIC = re.compile(
    r"(?i)\bpuzzle\b|\bdeduce\b|\bmust be true\b|\bif all\b|\bexactly one\b|"
    r"\btaller than\b|\bolder than\b|\bfinished (before|after)\b|\bwho is (the |telling )\b|"
    r"\bknights?\b.{0,30}\bliars?\b|\beither\b.{0,50}\bor\b.{0,80}\bwho\b"
)
_MATH = re.compile(
    r"(?i)\b(how many|how much|calculate|compute|sum of|total|percent|average|"
    r"remainder|product of|difference between|per (hour|day|week|month))\b"
    r"|[0-9]+\s*[+\-*/^]\s*[0-9]+"
)


def classify(task: Task) -> str:
    if task.type:
        return task.type
    text = task.input
    if _SENTIMENT.search(text):
        return "sentiment"
    if _SUMMARY.search(text):
        return "summarization"
    if _NER.search(text):
        return "ner"
    if _CODE.search(text) and _DEBUG.search(text):
        return "code_debug"
    if _CODE.search(text) or _CODE_GEN.search(text):
        return "code_gen"
    if _LOGIC.search(text):
        return "logic"
    if _MATH.search(text) and re.search(r"\d", text):
        return "math"
    return "factual"
