"""Heuristic category classification.

The judging harness supplies only {task_id, prompt}, so the category must be
read from the prompt text. Category is what decides the answer contract we send
(format, length, model), and the judge grades format/constraint compliance, so a
mis-classified task comes back in the wrong shape and scores zero even when the
model knew the answer. This classifier is therefore deliberately thorough: many
surface phrasings per category, checked in priority order so the higher-signal,
more specific categories win over the general 'factual' fallback.
"""
from __future__ import annotations

import re

from .tasks import Task

# --- Raw-code signals (used to disambiguate code tasks) ---------------------
_CODE_FENCE = re.compile(r"```")
_CODE_SYNTAX = re.compile(
    r"\bdef\s+\w|\bclass\s+\w+\s*[:( ]|\bfunction\b|\bimport\s+\w|#include|"
    r"\bprint\(|\bconsole\.log|System\.out|printf|\bpublic\s+(static|void|class)\b|"
    r"=>|\breturn\b|;\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# --- Per-category phrasings, generous coverage ------------------------------
_KEYWORDS: dict[str, list[str]] = {
    "code_debug": [
        r"\bbug\b", r"\bdebug\b",
        r"\bfix (this|the|my|it|a|an|these)?\s*(code|function|snippet|program|method|script|error)?\b",
        r"\bwhat'?s wrong\b", r"\bwhat is wrong\b", r"\bwhat went wrong\b",
        r"\bwhy (does|is)n'?t (this|it|my|the)\b", r"\bwhy (does|is) (this|it|my|the)\b.*\b(fail|error|crash|wrong|not)\b",
        r"\berror in\b", r"\btraceback\b", r"\bstack ?trace\b",
        r"\bthrows? an? (error|exception)\b", r"\braises? (an? )?(error|exception)\b",
        r"\b(runs?|loops?) forever\b", r"\binfinite loop\b",
        r"\breturns? \w+ instead\b", r"\bwrong (output|result|answer)\b",
        r"\bincorrect (output|result|behavior)\b", r"\bdoes ?n'?t work\b", r"\bnot working\b",
        r"\bcorrect(ed)? (version|implementation|code)\b", r"\bcrash(es|ing|ed)?\b",
    ],
    "code_gen": [
        r"\b(write|create|produce|build|give me|implement|generate|make|need|design)\b[^.]{0,40}"
        r"\b(a |an |the |me a )?(\w+\s+)?(function|program|script|method|class|routine|module|snippet|code)\b",
        r"\bimplement (a |an |the )?\w+", r"\bgenerate (code|a function)\b",
        r"\bcode that\b", r"\bfunction (that|to|which)\b", r"\bscript (that|to)\b",
        r"\bmethod (that|to)\b", r"\bwrite code\b", r"\bin (python|java|c\+\+|javascript|go|rust|c#)\b",
    ],
    "sentiment": [
        r"\bsentiment\b", r"\bpositive or negative\b", r"\bpositive, negative\b",
        r"\bnegative or positive\b", r"\bclassify the (tone|emotion|sentiment|mood)\b",
        r"\bis this (review|tweet|comment|message|text|post)\b.*\b(positive|negative|good|bad)\b",
        r"\b(positive|negative|neutral)\s+sentiment\b", r"\bemotional tone\b",
        r"\btone of (this|the|that|it)\b", r"\bhow (positive|negative)\b",
        r"\b(mood|emotion|attitude|feeling) (of|in|behind) (this|the|that|it)\b",
        r"\brate the (mood|tone|sentiment)\b", r"\bis the (author|writer|speaker) (happy|angry|upset|sad|pleased)\b",
        r"\b(happy|upset|angry|sad|positive|negative) or\b",
    ],
    "ner": [
        r"\bnamed entit", r"\bentities\b",
        r"\bextract (all |every |each |the )*(entit|name|person|people|organi|compan|location|place|date)",
        r"\blist (all )?(the )?(people|persons?|organi[sz]ations?|companies|locations?|places?|dates?)\b",
        r"\bidentify (all )?(the )?(person|people|organi|compan|location|place|date|entit)",
        r"\b(person|people|org|organization|location|place|date)s?\s*[:=]",
        r"\b(mentioned|named|referenced) in (this|the|below|following)",
        r"\bpull out (every|all|the)\b", r"\bwho and what\b",
        r"\b(company|people|place|person|organization) names?\b",
        r"\bfind (all )?(the )?(names?|people|organi|locations?|dates?)\b",
    ],
    "summarization": [
        r"\bsummari[sz]e\b", r"\bsummary\b", r"\btl;?dr\b", r"\bcondense\b",
        r"\bin (one|a single|two|three|a few|\d+) sentences?\b", r"\bin \d+ words?\b",
        r"\bshorten\b", r"\bkey points?\b", r"\bthe gist\b", r"\bboil (it|this|that|.*) down\b",
        r"\bmain (idea|point|takeaway|message)", r"\bin a (single|one) line\b",
        r"\bgive me the (gist|summary|highlights|key)\b", r"\bhighlights\b",
        r"\bwhat is (this|the (text|article|passage|paragraph)) about\b",
        r"\brecap\b", r"\bbrief overview\b", r"\bsum(marize)? up\b",
    ],
    "logic": [
        r"\bpuzzle\b", r"\briddle\b",
        r"\bwho (is|are|owns?|sits?|lives?|has|had|drinks?|wears?|likes?|finished|won|did it)\b",
        r"\bif and only if\b", r"\bexactly one\b", r"\bat least one\b", r"\bat most one\b",
        r"\beach (person|house|box|day|one|of them)\b.*\b(exactly|only|one|different)\b",
        r"\bconstraints?\b", r"\bdeduce\b", r"\blogically\b", r"\bmust be true\b",
        r"\bthe following (clues|facts|statements|conditions)\b",
        r"\beach (have|has|own|owns|is|are)?\s*a different\b",
        r"\bif all \w+ are\b", r"\b(definitely|necessarily|must) (be )?(true|false|follows?)\b",
        r"\bknights?\b.{0,30}\b(knaves?|liars?)\b", r"\balways (lie|tell the truth)\b",
        r"\btaller than\b", r"\bshorter than\b", r"\bolder than\b", r"\byounger than\b",
        r"\bfinished (before|after)\b", r"\b(ranked|ordered|arranged) (from|in)\b",
        r"\beither\b.{0,60}\bor\b.{0,80}\b(who|which|what)\b",
    ],
    "math": [
        r"\bcalculate\b", r"\bcompute\b", r"\bhow (much|many|far|old|long|fast)\b",
        r"\bpercent", r"\b\d+\s*%", r"\bsum of\b", r"\baverage\b", r"\bmean of\b",
        r"\bproduct of\b", r"\bdifference between\b", r"\bremainder\b",
        r"\bwhat is \d", r"\bwhat'?s \d", r"\b\d+\s*[+\-*/x×÷^]\s*\d+",
        r"\bsolve for\b", r"\bsolve the equation\b",
        r"\btotal (cost|price|amount|number|of)\b", r"\bround(ed)? (to|off)\b",
        r"\bdecimal (place|point)", r"\bratio\b", r"\b\d+\s*:\s*\d+",
        r"\b(interest|discount|profit|tax|tip)\b.*\b\d", r"\bper (hour|day|week|month|year|unit|item)\b",
        r"\bfind the (largest|smallest|value|angle|area|perimeter|sum|total|number|result)\b",
        r"\bhow (much|many) (is|are|would|will|does)\b",
    ],
    "factual": [
        r"\bwhat is\b", r"\bwhat are\b", r"\bwhat was\b", r"\bwho (was|were|is|are)\b",
        r"\bwhen (did|was|is|were)\b", r"\bwhere (is|was|are|were)\b",
        r"\bwhy (is|do|does|did|are)\b", r"\bhow (do|does|did|is|are)\b",
        r"\bexplain\b", r"\bdefine\b", r"\bdescribe\b", r"\bwhat does .* mean\b",
        r"\btell me about\b", r"\bwhat happens\b",
    ],
}

# Priority: specific/high-signal first, so a summarization prompt that also
# contains "what is" still routes to summarization, and code beats math on a
# "write a function that sums..." prompt.
_PRIORITY = ["code_debug", "code_gen", "sentiment", "ner", "summarization", "logic", "math", "factual"]

_COMPILED: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in _KEYWORDS.items()
}


def _has_code(text: str) -> bool:
    return bool(_CODE_FENCE.search(text) or _CODE_SYNTAX.search(text))


def classify(task: Task) -> str:
    """Best-guess category in a single priority-ordered heuristic pass."""
    if task.type:
        return task.type
    text = task.input or ""

    for cat in _PRIORITY:
        if any(rx.search(text) for rx in _COMPILED[cat]):
            # Math patterns are noisy on incidental digits; require the prompt to
            # actually contain a number before committing to math.
            if cat == "math" and not re.search(r"\d", text):
                continue
            return cat

    # No phrasing matched: raw code present is almost always a debugging ask.
    return "code_debug" if _has_code(text) else "factual"
