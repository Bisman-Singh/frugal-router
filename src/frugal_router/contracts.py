"""The Track 1 capability categories and their judge-shaped answer contracts.

The accuracy gate is an LLM judge scoring answers against expected intent, so
answers must be intent-complete: a sentiment call needs its justification, a
debugging answer needs the corrected code, a summary must obey the stated
constraints. Over-terse answers fail the gate, and failing the gate is worth
infinitely more than the tokens a terse answer saves.
"""
from __future__ import annotations

from dataclasses import dataclass

STYLE_LINE = "line"  # reason freely, finish with an "Answer:" line that is extracted
STYLE_FULL = "full"  # the whole response is the answer


@dataclass(frozen=True)
class Contract:
    name: str
    style: str
    local_instruction: str  # verbose is fine, local tokens are free
    remote_instruction: str  # every word here is billed


CONTRACTS: dict[str, Contract] = {
    "factual": Contract(
        "factual",
        STYLE_LINE,
        "Answer the question accurately. Think briefly if needed, then output the "
        "final line as exactly: Answer: <the direct answer, one short sentence>",
        "Answer directly in one short sentence.",
    ),
    "math": Contract(
        "math",
        STYLE_LINE,
        "Solve the problem step by step, showing your working. Then output the "
        "final line as exactly: Answer: <the final number>",
        "Solve step by step, then end with the line: Answer: <the final number>",
    ),
    "sentiment": Contract(
        "sentiment",
        STYLE_LINE,
        "Decide the sentiment. Output the final line as exactly: "
        "Answer: <positive/negative/neutral/mixed> - <one sentence of justification "
        "citing the wording of the text>",
        "Reply with the sentiment label and one sentence of justification.",
    ),
    "summarization": Contract(
        "summarization",
        STYLE_FULL,
        "Write the summary, following any length or format constraint in the task "
        "exactly (sentence count, word limit, bullet points). Output only the summary.",
        "Output only the summary, following the stated length and format constraints exactly.",
    ),
    "ner": Contract(
        "ner",
        STYLE_FULL,
        "Extract every named entity and label each with its type: person, "
        "organization, location, or date. List one entity per line as "
        "'<type>: <entity>'. Output only the list.",
        "List every named entity, one per line, as '<type>: <entity>' using the "
        "types person, organization, location, date.",
    ),
    "logic": Contract(
        "logic",
        STYLE_LINE,
        "Work through the constraints step by step. Then output the final line as "
        "exactly: Answer: <the conclusion, as briefly as the question allows>",
        "Reason step by step, then end with the line: Answer: <the conclusion>",
    ),
    "code_debug": Contract(
        "code_debug",
        STYLE_FULL,
        "Identify the bug in one sentence, then provide the complete corrected code "
        "in a fenced code block. Nothing else.",
        "State the bug in one sentence, then give the complete corrected code in a "
        "fenced code block.",
    ),
    "code_gen": Contract(
        "code_gen",
        STYLE_FULL,
        "Write the complete, runnable code that satisfies the specification. Output "
        "only a fenced code block, no commentary.",
        "Output only the complete code in a fenced code block.",
    ),
}

_DEFAULT = CONTRACTS["factual"]


def contract_for(category: str) -> Contract:
    return CONTRACTS.get(category, _DEFAULT)


def style_of(category: str) -> str:
    return contract_for(category).style
