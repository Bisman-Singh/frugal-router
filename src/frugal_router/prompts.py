"""Prompt builders.

Local prompts may be verbose because local tokens are free. Remote prompts must
be minimal because every remote token counts against the score, but never so
terse that the answer fails the intent judge.
"""
from __future__ import annotations

from .contracts import contract_for


def question_text(task_input: str, context: str | None) -> str:
    if not context:
        return task_input
    return f"{context.strip()}\n\nQuestion: {task_input}"


def local_solve(
    question: str, category: str, few_shot: list[dict] | None = None
) -> tuple[str, str]:
    contract = contract_for(category)
    system = (
        f"You are an expert at {category} tasks. Work carefully. "
        f"{contract.local_instruction}"
    )
    parts = []
    for example in few_shot or []:
        parts.append(f"{example['input']}\n{example['answer']}\n")
    parts.append(question)
    return system, "\n".join(parts)


def compression(question: str, context: str) -> tuple[str, str]:
    system = (
        "From the text below, extract only the minimal set of sentences needed to "
        "answer the question. Output just that excerpt, nothing else."
    )
    return system, f"Question: {question}\nText:\n{context}"


def remote_solve(question: str, category: str, cot: bool = False) -> str:
    contract = contract_for(category)
    if cot:
        return f"{question}\n\n{contract.remote_instruction}"
    return f"{question}\n\n{contract.remote_instruction} No preamble."


def remote_with_draft(question: str, category: str, draft: str) -> str:
    """Minions-style collaboration for the case where every scored answer must
    come from a Fireworks call: the local draft rides along so a correct draft
    costs only its confirmation."""
    contract = contract_for(category)
    return (
        f"{question}\n\n"
        f"Draft answer: {draft}\n"
        f"If the draft is correct and complete, output it verbatim. "
        f"Otherwise output a corrected answer. {contract.remote_instruction}"
    )


def reformat(raw_answer: str, category: str) -> tuple[str, str]:
    contract = contract_for(category)
    system = (
        "Reshape the answer to the required format without changing its meaning. "
        f"{contract.remote_instruction} Output only the reshaped answer."
    )
    return system, raw_answer
