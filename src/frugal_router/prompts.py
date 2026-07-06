"""Prompt builders.

Local prompts may be verbose because local tokens are free. Remote prompts must
be minimal because every remote token counts against the score.
"""
from __future__ import annotations

FORMAT_SPECS = {
    "math": "the final number",
    "mcq": "the letter of the correct option",
    "classification": "the label",
    "extraction": "the exact value from the text",
    "summarization": "a summary of at most two sentences",
    "general": "the answer, as briefly as possible",
}


def format_spec_for(task_type: str, override: str | None = None) -> str:
    return override or FORMAT_SPECS.get(task_type, FORMAT_SPECS["general"])


def question_text(task_input: str, context: str | None) -> str:
    if not context:
        return task_input
    return f"{context.strip()}\n\nQuestion: {task_input}"


def local_solve(
    question: str,
    task_type: str,
    format_spec: str,
    few_shot: list[dict] | None = None,
) -> tuple[str, str]:
    system = (
        f"You are an expert at {task_type} tasks. Work carefully. "
        "Think through the problem step by step, then output the final line as exactly: "
        f"Answer: <{format_spec}>"
    )
    parts = []
    for example in few_shot or []:
        parts.append(f"{example['input']}\nAnswer: {example['answer']}\n")
    parts.append(question)
    return system, "\n".join(parts)


def verification(question: str, candidate: str) -> str:
    return (
        f"Question: {question}\n"
        f"Proposed answer: {candidate}\n"
        "Is the proposed answer correct? Reply with only one word: yes or no."
    )


def compression(question: str, context: str) -> tuple[str, str]:
    system = (
        "From the text below, extract only the minimal set of sentences needed to "
        "answer the question. Output just that excerpt, nothing else."
    )
    return system, f"Question: {question}\nText:\n{context}"


def remote_minimal(question: str, format_spec: str) -> str:
    return f"{question}\n\nAnswer with only {format_spec}. No explanation."


def remote_cot(question: str, format_spec: str) -> str:
    return (
        f"{question}\n\n"
        f"Think step by step, then output the final line as: Answer: <{format_spec}>"
    )


def reformat(raw_answer: str, format_spec: str) -> tuple[str, str]:
    system = (
        "Reformat the answer to the requested format. "
        "Output only the formatted answer, nothing else."
    )
    return system, f"Format: {format_spec}\nAnswer: {raw_answer}"
