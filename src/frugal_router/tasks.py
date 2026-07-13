"""Task model and JSONL dataset loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TASK_TYPES = (
    "factual",
    "math",
    "sentiment",
    "summarization",
    "ner",
    "logic",
    "code_debug",
    "code_gen",
)

_CHOICE_LETTERS = "ABCDE"


@dataclass
class Task:
    id: str
    input: str
    type: str | None = None
    context: str | None = None
    choices: list[str] | None = None
    expected: str | list[str] | None = None
    grader: str | None = None  # exact | numeric | contains_all; inferred from type when unset

    @classmethod
    def from_dict(cls, raw: dict) -> "Task":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def rendered_input(self) -> str:
        """The question text as shown to a model, with choices spelled out."""
        if not self.choices:
            return self.input
        options = "\n".join(
            f"{_CHOICE_LETTERS[i]}. {choice}" for i, choice in enumerate(self.choices)
        )
        return f"{self.input}\n{options}"


def load_tasks(path: str | Path) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw.setdefault("id", f"task-{line_no}")
            tasks.append(Task.from_dict(raw))
    return tasks
