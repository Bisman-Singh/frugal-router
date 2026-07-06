"""Per-task decision and token ledger. JSONL on disk, running totals in memory."""
from __future__ import annotations

import json
from pathlib import Path


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def record(self, entry: dict) -> None:
        self.entries.append(entry)
        if self._path:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        total = len(self.entries)
        sources = [e.get("source") for e in self.entries]
        return {
            "tasks": total,
            "local_answers": sources.count("local"),
            "remote_answers": sources.count("remote"),
            "fallback_answers": sources.count("fallback"),
            "remote_prompt_tokens": sum(e.get("remote_prompt_tokens", 0) for e in self.entries),
            "remote_completion_tokens": sum(
                e.get("remote_completion_tokens", 0) for e in self.entries
            ),
        }
