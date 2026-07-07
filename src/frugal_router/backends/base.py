"""Backend interfaces shared by real and mock implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Generation:
    text: str
    token_logprobs: list[float] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""  # "length" means truncated at max_tokens


@runtime_checkable
class LocalBackend(Protocol):
    """Free inference. Spend it lavishly."""

    def generate(
        self,
        system: str | None,
        user: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> list[Generation]: ...


@runtime_checkable
class RemoteBackend(Protocol):
    """Billed inference. Every token counts against the score."""

    def generate(
        self,
        system: str | None,
        user: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> Generation: ...
