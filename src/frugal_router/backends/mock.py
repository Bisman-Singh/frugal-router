"""Deterministic mock backends for tests and offline development."""
from __future__ import annotations

from .base import Generation


class MockLocalBackend:
    """Replays a queue of scripted responses.

    Each queue item serves one generate() call: a string is replicated n times,
    a list is padded with its last element to length n.
    """

    def __init__(
        self,
        responses: list | None = None,
        token_logprobs: list[float] | None = (-0.05, -0.05, -0.05),
        fail: bool = False,
    ):
        self._queue = list(responses or [])
        self.token_logprobs = list(token_logprobs) if token_logprobs else None
        self.fail = fail
        self.calls: list[dict] = []

    def generate(self, system, user, *, n=1, temperature=0.0, max_tokens=512):
        if self.fail:
            raise RuntimeError("mock local backend down")
        self.calls.append({"system": system, "user": user, "n": n, "temperature": temperature})
        item = self._queue.pop(0) if self._queue else "Answer: unknown"
        texts = [item] * n if isinstance(item, str) else list(item)
        while len(texts) < n:
            texts.append(texts[-1])
        return [
            Generation(text=t, token_logprobs=self.token_logprobs, completion_tokens=len(t) // 4)
            for t in texts[:n]
        ]


class MockRemoteBackend:
    def __init__(
        self,
        replies: list[str] | None = None,
        prompt_tokens: int = 40,
        completion_tokens: int = 6,
        fail: bool = False,
    ):
        self._replies = list(replies or ["Answer: 42"])
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.fail = fail
        self.calls: list[dict] = []

    def generate(self, system, user, *, model, temperature=0.0, max_tokens=64):
        if self.fail:
            raise RuntimeError("mock remote backend down")
        self.calls.append({"system": system, "user": user, "model": model, "max_tokens": max_tokens})
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return Generation(
            text=reply,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
