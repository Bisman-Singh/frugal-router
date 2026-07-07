"""Remote backend for the Fireworks AI OpenAI-compatible API. Billed tokens."""
from __future__ import annotations

import os

from .base import Generation

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksBackend:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 25.0,
        max_retries: int = 1,
    ):
        from openai import OpenAI

        key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. Export it or put it in the environment."
            )
        # The judging harness routes and records all traffic through its own
        # base URL; calls that bypass it invalidate the submission.
        resolved_base = os.environ.get("FIREWORKS_BASE_URL") or base_url
        self._client = OpenAI(
            api_key=key, base_url=resolved_base, timeout=timeout, max_retries=max_retries
        )

    def generate(self, system, user, *, model, temperature=0.0, max_tokens=64):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = resp.usage
        return Generation(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
