"""Remote backend for the Fireworks AI OpenAI-compatible API. Billed tokens."""
from __future__ import annotations

import os

from .base import Generation

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

MODEL_PREFIX = "accounts/fireworks/models/"


def normalize_base_url(url: str) -> str:
    """The harness may supply the full endpoint; the SDK appends the path
    itself, so a naive join would 404 every call."""
    url = url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


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
        resolved_base = normalize_base_url(os.environ.get("FIREWORKS_BASE_URL") or base_url)
        self._client = OpenAI(
            api_key=key, base_url=resolved_base, timeout=timeout, max_retries=max_retries
        )

    def generate(self, system, user, *, model, temperature=0.0, max_tokens=64,
                 reasoning_effort=""):
        from openai import BadRequestError, NotFoundError

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        extra = {}
        if reasoning_effort:
            # Reasoning models (gpt-oss, minimax) bill hidden reasoning tokens;
            # capping the effort is the single biggest completion-token saver.
            extra["reasoning_effort"] = reasoning_effort

        def call(model_id, extra_body):
            return self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )

        try:
            resp = call(model, extra)
        except BadRequestError:
            if not extra:
                raise
            # Non-reasoning models may reject reasoning_effort outright;
            # dropping the param must never cost the task.
            resp = call(model, None)
        except NotFoundError:
            # ALLOWED_MODELS may arrive as bare ids while serving ids are
            # account-scoped. Retry once with the canonical prefix.
            if model.startswith(MODEL_PREFIX):
                raise
            resp = call(MODEL_PREFIX + model, extra)

        usage = resp.usage
        choice = resp.choices[0]
        return Generation(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "",
        )
