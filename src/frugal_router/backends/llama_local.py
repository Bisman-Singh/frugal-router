"""llama.cpp local backend. Loads once, stays warm, exposes token logprobs."""
from __future__ import annotations

from .base import Generation


class LlamaLocalBackend:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
    ):
        from llama_cpp import Llama  # imported lazily so tests run without the wheel

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or None,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,  # None = use the template from GGUF metadata
            verbose=False,
        )
        self.n_ctx = n_ctx
        self._supports_logprobs = True

    def _chat(self, system: str | None, user: str, **kwargs) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        if not self._supports_logprobs:
            kwargs.pop("logprobs", None)
            kwargs.pop("top_logprobs", None)
        try:
            return self._llm.create_chat_completion(messages=messages, **kwargs)
        except TypeError:
            # Older llama-cpp-python without logprobs in the chat API.
            self._supports_logprobs = False
            kwargs.pop("logprobs", None)
            kwargs.pop("top_logprobs", None)
            return self._llm.create_chat_completion(messages=messages, **kwargs)

    def generate(self, system, user, *, n=1, temperature=0.0, max_tokens=512):
        generations = []
        for _ in range(n):
            resp = self._chat(
                system,
                user,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=1,
            )
            choice = resp["choices"][0]
            usage = resp.get("usage") or {}
            generations.append(
                Generation(
                    text=choice.get("message", {}).get("content") or "",
                    token_logprobs=_token_logprobs(choice),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason") or "",
                )
            )
        return generations


def _token_logprobs(choice: dict) -> list[float] | None:
    content = (choice.get("logprobs") or {}).get("content") or []
    values = [t["logprob"] for t in content if t.get("logprob") is not None]
    return values or None
