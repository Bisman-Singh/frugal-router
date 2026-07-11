"""Local answer tier: a baked GGUF model answering the cheap categories free.

Scoring counts only remote tokens, so every task the local model answers
confidently is free. The tier is gated hard: an answer is kept ONLY when it
passes the same category validation as remote answers AND the model's own
self-verification; anything doubtful escalates to the remote path. A
confident-but-wrong local answer is the only way this tier can cost accuracy,
so the gate errs toward escalation.

The model handle is a lazy singleton behind a lock (llama.cpp contexts are not
concurrent); loading failure disables the tier for the whole run rather than
crashing it.
"""
from __future__ import annotations

import os
import re
import threading
import time

_LOCK = threading.Lock()
_LLM = None
_DISABLED = False

_SUFFIX = os.environ.get("LOCAL_SUFFIX", "")  # e.g. "/no_think" for hybrid-thinking models
_THINK_RE = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")

# max new tokens per category: answers here are short by construction.
# code_gen/math are intentionally NOT here: ast.parse is a syntax check, not a
# correctness check ("valid but wrong" sails through), and code/math are
# execution-graded. They stay remote until an EXECUTION gate (run generated code
# against extracted/inferred tests) proves per-category local accuracy on the
# dev set above threshold+margin. Data decides, not the architecture diagram.
CAPS = {"sentiment": 40, "factual": 100, "summarization": 70, "ner": 90, "logic": 80}
CATEGORIES = frozenset(CAPS)


def _load():
    global _LLM, _DISABLED
    if _LLM is not None or _DISABLED:
        return _LLM
    path = os.environ.get("LOCAL_MODEL_PATH", "models/local.gguf")
    if not os.path.exists(path):
        _DISABLED = True
        return None
    try:
        from llama_cpp import Llama

        _LLM = Llama(
            model_path=path,
            n_ctx=int(os.environ.get("LOCAL_CTX", "2048")),
            n_threads=int(os.environ.get("LOCAL_THREADS", "0")) or None,
            verbose=False,
        )
    except Exception as exc:  # tier off, run continues remote-only
        print(f"local tier disabled: {type(exc).__name__}: {exc}", flush=True)
        _DISABLED = True
    return _LLM


def available() -> bool:
    return _load() is not None


def generate(system: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
    """One serialized local chat completion; '' on any failure."""
    llm = _load()
    if llm is None:
        return ""
    with _LOCK:
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": (system + " " + _SUFFIX).strip()},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (out["choices"][0]["message"]["content"] or "").strip()
            return _THINK_RE.sub("", text).strip()
        except Exception:
            return ""


def verify(prompt: str, answer: str) -> bool:
    """Self-verification: the model judges its own answer YES/NO. Anything
    that is not an unambiguous YES escalates."""
    llm = _load()
    if llm is None or not answer:
        return False
    question = (
        f"TASK:\n{prompt}\n\nANSWER:\n{answer}\n\n"
        "Is the answer factually correct, complete, and properly formatted for "
        "the task? Reply with exactly YES or NO."
    )
    with _LOCK:
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": ("You are a strict, skeptical grader. " + _SUFFIX).strip()},
                          {"role": "user", "content": question}],
                max_tokens=16,
                temperature=0.0,
            )
            text = (out["choices"][0]["message"]["content"] or "").strip()
            text = _THINK_RE.sub("", text).strip().upper()
            return text.startswith("YES")
        except Exception:
            return False


def timed_generate(system: str, prompt: str, max_tokens: int,
                   temperature: float = 0.0) -> tuple[str, float]:
    t0 = time.monotonic()
    text = generate(system, prompt, max_tokens, temperature)
    return text, time.monotonic() - t0
