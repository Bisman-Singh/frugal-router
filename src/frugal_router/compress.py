"""Local context compression. Free reduction of remote input tokens."""
from __future__ import annotations

from . import prompts


def maybe_compress(
    local, question: str, context: str | None, over_chars: int
) -> tuple[str | None, bool]:
    """Return (context to use, whether it was compressed)."""
    if not context or len(context) <= over_chars or local is None:
        return context, False
    system, user = prompts.compression(question, context)
    try:
        gens = local.generate(
            system, user, n=1, temperature=0.0,
            max_tokens=min(1024, max(256, over_chars // 8)),
        )
    except Exception:
        return context, False
    excerpt = (gens[0].text or "").strip()
    if not excerpt or len(excerpt) >= len(context):
        return context, False
    return excerpt, True
