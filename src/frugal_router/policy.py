"""Per-task-type routing policies.

One global threshold either wastes tokens on the easy categories or bleeds
accuracy on the hard ones, so every knob lives here, per type.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class Policy:
    task_type: str = "general"
    n_samples: int = 3
    sample_temperature: float = 0.7
    local_max_tokens: int = 512
    escalation_threshold: float = 0.65
    p_fail_cutoff: float = 0.65
    use_verification: bool = True
    remote_model: str = ""  # empty = the configured default remote model
    remote_max_tokens: int = 32
    remote_cot: bool = False
    always_remote: bool = False
    compress_over_chars: int = 6000
    format_spec: str = ""  # empty = the default spec for the task type
    few_shot: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict, task_type: str = "general") -> "Policy":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs["task_type"] = task_type
        return cls(**kwargs)


class PolicyBook:
    """Per-type policies with shared defaults."""

    def __init__(self, defaults: dict | None = None, per_type: dict[str, dict] | None = None):
        self._defaults = defaults or {}
        self._per_type = per_type or {}

    def for_type(self, task_type: str) -> Policy:
        merged = dict(self._defaults)
        merged.update(self._per_type.get(task_type, {}))
        return Policy.from_dict(merged, task_type=task_type)
