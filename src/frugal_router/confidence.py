"""Confidence signals and their combination into one escalation score.

The stack is deliberately small. Research on small instruct models ranks
self-consistency agreement as the strongest cheap signal, answer-span logprob
as a usable tiebreaker when aggregated pessimistically (quantile, not mean),
and yes/no self-verification as degenerate at this scale, so it is not here.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class ConfidenceReport:
    candidate: str | None  # normalized vote key of the winning local answer
    agreement: float
    n_samples: int
    logprob: float | None = None  # pessimistic quantile over answer token logprobs
    p_fail: float | None = None
    format_valid: bool = False
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "agreement": self.agreement,
            "n_samples": self.n_samples,
            "logprob": self.logprob,
            "p_fail": self.p_fail,
            "format_valid": self.format_valid,
            "score": self.score,
        }


DEFAULT_WEIGHTS = {"agreement": 0.6, "logprob": 0.25, "predictor": 0.15}

LOGPROB_QUANTILE = 0.25


def vote(normalized_candidates: list[str | None]) -> tuple[str | None, float]:
    """Majority vote over normalized answers. Agreement is top count over all samples."""
    valid = [c for c in normalized_candidates if c]
    if not valid:
        return None, 0.0
    top, count = Counter(valid).most_common(1)[0]
    return top, count / len(normalized_candidates)


def logprob_quantile(token_logprobs: list[float] | None, q: float = LOGPROB_QUANTILE) -> float | None:
    """Pessimistic aggregate of per-token logprobs. Mean hides one very unsure
    token inside a long confident answer; a low quantile does not."""
    if not token_logprobs:
        return None
    ordered = sorted(token_logprobs)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[index]


def combine(report: ConfidenceReport, weights: dict | None = None) -> float:
    """Weighted mean of the signals that are actually available, in [0, 1]."""
    weights = weights or DEFAULT_WEIGHTS
    signals: dict[str, float] = {"agreement": report.agreement}
    if report.logprob is not None:
        signals["logprob"] = math.exp(report.logprob)
    if report.p_fail is not None:
        signals["predictor"] = 1.0 - report.p_fail
    total_weight = sum(weights.get(k, 0.0) for k in signals)
    if total_weight <= 0:
        return 0.0
    return sum(weights.get(k, 0.0) * v for k, v in signals.items()) / total_weight
