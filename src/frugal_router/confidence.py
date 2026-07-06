"""Confidence signals and their combination into one escalation score."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ConfidenceReport:
    candidate: str | None  # normalized local answer
    agreement: float
    n_samples: int
    mean_logprob: float | None = None
    verify_yes_prob: float | None = None
    p_fail: float | None = None
    format_valid: bool = False
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "agreement": self.agreement,
            "n_samples": self.n_samples,
            "mean_logprob": self.mean_logprob,
            "verify_yes_prob": self.verify_yes_prob,
            "p_fail": self.p_fail,
            "format_valid": self.format_valid,
            "score": self.score,
        }


DEFAULT_WEIGHTS = {"agreement": 0.4, "logprob": 0.2, "verify": 0.2, "predictor": 0.2}


def vote(normalized_candidates: list[str | None]) -> tuple[str | None, float]:
    """Majority vote over normalized answers. Agreement is top count over all samples."""
    valid = [c for c in normalized_candidates if c]
    if not valid:
        return None, 0.0
    top, count = Counter(valid).most_common(1)[0]
    return top, count / len(normalized_candidates)


def combine(report: ConfidenceReport, weights: dict | None = None) -> float:
    """Weighted mean of the signals that are actually available, in [0, 1]."""
    weights = weights or DEFAULT_WEIGHTS
    signals: dict[str, float] = {"agreement": report.agreement}
    if report.mean_logprob is not None:
        signals["logprob"] = math.exp(report.mean_logprob)
    if report.verify_yes_prob is not None:
        signals["verify"] = report.verify_yes_prob
    if report.p_fail is not None:
        signals["predictor"] = 1.0 - report.p_fail
    total_weight = sum(weights.get(k, 0.0) for k in signals)
    if total_weight <= 0:
        return 0.0
    return sum(weights.get(k, 0.0) * v for k, v in signals.items()) / total_weight
