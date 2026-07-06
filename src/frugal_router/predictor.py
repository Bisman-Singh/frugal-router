"""Learned failure predictor: P(the local model gets this task wrong).

Trained on the local model's actual mistakes on a labeled eval set, not on a
generic notion of difficulty. TF-IDF keeps the dependency footprint small and
works at the few-hundred-example scale available at kickoff.
"""
from __future__ import annotations

from pathlib import Path

MIN_EXAMPLES = 30


class FailurePredictor:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], local_correct: list[bool]) -> "FailurePredictor":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        if len(texts) != len(local_correct):
            raise ValueError("texts and labels differ in length")
        if len(texts) < MIN_EXAMPLES:
            raise ValueError(
                f"need at least {MIN_EXAMPLES} labeled examples, got {len(texts)}"
            )
        y = [0 if ok else 1 for ok in local_correct]
        if len(set(y)) < 2:
            raise ValueError("need both correct and incorrect examples to train")
        pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        pipeline.fit(texts, y)
        return cls(pipeline)

    def p_fail(self, text: str) -> float:
        return float(self._pipeline.predict_proba([text])[0][1])

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "FailurePredictor | None":
        import joblib

        path = Path(path)
        if not path.exists():
            return None
        return cls(joblib.load(path))
