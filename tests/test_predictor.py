import pytest

from frugal_router.predictor import FailurePredictor

EASY = [f"add {i} and {i + 1} together" for i in range(20)]
HARD = [f"prove theorem {i} about quantum manifolds" for i in range(20)]


def test_train_predict_save_load(tmp_path):
    predictor = FailurePredictor.train(EASY + HARD, [True] * 20 + [False] * 20)
    easy_p = predictor.p_fail("add 3 and 4 together")
    hard_p = predictor.p_fail("prove a theorem about quantum manifolds")
    assert hard_p > easy_p

    path = tmp_path / "predictor.joblib"
    predictor.save(path)
    loaded = FailurePredictor.load(path)
    assert loaded.p_fail("add 3 and 4 together") == pytest.approx(easy_p)


def test_train_requires_enough_examples():
    with pytest.raises(ValueError):
        FailurePredictor.train(["x"] * 5, [True] * 5)


def test_train_requires_both_classes():
    with pytest.raises(ValueError):
        FailurePredictor.train(EASY + HARD, [True] * 40)


def test_load_missing_returns_none(tmp_path):
    assert FailurePredictor.load(tmp_path / "missing.joblib") is None
