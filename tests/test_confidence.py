import pytest

from frugal_router.confidence import ConfidenceReport, combine, vote


def test_vote_unanimous():
    assert vote(["4", "4", "4"]) == ("4", 1.0)


def test_vote_split():
    candidate, agreement = vote(["4", "5", "4"])
    assert candidate == "4"
    assert agreement == pytest.approx(2 / 3)


def test_vote_all_invalid():
    assert vote([None, None]) == (None, 0.0)


def test_invalid_samples_count_against_agreement():
    _, agreement = vote(["4", None, "4"])
    assert agreement == pytest.approx(2 / 3)


def make_report(**kwargs):
    base = dict(candidate="4", agreement=1.0, n_samples=3, format_valid=True)
    base.update(kwargs)
    return ConfidenceReport(**base)


def test_combine_renormalizes_over_available_signals():
    assert combine(make_report()) == pytest.approx(1.0)


def test_combine_all_signals_perfect():
    report = make_report(mean_logprob=0.0, verify_yes_prob=1.0, p_fail=0.0)
    assert combine(report) == pytest.approx(1.0)


def test_combine_high_p_fail_drags_score():
    report = make_report(mean_logprob=0.0, verify_yes_prob=1.0, p_fail=1.0)
    assert combine(report) == pytest.approx(0.8)
