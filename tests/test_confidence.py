import pytest

from frugal_router.confidence import (
    ConfidenceReport,
    combine,
    logprob_quantile,
    vote,
)


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


def test_logprob_quantile_is_pessimistic():
    # One very unsure token in an otherwise confident answer must show through.
    confident = [-0.01] * 9
    with_doubt = sorted([-3.0] + [-0.01] * 9)
    assert logprob_quantile(with_doubt, q=0.0) == -3.0
    assert logprob_quantile(confident) == -0.01
    assert logprob_quantile(None) is None
    assert logprob_quantile([]) is None


def make_report(**kwargs):
    base = dict(candidate="4", agreement=1.0, n_samples=3, format_valid=True)
    base.update(kwargs)
    return ConfidenceReport(**base)


def test_combine_renormalizes_over_available_signals():
    assert combine(make_report()) == pytest.approx(1.0)


def test_combine_all_signals_perfect():
    report = make_report(logprob=0.0, p_fail=0.0)
    assert combine(report) == pytest.approx(1.0)


def test_combine_high_p_fail_drags_score():
    report = make_report(logprob=0.0, p_fail=1.0)
    assert combine(report) == pytest.approx(0.85)
