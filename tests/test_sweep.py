from frugal_router.sweep import pareto, replay, sweep


def make_record(score, local_ok, remote_ok, tokens=50, p_fail=None, valid=True):
    return {
        "conf_score": score,
        "conf_format_valid": valid,
        "conf_p_fail": p_fail,
        "local_correct": local_ok,
        "remote_correct": remote_ok,
        "remote_probe_prompt_tokens": tokens,
        "remote_probe_completion_tokens": 5,
    }


RECORDS = [
    make_record(0.9, True, True),   # easy: local right
    make_record(0.2, False, True),  # hard: local wrong, remote right
]


def test_replay_escalates_only_below_threshold():
    row = replay(RECORDS, 0.5, 1.0)
    assert row["accuracy"] == 1.0
    assert row["remote_tokens"] == 55


def test_replay_invalid_format_always_escalates():
    rows = [make_record(0.99, True, True, valid=False)]
    assert replay(rows, 0.0, 1.0)["remote_tokens"] == 55


def test_replay_p_fail_cutoff():
    rows = [make_record(0.9, False, True, p_fail=0.95)]
    row = replay(rows, 0.5, 0.9)
    assert row["accuracy"] == 1.0
    assert row["remote_tokens"] == 55


def test_sweep_recommends_cheapest_above_target():
    rows, recommendation = sweep(RECORDS, [0.0, 0.5, 1.0], [1.0], 0.9, margin=0.0)
    assert recommendation["threshold"] == 0.5
    assert recommendation["remote_tokens"] == 55
    assert recommendation["accuracy"] == 1.0


def test_pareto_frontier_monotonic():
    rows, _ = sweep(RECORDS, [0.0, 0.5, 1.0], [1.0], 0.9, margin=0.0)
    frontier = pareto(rows)
    tokens = [r["remote_tokens"] for r in frontier]
    accuracies = [r["accuracy"] for r in frontier]
    assert tokens == sorted(tokens)
    assert accuracies == sorted(accuracies)
