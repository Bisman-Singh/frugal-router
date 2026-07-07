from frugal_router.agent import RoutingAgent
from frugal_router.backends.mock import MockLocalBackend, MockRemoteBackend
from frugal_router.ledger import Ledger
from frugal_router.policy import PolicyBook
from frugal_router.tasks import Task

MATH = Task(id="m1", input="What is 12 + 7?", type="math")


def make_agent(local, remote, *, per_type=None, ledger=None, answer_source="local"):
    defaults = {"n_samples": 3, "escalation_threshold": 0.6}
    return RoutingAgent(
        local,
        remote,
        PolicyBook(defaults, per_type),
        default_remote_model="test-model",
        answer_source=answer_source,
        ledger=ledger,
    )


def test_confident_local_answer_costs_nothing():
    local = MockLocalBackend(["Answer: 19"])
    remote = MockRemoteBackend()
    result = make_agent(local, remote).solve(MATH)
    assert result.answer == "19"
    assert result.source == "local"
    assert result.remote_prompt_tokens == 0
    assert result.remote_completion_tokens == 0
    assert remote.calls == []


def test_disagreement_escalates():
    local = MockLocalBackend([["Answer: 19", "Answer: 3", "Answer: 7"]])
    remote = MockRemoteBackend(["19"])
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "remote"
    assert result.answer == "19"
    assert result.remote_prompt_tokens > 0
    assert len(remote.calls) == 1


def test_local_crash_escalates():
    local = MockLocalBackend(fail=True)
    remote = MockRemoteBackend(["Answer: 42"])
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "remote"
    assert result.answer == "42"


def test_total_failure_returns_fallback_not_crash():
    local = MockLocalBackend(fail=True)
    remote = MockRemoteBackend(fail=True)
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "fallback"
    assert result.answer == ""


def test_remote_failure_falls_back_to_local_candidate():
    local = MockLocalBackend([["Answer: 19", "Answer: 3", "Answer: 7"]])
    remote = MockRemoteBackend(fail=True)
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "fallback"
    assert result.answer == "19"


def test_always_remote_policy_skips_local():
    local = MockLocalBackend(["Answer: 19"])
    remote = MockRemoteBackend(["19"])
    agent = make_agent(local, remote, per_type={"math": {"always_remote": True}})
    result = agent.solve(MATH)
    assert result.source == "remote"
    assert local.calls == []


def test_invalid_remote_format_is_reformatted_locally():
    local = MockLocalBackend([["Answer: 1", "Answer: 2", "Answer: 3"], "Answer: 42"])
    remote = MockRemoteBackend(["roughly forty-two"])
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "remote"
    assert result.answer == "42"


def test_long_context_is_compressed_before_remote():
    context = "This filler sentence pads the passage. " * 500
    task = Task(id="e1", input="Who is the CEO?", context=context, type="factual")
    local = MockLocalBackend(
        [
            ["Answer: alice", "Answer: bob", "Answer: carol"],  # disagreeing solve samples
            "The CEO is Alice.",  # compression excerpt
        ]
    )
    remote = MockRemoteBackend(["Alice"])
    result = make_agent(local, remote).solve(task)
    assert result.source == "remote"
    assert result.answer == "Alice"
    assert "compressed_context" in result.decision_path
    assert "filler sentence" not in remote.calls[0]["user"]


def test_no_local_backend_goes_remote():
    remote = MockRemoteBackend(["19"])
    result = make_agent(None, remote).solve(MATH)
    assert result.source == "remote"
    assert result.answer == "19"


def test_ledger_records_and_totals():
    ledger = Ledger()
    local = MockLocalBackend(["Answer: 19"])
    agent = make_agent(local, MockRemoteBackend(), ledger=ledger)
    agent.solve(MATH)
    summary = ledger.summary()
    assert summary["tasks"] == 1
    assert summary["local_answers"] == 1
    assert summary["remote_prompt_tokens"] == 0


def test_greedy_mode_forces_single_sample():
    local = MockLocalBackend(["Answer: 19"])
    agent = make_agent(local, MockRemoteBackend(), per_type={"math": {"n_samples": 5}})
    result = agent.solve(MATH, mode="greedy")
    assert result.source == "local"
    assert local.calls[0]["n"] == 1


def test_remote_direct_mode_skips_local():
    local = MockLocalBackend(["Answer: 19"])
    remote = MockRemoteBackend(["19"])
    result = make_agent(local, remote).solve(MATH, mode="remote_direct")
    assert result.source == "remote"
    assert local.calls == []


def test_fireworks_mode_confirms_confident_draft_remotely():
    local = MockLocalBackend(["Answer: 19"])
    remote = MockRemoteBackend(["19"])
    agent = make_agent(
        local, remote, per_type={"math": {"use_draft": True}}, answer_source="fireworks"
    )
    result = agent.solve(MATH)
    assert result.source == "remote"  # scored answer originates from Fireworks
    assert result.answer == "19"
    assert result.remote_prompt_tokens > 0
    assert "Draft answer: 19" in remote.calls[0]["user"]
    assert "draft_confirm" in result.decision_path


def test_fireworks_mode_low_confidence_solves_remotely_without_draft():
    local = MockLocalBackend([["Answer: 19", "Answer: 3", "Answer: 7"]])
    remote = MockRemoteBackend(["19"])
    agent = make_agent(local, remote, answer_source="fireworks")
    result = agent.solve(MATH)
    assert result.source == "remote"
    assert "Draft answer" not in remote.calls[0]["user"]


def test_fireworks_mode_remote_failure_still_falls_back_to_local():
    local = MockLocalBackend(["Answer: 19"])
    remote = MockRemoteBackend(fail=True)
    result = make_agent(local, remote, answer_source="fireworks").solve(MATH)
    assert result.source == "fallback"
    assert result.answer == "19"


def test_remote_direct_with_dead_remote_makes_late_local_attempt():
    local = MockLocalBackend(["Answer: 19"])
    remote = MockRemoteBackend(fail=True)
    result = make_agent(local, remote).solve(MATH, mode="remote_direct")
    assert result.source == "fallback"
    assert result.answer == "19"
    assert "late_local_attempt" in result.decision_path


def test_empty_remote_answer_falls_back_to_local_candidate():
    local = MockLocalBackend([["Answer: 19", "Answer: 3", "Answer: 7"]])
    remote = MockRemoteBackend(["   "])
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "fallback"
    assert result.answer == "19"
    assert "remote_empty" in result.decision_path


def test_truncated_remote_response_is_flagged():
    local = MockLocalBackend([["Answer: 1", "Answer: 2", "Answer: 3"], "Answer: 42"])
    remote = MockRemoteBackend(["Answer: 4"], finish_reason="length")
    result = make_agent(local, remote).solve(MATH)
    assert "remote_truncated" in result.decision_path


def test_pick_model_resolves_against_allowed_list():
    from frugal_router.agent import pick_model

    allowed = [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/gemma-4-31b-it",
    ]
    assert pick_model(allowed, ["gemma"], "fallback") == allowed[1]
    assert pick_model(allowed, ["kimi", "minimax"], "fallback") == allowed[0]
    assert pick_model(allowed, ["nope"], "fallback") == allowed[0]
    assert pick_model([], ["gemma"], "fallback") == "fallback"


def test_adaptive_sampling_stops_early_on_unanimity():
    local = MockLocalBackend([["Answer: 19", "Answer: 19", "Answer: 19"]])
    agent = make_agent(local, MockRemoteBackend(), per_type={"math": {"n_samples": 5}})
    result = agent.solve(MATH)
    assert result.source == "local"
    assert len(local.calls) == 1  # unanimous first window, no extension
    assert local.calls[0]["n"] == 3


def test_adaptive_sampling_extends_on_disagreement():
    local = MockLocalBackend(
        [["Answer: 19", "Answer: 3", "Answer: 19"], ["Answer: 19", "Answer: 19"]]
    )
    agent = make_agent(local, MockRemoteBackend(), per_type={"math": {"n_samples": 5}})
    result = agent.solve(MATH)
    assert result.source == "local"  # 4 of 5 agree after extension
    assert len(local.calls) == 2
    assert local.calls[1]["n"] == 2
    assert result.confidence.n_samples == 5
    assert result.confidence.agreement == 4 / 5
