from frugal_router.agent import RoutingAgent
from frugal_router.backends.mock import MockLocalBackend, MockRemoteBackend
from frugal_router.ledger import Ledger
from frugal_router.policy import PolicyBook
from frugal_router.tasks import Task

MATH = Task(id="m1", input="What is 12 + 7?", type="math")


def make_agent(local, remote, *, per_type=None, ledger=None):
    defaults = {"n_samples": 3, "escalation_threshold": 0.6}
    return RoutingAgent(
        local,
        remote,
        PolicyBook(defaults, per_type),
        default_remote_model="test-model",
        ledger=ledger,
    )


def test_confident_local_answer_costs_nothing():
    local = MockLocalBackend(["Answer: 19"], yes_prob=0.95)
    remote = MockRemoteBackend()
    result = make_agent(local, remote).solve(MATH)
    assert result.answer == "19"
    assert result.source == "local"
    assert result.remote_prompt_tokens == 0
    assert result.remote_completion_tokens == 0
    assert remote.calls == []


def test_disagreement_escalates():
    local = MockLocalBackend([["Answer: 19", "Answer: 3", "Answer: 7"]], yes_prob=0.2)
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
    local = MockLocalBackend([["Answer: 19", "Answer: 3", "Answer: 19"]], yes_prob=0.0)
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
    local = MockLocalBackend(
        [["Answer: 1", "Answer: 2", "Answer: 3"], "Answer: 42"], yes_prob=0.0
    )
    remote = MockRemoteBackend(["roughly forty-two"])
    result = make_agent(local, remote).solve(MATH)
    assert result.source == "remote"
    assert result.answer == "42"


def test_long_context_is_compressed_before_remote():
    context = "This filler sentence pads the passage. " * 500
    task = Task(id="e1", input="Who is the CEO?", context=context, type="extraction")
    local = MockLocalBackend(
        [
            ["Answer: alice", "Answer: bob", "Answer: carol"],  # disagreeing solve samples
            "The CEO is Alice.",  # compression excerpt
        ],
        yes_prob=0.0,
    )
    remote = MockRemoteBackend(["Alice"])
    result = make_agent(local, remote).solve(task)
    assert result.source == "remote"
    assert result.answer == "alice"
    assert "compressed_context" in result.decision_path
    assert "filler sentence" not in remote.calls[0]["user"]


def test_no_local_backend_goes_remote():
    remote = MockRemoteBackend(["19"])
    result = make_agent(None, remote).solve(MATH)
    assert result.source == "remote"
    assert result.answer == "19"


def test_ledger_records_and_totals():
    ledger = Ledger()
    local = MockLocalBackend(["Answer: 19"], yes_prob=0.95)
    agent = make_agent(local, MockRemoteBackend(), ledger=ledger)
    agent.solve(MATH)
    summary = ledger.summary()
    assert summary["tasks"] == 1
    assert summary["local_answers"] == 1
    assert summary["remote_prompt_tokens"] == 0
