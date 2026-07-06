from frugal_router.agent import RoutingAgent
from frugal_router.backends.mock import MockLocalBackend, MockRemoteBackend
from frugal_router.evaluation import grade, load_records, run_eval
from frugal_router.policy import PolicyBook
from frugal_router.tasks import Task


def make_agent(local, remote):
    return RoutingAgent(
        local,
        remote,
        PolicyBook({"n_samples": 1, "escalation_threshold": 0.6}),
        default_remote_model="test-model",
    )


def test_grade_numeric_tolerates_formatting():
    task = Task(id="t", input="q", expected="1,234")
    assert grade(task, "1234", "math") is True
    assert grade(task, "1235", "math") is False


def test_grade_exact_normalized():
    task = Task(id="t", input="q", expected="Paris")
    assert grade(task, "paris.", "factual") is True
    assert grade(task, "London", "factual") is False


def test_grade_contains_all():
    task = Task(id="t", input="q", expected=["library", "mayor"])
    assert grade(task, "The mayor opened the new library downtown.", "summarization") is True
    assert grade(task, "The mayor gave a speech.", "summarization") is False


def test_grade_ungradable_returns_none():
    task = Task(id="t", input="q")
    assert grade(task, "anything", "factual") is None


def test_run_eval_metrics_and_records(tmp_path):
    tasks = [
        Task(id="1", input="What is 2 + 2?", type="math", expected="4"),
        Task(id="2", input="What is 3 + 3?", type="math", expected="6"),
    ]
    # Confident both times; the second local answer is wrong.
    local = MockLocalBackend(["Answer: 4", "Answer: 7"])
    agent = make_agent(local, MockRemoteBackend())
    report = run_eval(agent, tasks, out_dir=str(tmp_path))

    assert report["summary"]["accuracy"] == 0.5
    assert report["summary"]["remote_total_tokens"] == 0
    assert report["summary"]["local_answer_rate"] == 1.0
    records = load_records(tmp_path / "records.jsonl")
    assert len(records) == 2
    assert records[0]["local_correct"] is True
    assert records[1]["local_correct"] is False


def test_collect_remote_adds_counterfactuals(tmp_path):
    tasks = [Task(id="1", input="What is 2 + 2?", type="math", expected="4")]
    local = MockLocalBackend(["Answer: 4"])
    remote = MockRemoteBackend(["4"])
    agent = make_agent(local, remote)
    report = run_eval(agent, tasks, collect_remote=True)

    record = report["records"][0]
    assert record["source"] == "local"
    assert record["remote_answer"] == "4"
    assert record["remote_correct"] is True
    assert record["remote_probe_prompt_tokens"] > 0
    # The probe billed nothing against the run itself.
    assert report["summary"]["remote_total_tokens"] == 0
