import json

from frugal_router.harness import run_batch


class StubAgent:
    """Records solve() calls; answers with the task id so ordering is visible."""

    def __init__(self, fail_ids=()):
        self.calls = []
        self.fail_ids = set(fail_ids)

    def solve(self, task, mode="full"):
        self.calls.append({"id": task.id, "mode": mode})
        if task.id in self.fail_ids:
            raise RuntimeError("boom")
        from frugal_router.agent import SolveResult

        return SolveResult(task_id=task.id, answer=f"answer-{task.id}", source="local", task_type="factual")


def write_tasks(tmp_path, tasks):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks))
    return str(path)


def read_results(path):
    return json.loads(open(path, encoding="utf-8").read())


def test_batch_answers_every_task_in_input_order(tmp_path):
    tasks = [
        {"task_id": "a", "prompt": "Write a Python function that sorts a list."},
        {"task_id": "b", "prompt": "What is the sentiment of this review: 'great'?"},
        {"task_id": "c", "prompt": "What is the capital of France?"},
    ]
    out = str(tmp_path / "results.json")
    agent = StubAgent()
    code = run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=600)

    assert code == 0
    results = read_results(out)
    assert [r["task_id"] for r in results] == ["a", "b", "c"]  # input order preserved
    assert all(r["answer"].startswith("answer-") for r in results)
    # Cheap categories were processed first: sentiment (b), factual (c), code (a).
    assert [c["id"] for c in agent.calls] == ["b", "c", "a"]


def test_failing_task_never_takes_down_the_batch(tmp_path):
    tasks = [{"task_id": "a", "prompt": "q1"}, {"task_id": "b", "prompt": "q2"}]
    out = str(tmp_path / "results.json")
    code = run_batch(write_tasks(tmp_path, tasks), out, agent=StubAgent(fail_ids={"a"}), time_budget_s=600)

    assert code == 0
    results = {r["task_id"]: r["answer"] for r in read_results(out)}
    assert results["a"] == ""  # failed but present
    assert results["b"] == "answer-b"


def test_unreadable_input_still_writes_valid_output(tmp_path):
    bad = tmp_path / "tasks.json"
    bad.write_text("{not json")
    out = str(tmp_path / "results.json")
    assert run_batch(str(bad), out, agent=StubAgent()) == 0
    assert read_results(out) == []


def test_missing_prompt_field_yields_empty_answer_entry(tmp_path):
    tasks = [{"task_id": "a"}, "not-a-dict", {"prompt": "no id"}]
    out = str(tmp_path / "results.json")
    assert run_batch(write_tasks(tmp_path, tasks), out, agent=StubAgent(), time_budget_s=600) == 0
    results = read_results(out)
    ids = [r["task_id"] for r in results]
    assert "a" in ids
    assert "task-2" in ids  # the dict without a task_id got a synthetic one


def test_scheduler_degrades_to_remote_direct_when_out_of_time(tmp_path):
    tasks = [{"task_id": str(i), "prompt": f"question {i}"} for i in range(5)]
    out = str(tmp_path / "results.json")
    agent = StubAgent()
    run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=0.0)
    assert {c["mode"] for c in agent.calls} == {"remote_direct"}


def test_scheduler_uses_full_mode_with_generous_budget(tmp_path):
    tasks = [{"task_id": "a", "prompt": "What is the capital of France?"}]
    out = str(tmp_path / "results.json")
    agent = StubAgent()
    run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=600)
    assert agent.calls[0]["mode"] == "full"
