import json

from frugal_router.harness import run_batch


class StubAgent:
    """Records solve() calls; answers with the task id so ordering is visible.

    Has a truthy `local` so the harness runs it sequentially (deterministic
    call order for assertions). ParallelStub flips that off.
    """

    def __init__(self, fail_ids=()):
        self.calls = []
        self.fail_ids = set(fail_ids)
        self.local = "stub-local"

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


def test_scheduler_degrades_when_short_on_time(tmp_path):
    tasks = [{"task_id": str(i), "prompt": f"question {i}"} for i in range(5)]
    out = str(tmp_path / "results.json")
    agent = StubAgent()
    run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=18.0)
    modes = [c["mode"] for c in agent.calls]
    assert modes[0] == "remote_direct"  # 30s (minus 15s margin) across 5 tasks cannot afford local up front
    # As instant (stub) tasks free up budget the scheduler recovers toward local;
    # with real slow local tasks the budget would keep shrinking. Both are intended.


def test_hard_stop_flushes_valid_output_without_solving(tmp_path):
    tasks = [{"task_id": str(i), "prompt": f"question {i}"} for i in range(3)]
    out = str(tmp_path / "results.json")
    agent = StubAgent()
    assert run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=0.0) == 0
    assert agent.calls == []  # past the hard stop before the first task
    results = read_results(out)
    assert len(results) == 3
    assert all(r["answer"] == "" for r in results)


def test_scheduler_uses_full_mode_with_generous_budget(tmp_path):
    tasks = [{"task_id": "a", "prompt": "What is the capital of France?"}]
    out = str(tmp_path / "results.json")
    agent = StubAgent()
    run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=600)
    assert agent.calls[0]["mode"] == "full"


class ParallelStub(StubAgent):
    def __init__(self, fail_ids=()):
        super().__init__(fail_ids)
        self.local = None  # no local model -> harness parallelizes


def test_parallel_batch_answers_every_task(tmp_path):
    tasks = [{"task_id": str(i), "prompt": f"question {i}"} for i in range(8)]
    out = str(tmp_path / "results.json")
    agent = ParallelStub()
    assert run_batch(write_tasks(tmp_path, tasks), out, agent=agent, time_budget_s=600) == 0
    results = {r["task_id"]: r["answer"] for r in read_results(out)}
    assert len(results) == 8
    assert all(v.startswith("answer-") for v in results.values())
    assert len(agent.calls) == 8


def test_parallel_failure_is_isolated(tmp_path):
    tasks = [{"task_id": "a", "prompt": "q"}, {"task_id": "b", "prompt": "q"}]
    out = str(tmp_path / "results.json")
    run_batch(write_tasks(tmp_path, tasks), out, agent=ParallelStub(fail_ids={"a"}), time_budget_s=600)
    results = {r["task_id"]: r["answer"] for r in read_results(out)}
    assert results["a"] == ""
    assert results["b"] == "answer-b"
