"""Guard the 19-task real-image rehearsal's intended lane coverage."""
import json
from pathlib import Path

from frugal_router.classify import classify
from frugal_router.solvers import solve_any
from frugal_router.tasks import Task


_ROOT = Path(__file__).resolve().parents[1]


def test_rehearsal_fixture_is_complete_and_targets_execution_lanes():
    tasks = json.loads((_ROOT / "data" / "rehearsal_local_19.json").read_text())
    assert len(tasks) == 19
    ids = [task["task_id"] for task in tasks]
    assert len(ids) == len(set(ids))

    by_id = {task["task_id"]: task["prompt"] for task in tasks}
    assert classify(Task(id="math", input=by_id["math-pot-1"])) == "math"
    assert solve_any(by_id["math-pot-1"]) is None  # reaches PoT rather than a solver
    assert classify(Task(id="code-gen", input=by_id["code-gen-1"])) == "code_gen"
    assert classify(Task(id="code-debug", input=by_id["code-debug-1"])) == "code_debug"
