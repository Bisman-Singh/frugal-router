"""Judging-harness entrypoint: /input/tasks.json in, /output/results.json out.

Contract from the official guide: the container reads [{"task_id", "prompt"}]
on startup, writes [{"task_id", "answer"}] before exiting, exit code 0, whole
batch within 10 minutes. A missing or invalid output file scores zero, so this
module never crashes and never writes partial output: every task_id gets an
answer entry no matter what fails in between.

The binding constraint is the wall clock, not local tokens. The scheduler
banks cheap categories first and degrades the strategy (voting, then single
greedy local, then one direct remote call) as the remaining budget shrinks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .classify import classify
from .config import SchedulerConfig, build_agent, load_settings
from .ledger import Ledger
from .tasks import Task

# Cheap, local-safe categories first; slow reasoning categories last so they
# are the ones degraded if the clock runs down.
CATEGORY_ORDER = [
    "sentiment",
    "factual",
    "ner",
    "summarization",
    "math",
    "logic",
    "code_debug",
    "code_gen",
]


def run_batch(
    input_path: str = "/input/tasks.json",
    output_path: str = "/output/results.json",
    *,
    config_path: str = "configs/default.yaml",
    agent=None,
    time_budget_s: float | None = None,
) -> int:
    started = time.monotonic()
    answers: dict[str, str] = {}

    tasks = _read_tasks(input_path, answers)
    if not tasks:
        _write_results(output_path, answers)
        return 0

    scheduler = SchedulerConfig()
    ledger = Ledger()
    try:
        settings = load_settings(config_path)
        scheduler = settings.scheduler
        if agent is None:
            agent = build_agent(settings, ledger=ledger)
    except Exception as exc:
        print(f"setup degraded: {type(exc).__name__}: {exc}", file=sys.stderr)

    budget = time_budget_s if time_budget_s is not None else scheduler.time_budget_s
    deadline = started + budget

    if agent is not None:
        ordered = sorted(tasks, key=lambda t: _category_rank(classify(t)))
        for index, task in enumerate(ordered):
            mode = _mode(deadline - time.monotonic(), len(ordered) - index, scheduler)
            try:
                result = agent.solve(task, mode=mode)
                answers[task.id] = result.answer
            except Exception as exc:
                # One bad task must never take down the batch.
                print(f"task {task.id} failed: {type(exc).__name__}", file=sys.stderr)

    _write_results(output_path, answers)
    summary = ledger.summary()
    summary["elapsed_s"] = round(time.monotonic() - started, 1)
    print(json.dumps(summary), file=sys.stderr)
    return 0


def _read_tasks(input_path: str, answers: dict[str, str]) -> list[Task]:
    """Seed an answer slot for every task up front; unreadable input still
    produces a valid (empty) results file."""
    try:
        raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read {input_path}: {type(exc).__name__}", file=sys.stderr)
        return []
    tasks = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", f"task-{index}"))
        answers[task_id] = ""
        tasks.append(Task(id=task_id, input=str(item.get("prompt", ""))))
    return tasks


def _write_results(output_path: str, answers: dict[str, str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = [{"task_id": task_id, "answer": answer} for task_id, answer in answers.items()]
    path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


def _category_rank(category: str) -> int:
    try:
        return CATEGORY_ORDER.index(category)
    except ValueError:
        return len(CATEGORY_ORDER)


def _mode(remaining_s: float, tasks_left: int, scheduler: SchedulerConfig) -> str:
    per_task = remaining_s / max(1, tasks_left)
    if per_task >= scheduler.est_full_s:
        return "full"
    if per_task >= scheduler.est_greedy_s:
        return "greedy"
    return "remote_direct"
