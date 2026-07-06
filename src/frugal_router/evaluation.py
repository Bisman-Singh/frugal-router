"""Eval harness: run the agent over a labeled dataset, grade, slice, and log.

Records include the local candidate and its correctness even when the task was
escalated, because that is exactly the training data the failure predictor and
the offline sweep need.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .extract import normalize_number, text_key
from .tasks import Task


def grade(task: Task, answer: str | None, task_type: str) -> bool | None:
    """True/False when gradable, None when the task has no expected answer."""
    if task.expected is None:
        return None
    if answer is None or answer == "":
        return False
    grader = task.grader or _default_grader(task, task_type)
    if grader == "contains_all":
        expected = task.expected if isinstance(task.expected, list) else [task.expected]
        hay = str(answer).casefold()
        return all(str(kw).casefold() in hay for kw in expected)
    if grader == "numeric":
        expected_num = normalize_number(str(task.expected))
        answer_num = normalize_number(str(answer))
        try:
            return (
                expected_num is not None
                and answer_num is not None
                and abs(float(expected_num) - float(answer_num)) < 1e-6
            )
        except ValueError:
            return False
    expected_key = text_key(str(task.expected))
    return expected_key is not None and expected_key == text_key(str(answer))


def _default_grader(task: Task, task_type: str) -> str:
    if isinstance(task.expected, list):
        return "contains_all"
    if task_type == "math":
        return "numeric"
    if task_type in ("factual", "logic"):
        return "exact"
    return "contains_all"


def run_eval(agent, tasks: list[Task], *, out_dir: str | None = None,
             collect_remote: bool = False) -> dict:
    records = []
    for task in tasks:
        result = agent.solve(task)
        record = {
            "task_id": task.id,
            "type": result.task_type,
            "input": task.input,
            "expected": task.expected,
            "answer": result.answer,
            "correct": grade(task, result.answer, result.task_type),
            "source": result.source,
            "local_answer": result.local_answer,
            "local_correct": grade(task, result.local_answer, result.task_type),
            "remote_prompt_tokens": result.remote_prompt_tokens,
            "remote_completion_tokens": result.remote_completion_tokens,
            "decision_path": result.decision_path,
        }
        if result.confidence:
            conf = result.confidence.to_dict()
            conf.pop("candidate", None)
            record.update({f"conf_{k}": v for k, v in conf.items()})
        if collect_remote:
            try:
                r_answer, r_pt, r_ct = agent.remote_answer(task, result.task_type)
                record["remote_answer"] = r_answer
                record["remote_correct"] = grade(task, r_answer, result.task_type)
                record["remote_probe_prompt_tokens"] = r_pt
                record["remote_probe_completion_tokens"] = r_ct
            except Exception as exc:
                record["remote_answer"] = None
                record["remote_correct"] = None
                record["remote_probe_error"] = type(exc).__name__
        records.append(record)

    summary = summarize(records)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "records.jsonl", "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(out / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
    return {"records": records, "summary": summary}


def summarize(records: list[dict]) -> dict:
    graded = [r for r in records if r["correct"] is not None]
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in graded:
        by_type[r["type"]].append(bool(r["correct"]))
    escalated = [r for r in records if r["source"] in ("remote", "cache")]
    return {
        "tasks": len(records),
        "graded": len(graded),
        "accuracy": round(sum(r["correct"] for r in graded) / len(graded), 4) if graded else None,
        "accuracy_by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
        "local_answer_rate": round(
            sum(r["source"] == "local" for r in records) / len(records), 4
        ) if records else None,
        "escalation_rate": round(len(escalated) / len(records), 4) if records else None,
        "remote_prompt_tokens": sum(r["remote_prompt_tokens"] for r in records),
        "remote_completion_tokens": sum(r["remote_completion_tokens"] for r in records),
        "remote_total_tokens": sum(
            r["remote_prompt_tokens"] + r["remote_completion_tokens"] for r in records
        ),
    }


def load_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
