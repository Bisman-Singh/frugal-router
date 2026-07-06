"""Offline threshold sweep over collected eval records.

Signals are collected once (eval --collect-remote), then decisions are replayed
here without touching a model, so a full grid costs zero tokens and seconds of
wall clock. The recommendation is the cheapest point at or above the accuracy
target plus a safety margin.
"""
from __future__ import annotations

import csv
from pathlib import Path


def replay(records: list[dict], threshold: float, p_fail_cutoff: float) -> dict:
    correct = 0
    graded = 0
    tokens = 0
    for r in records:
        p_fail = r.get("conf_p_fail")
        escalate = (
            not r.get("conf_format_valid", False)
            or (p_fail is not None and p_fail > p_fail_cutoff)
            or (r.get("conf_score") or 0.0) < threshold
        )
        if escalate:
            ok = r.get("remote_correct")
            tokens += (r.get("remote_probe_prompt_tokens") or 0) + (
                r.get("remote_probe_completion_tokens") or 0
            )
        else:
            ok = r.get("local_correct")
        if ok is not None:
            graded += 1
            correct += bool(ok)
    return {
        "threshold": threshold,
        "p_fail_cutoff": p_fail_cutoff,
        "accuracy": round(correct / graded, 4) if graded else 0.0,
        "remote_tokens": tokens,
    }


def sweep(
    records: list[dict],
    thresholds: list[float],
    p_fail_cutoffs: list[float],
    target_accuracy: float,
    margin: float = 0.02,
) -> tuple[list[dict], dict]:
    rows = [replay(records, t, p) for t in thresholds for p in p_fail_cutoffs]
    rows.sort(key=lambda r: (r["remote_tokens"], -r["accuracy"]))
    feasible = [r for r in rows if r["accuracy"] >= target_accuracy + margin]
    recommendation = feasible[0] if feasible else max(rows, key=lambda r: r["accuracy"])
    return rows, recommendation


def pareto(rows: list[dict]) -> list[dict]:
    frontier = []
    top_accuracy = -1.0
    for row in sorted(rows, key=lambda r: (r["remote_tokens"], -r["accuracy"])):
        if row["accuracy"] > top_accuracy:
            frontier.append(row)
            top_accuracy = row["accuracy"]
    return frontier


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
