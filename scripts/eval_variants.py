#!/usr/bin/env python3
"""Measure the zero-token DETERMINISTIC tiers on the archetype variants.

Runs `solvers.solve_any` (bare arithmetic, the percent/discount/interest/rate/
geometry families, transitive ordering, and the assignment-CSP puzzle solver)
over a variant file and reports a per-category coverage/accuracy table. These
tiers need no model and no network, so this eval runs anywhere and isolates
exactly what the deterministic layer contributes to the accuracy gate.

Model-backed lanes (sentiment, NER, summarization, factual, and the model half
of code/math) have no deterministic path, so they show 0 coverage here by
design; measure those with the baked model via the image smoke.

Usage:
    python scripts/gen_variants.py --per 12 --out data/variants.jsonl
    python scripts/eval_variants.py --tasks data/variants.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frugal_router.solvers import solve_any  # noqa: E402

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _grade(grader: dict, answer: str) -> bool:
    kind = grader.get("type")
    if kind == "numeric":
        nums = _NUM.findall(answer.replace(",", ""))
        if not nums:
            return False
        try:
            return abs(float(nums[-1]) - float(grader["expected"])) < 1e-4
        except ValueError:
            return False
    if kind == "string":
        exp = str(grader["expected"]).strip().lower()
        return exp == answer.strip().lower() or bool(
            re.search(rf"\b{re.escape(exp)}\b", answer.lower()))
    return False  # non-deterministic grader: not scorable here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/variants.jsonl")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    det_categories = {"math", "logic"}  # the only categories a solver can carry

    stat = defaultdict(lambda: {"total": 0, "answered": 0, "correct": 0})
    misses = []
    for t in tasks:
        cat = t["category"]
        s = stat[cat]
        s["total"] += 1
        try:
            hit = solve_any(t["prompt"])
        except Exception:
            hit = None
        if hit is None:
            continue
        s["answered"] += 1
        ok = _grade(t.get("grader", {}), hit[0])
        s["correct"] += int(ok)
        if not ok:
            misses.append((t["id"], hit[0], t.get("grader", {}).get("expected")))

    print(f"{'category':<16}{'total':>7}{'answered':>10}{'correct':>9}"
          f"{'cover%':>9}{'acc%':>8}")
    print("-" * 59)
    det_total = det_answered = det_correct = 0
    for cat in sorted(stat):
        s = stat[cat]
        cov = 100.0 * s["answered"] / s["total"] if s["total"] else 0.0
        acc = 100.0 * s["correct"] / s["answered"] if s["answered"] else 0.0
        tag = "" if cat in det_categories else "  (model tier)"
        print(f"{cat:<16}{s['total']:>7}{s['answered']:>10}{s['correct']:>9}"
              f"{cov:>8.0f}%{acc:>7.0f}%{tag}")
        if cat in det_categories:
            det_total += s["total"]
            det_answered += s["answered"]
            det_correct += s["correct"]
    print("-" * 59)
    cov = 100.0 * det_answered / det_total if det_total else 0.0
    acc = 100.0 * det_correct / det_answered if det_answered else 0.0
    print(f"DETERMINISTIC (math+logic): {det_answered}/{det_total} answered "
          f"({cov:.0f}% coverage), {det_correct}/{det_answered} correct "
          f"({acc:.0f}% accuracy of answered)")
    if misses:
        print("\nsolver misses (id, got, expected) — must be empty; a wrong "
              "zero-token answer costs the gate:")
        for tid, got, exp in misses:
            print(f"  {tid}: {got!r} != {exp!r}")


if __name__ == "__main__":
    main()
