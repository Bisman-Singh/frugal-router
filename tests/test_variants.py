"""Archetype-variant generator: determinism, shape, and solver safety.

The safety test is the important one: across a freshly randomized set, every
deterministic solver hit on a math/logic task must match the independently
computed label. A single wrong zero-token answer would cost the accuracy gate.
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import gen_variants  # noqa: E402
from frugal_router.solvers import solve_any  # noqa: E402

CATEGORIES = {"math", "logic", "code_gen", "code_debug", "sentiment", "ner",
              "summarization", "factual"}


def test_generator_deterministic_under_seed():
    assert gen_variants.generate(12, 7) == gen_variants.generate(12, 7)


def test_generator_reseeds():
    a = [t["prompt"] for t in gen_variants.generate(12, 7)]
    b = [t["prompt"] for t in gen_variants.generate(12, 99)]
    assert a != b


def test_generator_shape():
    tasks = gen_variants.generate(12, 7)
    assert len(tasks) >= 80
    assert {t["category"] for t in tasks} == CATEGORIES
    for t in tasks:
        assert t["id"] and t["prompt"] and "grader" in t
    assert all(t["grader"]["type"] == "numeric"
               for t in tasks if t["category"] == "math")
    assert all(t["grader"]["type"] == "string"
               for t in tasks if t["category"] == "logic")


def _num(s):
    import re
    m = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m[-1]) if m else None


def test_solver_hits_are_correct_across_randomized_set():
    # Multiple seeds so the assertion sees many randomizations, not one.
    for seed in (1, 7, 42, 99):
        for t in gen_variants.generate(12, seed):
            if t["category"] not in ("math", "logic"):
                continue
            hit = solve_any(t["prompt"])
            if hit is None:
                continue  # deferral is always allowed
            exp = t["grader"]["expected"]
            if t["category"] == "math":
                assert _num(hit[0]) is not None
                assert abs(_num(hit[0]) - float(exp)) < 1e-4, (t["prompt"], hit)
            else:
                assert str(exp).lower() in hit[0].lower(), (t["prompt"], hit)


def test_math_and_logic_fully_covered_at_default_seed():
    # Regression guard: the solvers should carry every generated math/logic
    # archetype at the default seed (they are the shapes the solvers target).
    tasks = gen_variants.generate(12, 7)
    for cat in ("math", "logic"):
        cat_tasks = [t for t in tasks if t["category"] == cat]
        answered = [t for t in cat_tasks if solve_any(t["prompt"]) is not None]
        assert len(answered) == len(cat_tasks), f"{cat}: {len(answered)}/{len(cat_tasks)}"
