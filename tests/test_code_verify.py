"""Execution-verified code_gen: only test-passing local code is kept."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frugal_router import code_verify  # noqa: E402


def test_extract_tests_parses_common_forms():
    prompt = ("Write a function `add`. For example, add(2, 3) should return 5. "
              "Also add(0, 0) == 0.")
    tests = code_verify.extract_tests(prompt, "add")
    assert ("add(2, 3)", "5") in tests
    assert ("add(0, 0)", "0") in tests


def test_run_accepts_correct_code():
    assert code_verify._run("def add(a, b):\n    return a + b",
                            [("add(2, 3)", "5")]) is True


def test_run_rejects_wrong_code():
    assert code_verify._run("def add(a, b):\n    return a - b",
                            [("add(2, 3)", "5")]) is False


def test_verify_code_gen_accepts_passing_generation():
    prompt = "Write a function `double` where double(4) returns 8."

    def gen(system, p, cap):
        return "```python\ndef double(x):\n    return x * 2\n```"

    out = code_verify.verify_code_gen(prompt, gen)
    assert out is not None and "def double" in out


def test_verify_code_gen_rejects_failing_generation():
    prompt = "Write a function `double` where double(4) returns 8."

    def gen(system, p, cap):
        return "```python\ndef double(x):\n    return x + 2\n```"  # wrong

    assert code_verify.verify_code_gen(prompt, gen) is None


def test_verify_code_gen_defers_without_tests():
    prompt = "Write a function that reverses a linked list."  # no worked example

    def gen(system, p, cap):
        return "```python\ndef reverse(x):\n    return x\n```"

    assert code_verify.verify_code_gen(prompt, gen) is None
