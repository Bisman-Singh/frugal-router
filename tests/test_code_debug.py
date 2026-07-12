"""Execution-grounded code_debug: the cause is observed, the fix is verified."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frugal_router import code_debug  # noqa: E402

_WRONG_MAX = (
    "The function `get_max` should return the largest number in a list. "
    "For example, get_max([3, 1, 5]) returns 5. Find and fix the bug.\n"
    "```python\n"
    "def get_max(nums):\n"
    "    return nums[0]\n"
    "```"
)


def test_observe_failure_wrong_return():
    code = "def get_max(nums):\n    return nums[0]"
    out = code_debug.observe_failure(code, [("get_max([3, 1, 5])", "5")])
    assert out == "get_max([3, 1, 5]) returns 3 but should return 5"


def test_observe_failure_exception():
    code = "def f(x):\n    return x[10]"
    out = code_debug.observe_failure(code, [("f([1, 2, 3])", "3")])
    assert out == "calling f([1, 2, 3]) raises IndexError"


def test_observe_failure_syntax_error():
    code = "def f(x)\n    return x"           # missing colon
    out = code_debug.observe_failure(code, [("f(2)", "2")])
    assert out is not None and "fails to parse" in out


def test_observe_none_when_code_is_correct():
    code = "def f(x):\n    return x * 2"
    assert code_debug.observe_failure(code, [("f(3)", "6")]) is None


def test_verify_grounds_prompt_and_accepts_passing_fix():
    captured = {}

    def gen(system, ask, cap):
        captured["ask"] = ask
        return "```python\ndef get_max(nums):\n    return max(nums)\n```"

    out = code_debug.verify_code_debug(_WRONG_MAX, gen)
    # the OBSERVED failure was fed to the model, not a guessed cause
    assert "returns 3 but should return 5" in captured["ask"]
    assert out is not None
    assert out.startswith("Bug:") and "```python" in out and "max(nums)" in out


def test_verify_rejects_fix_that_still_fails():
    def gen(system, ask, cap):
        return "```python\ndef get_max(nums):\n    return sorted(nums)[0]\n```"  # still wrong

    assert code_debug.verify_code_debug(_WRONG_MAX, gen) is None


def test_verify_defers_without_worked_example():
    prompt = ("Fix the bug in this function.\n"
              "```python\ndef get_max(nums):\n    return nums[0]\n```")
    called = {"n": 0}

    def gen(system, ask, cap):
        called["n"] += 1
        return "```python\ndef get_max(nums):\n    return max(nums)\n```"

    assert code_debug.verify_code_debug(prompt, gen) is None
    assert called["n"] == 0            # no evidence -> never even asks the model


def test_verify_defers_without_snippet():
    prompt = "The function get_max should return the largest number. get_max([1,2]) returns 2."
    assert code_debug.verify_code_debug(prompt, lambda s, a, c: "") is None


def test_try_code_debug_wiring(monkeypatch):
    """simple._try_local routes code_debug through the execution-grounded lane."""
    from frugal_router import local_tier, simple

    def fake_generate(system, prompt, cap, temperature=0.0):
        return "```python\ndef get_max(nums):\n    return max(nums)\n```"

    monkeypatch.setattr(local_tier, "available", lambda: True)
    monkeypatch.setattr(local_tier, "generate", fake_generate)
    simple._LOCAL_SPENT["s"] = 0.0
    out = simple._try_local("t1", "code_debug", _WRONG_MAX, time.monotonic() + 600)
    assert out is not None and "max(nums)" in out and out.startswith("Bug:")
