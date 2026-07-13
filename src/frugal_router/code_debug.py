"""Execution-grounded local code debugging.

The shipped bug description is OBSERVED, not guessed. We run the buggy snippet
against the worked examples extracted from the prompt, capture the real failure
(a raised exception, a wrong return value, or a parse error), prompt the model
WITH that evidence, and keep the corrected code only when it parses AND passes
every extracted example in the sandbox. No reproducible failure, no extractable
example, or a fix that still fails -> defer to remote unchanged.

Nothing here can ship an unverified fix, and the cause line is grounded in a
real execution rather than the model's speculation.
"""
from __future__ import annotations

import ast
import json
import re

from . import code_verify
from .sandbox import run_python

_DEF = re.compile(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(")


def extract_snippet(text: str) -> str | None:
    """The first fenced code block, else None (inline/absent -> defer)."""
    m = code_verify._FENCE.search(text or "")
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


def _func_name(prompt: str, code: str) -> str | None:
    fm = code_verify._FUNC.search(prompt)
    if fm:
        return fm.group(1)
    dm = _DEF.search(code)
    return dm.group(1) if dm else None


def _probe(code: str, tests: list[tuple[str, str]]) -> list | None:
    """Run each example against the code; return per-test [idx, got, ok, err]
    or None when the program could not be run (hang, no output)."""
    lines = [code, "", "import json", "_R = []"]
    for i, (call, exp) in enumerate(tests):
        lines += [
            "try:",
            f"    _v = ({call})",
            "    try:",
            f"        _ok = bool(_v == ({exp}))",
            "    except Exception:",
            "        _ok = None",
            f"    _R.append([{i}, repr(_v)[:80], _ok, None])",
            "except Exception as _e:",
            f"    _R.append([{i}, None, None, type(_e).__name__])",
        ]
    lines.append("print('PROBE:' + json.dumps(_R))")
    result = run_python("\n".join(lines))
    for line in result.stdout.splitlines():
        if line.startswith("PROBE:"):
            try:
                return json.loads(line[len("PROBE:"):])
            except ValueError:
                return None
    return None


def observe_failure(code: str, tests: list[tuple[str, str]]) -> str | None:
    """A one-clause description of how the code misbehaves against the
    examples, or None when it parses and passes them all (nothing to fix) or
    the failure cannot be observed cheaply (defer)."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"the code fails to parse ({exc.msg})"
    probe = _probe(code, tests)
    if not probe:
        return None
    for idx, got, ok, err in probe:
        call = tests[idx][0]
        exp = tests[idx][1]
        if err:
            return f"calling {call} raises {err}"
        if ok is False:
            return f"{call} returns {got} but should return {exp}"
    return None  # passes the evidence: not buggy in a way we can ground


def verify_code_debug(prompt: str, generate_fn, cap: int = 400) -> str | None:
    """Bug sentence + fenced corrected code when a local fix, prompted with the
    observed failure, passes every extracted example; else None (defer)."""
    snippet = extract_snippet(prompt)
    if not snippet:
        return None
    func = _func_name(prompt, snippet)
    tests = code_verify.extract_tests(prompt, func)
    if not tests:
        return None                       # no evidence to verify against
    observed = observe_failure(snippet, tests)
    if observed is None:
        return None                       # not reproducibly buggy -> defer
    system = ("English only. Output ONLY the complete, corrected Python code in "
              "one fenced block. No explanation, no examples.")
    ask = (f"{prompt}\n\nWhen the code is run, {observed}. "
           "Return the corrected code.")
    text = generate_fn(system, ask, cap)
    fix = extract_snippet(text or "")
    if not fix:
        return None
    try:
        ast.parse(fix)
    except SyntaxError:
        return None
    if not code_verify._run(fix, tests):
        return None                       # fix still fails the evidence -> defer
    return f"Bug: {observed}.\n```python\n{fix}\n```"
