"""Execution-verified local code generation.

A locally generated function is kept at ZERO tokens ONLY when it passes tests
extracted verbatim from the prompt (doctest / "f(x) returns y" style). If no
tests can be extracted, or the code fails/does not parse, the task defers to
remote unchanged. ast.parse is a syntax check; execution is the correctness
gate. Nothing here can ship unverified code.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_FUNC = re.compile(r"(?i)function\s+(?:named\s+|called\s+)?`?([A-Za-z_]\w*)`?")
# "f(2) should return 3", "f(2) -> 3", "f(2) == 3", "f(2) returns 3", ">>> f(2)\n3"
_EXAMPLE = re.compile(
    r"(?im)(?:>>>\s*)?([A-Za-z_]\w*\([^()]*\))\s*"
    r"(?:should\s+return|returns?|->|==|gives?|yields?|=)\s*"
    r"([^\n.,;]+?)\s*(?:[\n.,;]|$)")


def extract_tests(prompt: str, func: str | None) -> list[tuple[str, str]]:
    tests = []
    for call, expected in _EXAMPLE.findall(prompt):
        if func and not call.startswith(func + "("):
            continue
        exp = expected.strip().strip("`'\" ")
        if not exp:
            continue
        tests.append((call.strip(), exp))
    return tests[:6]


def _run(code: str, tests: list[tuple[str, str]], timeout: float = 5.0) -> bool:
    lines = [code, ""]
    for call, exp in tests:
        lines.append(f"assert ({call}) == ({exp})")
    lines.append("print('ALLPASS')")
    prog = "\n".join(lines)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog)
            path = f.name
        out = subprocess.run(["python3", "-I", path], capture_output=True,
                             timeout=timeout, env={})
        os.unlink(path)
        return b"ALLPASS" in out.stdout
    except Exception:
        return False


def verify_code_gen(prompt: str, generate_fn, cap: int = 400) -> str | None:
    """Fenced code if a local generation passes prompt-extracted tests, else
    None. generate_fn(system, prompt, cap) -> text (the local model)."""
    fm = _FUNC.search(prompt)
    func = fm.group(1) if fm else None
    tests = extract_tests(prompt, func)
    if not tests:
        return None                      # nothing to prove against -> defer
    system = ("English only. Output ONLY the complete, correct, self-contained "
              "Python code in one fenced block. No explanation, no examples.")
    text = generate_fn(system, prompt, cap)
    m = _FENCE.search(text or "")
    code = (m.group(1) if m else (text or "")).strip()
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    if _run(code, tests):
        return f"```python\n{code}\n```"
    return None
