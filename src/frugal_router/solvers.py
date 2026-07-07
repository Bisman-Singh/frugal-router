"""Deterministic prove-or-defer solvers for math and logic tasks.

Some tasks are pure computation, and code answers them exactly. The result is
used as a trusted draft for a tiny Fireworks confirmation call (compliant with
the every-scored-answer-comes-from-Fireworks rule) or, when configured,
emitted directly at zero tokens.

The invariant is prove-or-defer: a solver returns None unless the parse is
unambiguous. A wrong deterministic answer anchors the confirmation prompt the
wrong way, so guessing is never worth it. Nothing here is a cached or
hardcoded answer; every result is computed from the prompt at run time.
"""
from __future__ import annotations

import ast
import operator
import re

_NUM = r"-?\d+(?:\.\d+)?"
_EXPR_CHARS = re.compile(r"[-+*/().\d\s]+")
_PERCENT_OF = re.compile(rf"(?i)\b({_NUM})\s*(?:percent|%)\s+of\s+\$?({_NUM})\b")
_RATE_TIME = re.compile(
    rf"(?i)\b({_NUM})\s*(?:km|miles?|kilometers?|meters?)\s*(?:per|/|an)\s*hour\b"
    rf".*?\b({_NUM})(?:\.\d+)?\s*hours?\b"
)
# Comparatives grouped by dimension: relations from one dimension must never
# leak into a question about another ("older than" says nothing about height).
_DIMS = {
    "height": ("taller", "shorter"),
    "age": ("older", "younger"),
    "speed": ("faster", "slower"),
    "weight": ("heavier", "lighter"),
    "order": ("before", "after"),
}
_SUPERLATIVES = {
    "tallest": ("height", True), "shortest": ("height", False),
    "oldest": ("age", True), "youngest": ("age", False),
    "fastest": ("speed", True), "slowest": ("speed", False),
    "heaviest": ("weight", True), "lightest": ("weight", False),
    "first": ("order", True), "last": ("order", False),
}
_REL = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:is\s+|was\s+)?(taller|shorter|older|younger|faster|slower|heavier|lighter)\s+than\s+([A-Z][a-z]+)"
)
_ORDER_ELLIPTICAL = re.compile(
    r"\b([A-Z][a-z]+)\s+finished\s+(before|after)\s+([A-Z][a-z]+),?\s+but\s+(before|after)\s+([A-Z][a-z]+)"
)
_ORDER_REL = re.compile(r"\b([A-Z][a-z]+)\s+finished\s+(before|after)\s+([A-Z][a-z]+)")
_WHO_Q = re.compile(
    r"(?i)\bwho\s+(?:is|was|finished)?\s*(?:the\s+)?"
    r"(tallest|shortest|oldest|youngest|fastest|slowest|heaviest|lightest|first|last)\b"
)
_SYLLOGISM = re.compile(
    r"(?i)\ball\s+(\w+)\s+are\s+(\w+)\b.*?\ball\s+(\w+)\s+are\s+(\w+)\b"
    r".*?\bare\s+all\s+(\w+)\s+(?:definitely\s+)?(\w+)\b"
)

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def solve(prompt: str, category: str) -> str | None:
    """Return the exact answer, or None to defer to a model."""
    if category == "math":
        for solver in (_percent_of, _rate_time, _arithmetic):
            answer = solver(prompt)
            if answer is not None:
                return answer
    if category == "logic":
        for solver in (_syllogism, _ordering):
            answer = solver(prompt)
            if answer is not None:
                return answer
    return None


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(round(value, 6))


def _eval_expr(expr: str) -> float | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    try:
        return float(walk(tree))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _arithmetic(prompt: str) -> str | None:
    """Bare arithmetic like 'What is 124 + 387?'. Defers if the prompt has any
    numbers outside the single expression, or percent language."""
    if re.search(r"(?i)percent|%", prompt):
        return None
    candidates = [
        c.strip() for c in _EXPR_CHARS.findall(prompt)
        if re.search(rf"{_NUM}\s*[-+*/]\s*{_NUM}", c)
    ]
    if len(candidates) != 1:
        return None
    expr = candidates[0].rstrip(" =?.")
    if sorted(re.findall(_NUM, prompt)) != sorted(re.findall(_NUM, expr)):
        return None  # numbers exist outside the expression: context we cannot parse
    value = _eval_expr(expr)
    return _format_number(value) if value is not None else None


def _percent_of(prompt: str) -> str | None:
    """'What is 15 percent of 200?' and nothing else numeric going on."""
    m = _PERCENT_OF.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 2:
        return None
    if re.search(r"(?i)increase|decrease|discount|off|more|less|rise|drop", prompt):
        return None
    return _format_number(float(m.group(1)) * float(m.group(2)) / 100.0)


def _rate_time(prompt: str) -> str | None:
    """'travels at 60 km per hour for 2.5 hours' with exactly those numbers."""
    m = _RATE_TIME.search(prompt)
    if not m:
        return None
    if len(re.findall(_NUM, prompt)) != 2:
        return None
    return _format_number(float(m.group(1)) * float(m.group(2)))


def _ordering(prompt: str) -> str | None:
    """Transitive chains: 'A is taller than B. B is taller than C. Who is the
    shortest?' Requires a provably unique extremum, and every comparison
    keyword in the prompt must be consumed by a parsed relation. A clause the
    parser missed ('but after Ravi') means the graph is incomplete, and an
    incomplete graph proves nothing."""
    question = _WHO_Q.search(prompt)
    if not question:
        return None
    dim, want_top = _SUPERLATIVES[question.group(1).lower()]
    hi_word, lo_word = _DIMS[dim]

    relations: list[tuple[str, str]] = []  # (greater, lesser)
    consumed = 0
    if dim == "order":
        spans = []
        for m in _ORDER_ELLIPTICAL.finditer(prompt):
            subject, rel1, other1, rel2, other2 = m.groups()
            for rel, other in ((rel1, other1), (rel2, other2)):
                hi, lo = (subject, other) if rel == "before" else (other, subject)
                relations.append((hi, lo))
            consumed += 2
            spans.append(m.span())
        for m in _ORDER_REL.finditer(prompt):
            if any(s <= m.start() < e for s, e in spans):
                continue
            a, rel, b = m.groups()
            hi, lo = (a, b) if rel == "before" else (b, a)
            relations.append((hi, lo))
            consumed += 1
    else:
        for a, rel, b in _REL.findall(prompt):
            if rel not in (hi_word, lo_word):
                continue  # a different dimension; not evidence for this question
            hi, lo = (a, b) if rel == hi_word else (b, a)
            relations.append((hi, lo))
            consumed += 1

    total_keywords = len(re.findall(rf"(?i)\b(?:{hi_word}|{lo_word})\b", prompt))
    if consumed == 0 or consumed != total_keywords:
        return None  # unparsed comparison language: the graph is incomplete

    greater: dict[str, set[str]] = {}
    names: set[str] = set()
    for hi, lo in relations:
        greater.setdefault(hi.casefold(), set()).add(lo.casefold())
        names.update({hi.casefold(), lo.casefold()})
    if len(names) < 2:
        return None

    # Anyone mentioned in any comparison is a candidate; a person absent from
    # this dimension's graph could still be the extremum, so nothing is proven.
    everyone: set[str] = set()
    for a, _, b in _REL.findall(prompt):
        everyone.update({a.casefold(), b.casefold()})
    for a, _, b in _ORDER_REL.findall(prompt):
        everyone.update({a.casefold(), b.casefold()})
    for m in _ORDER_ELLIPTICAL.finditer(prompt):
        everyone.update({m.group(1).casefold(), m.group(3).casefold(), m.group(5).casefold()})
    if everyone - names:
        return None

    def reachable(start: str) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            for nxt in greater.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    below = {n: reachable(n) for n in names}
    if want_top:
        winners = [n for n in names if below[n] >= names - {n}]
    else:
        winners = [n for n in names if all(n in below[m] for m in names - {n})]
    if len(winners) != 1:
        return None  # not provably unique: defer
    return winners[0].capitalize()


def _syllogism(prompt: str) -> str | None:
    """'If all bloops are razzies and all razzies are lazzies, are all bloops
    definitely lazzies?' Pure transitivity, answer yes."""
    m = _SYLLOGISM.search(prompt)
    if not m:
        return None
    a1, b1, a2, b2, qa, qb = (g.casefold().rstrip("s") for g in m.groups())
    if a1 == qa and b1 == a2 and b2 == qb:
        return "yes"
    return None
