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
import math
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
        return _first_hit(prompt, _MATH_SOLVERS)
    if category == "logic":
        return _first_hit(prompt, _LOGIC_SOLVERS)
    if category == "factual":
        from . import facts
        return facts.lookup(prompt)
    return None


def solve_any(prompt: str) -> tuple[str, str] | None:
    """Try every solver regardless of the task's category and return
    (answer, category) on a proven hit. Prove-or-defer keeps this safe: a
    solver returns None unless the parse is unambiguous, so running math
    solvers on a factual prompt simply finds nothing. This decouples free
    answers from classification, which routinely misses 'perimeter' or
    'average speed' as math."""
    answer = _first_hit(prompt, _MATH_SOLVERS)
    if answer is not None:
        return answer, "math"
    answer = _first_hit(prompt, _LOGIC_SOLVERS)
    if answer is not None:
        return answer, "logic"
    from . import facts
    answer = facts.lookup(prompt)
    if answer is not None:
        return answer, "factual"
    return None


def _first_hit(prompt: str, solvers) -> str | None:
    for solver in solvers:
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


_ARITH_TRAPS = re.compile(
    r"(?i)\b(binary|hexadecimal|hex|octal|base|roman|remainder|modulo|mod|"
    r"prime|factor|digit|rounded|round|nearest|estimate)\b"
)


def _arithmetic(prompt: str) -> str | None:
    """Bare arithmetic like 'What is 124 + 387?'. Defers if the prompt has any
    numbers outside the single expression, percent language, or wording that
    changes the meaning of the digits (base conversions, rounding, etc.)."""
    if re.search(r"(?i)percent|%", prompt) or _ARITH_TRAPS.search(prompt):
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


_SPEED = re.compile(
    rf"(?i)\btravels?\s+({_NUM})\s*(?:km|kilometers?|miles?|m)\b.*?\bin\s+({_NUM})\s*hours?\b"
)


def _speed(prompt: str) -> str | None:
    """'travels 240 km in 3 hours, average speed?' -> distance / time."""
    if not re.search(r"(?i)\b(speed|how fast|per hour)\b", prompt):
        return None
    m = _SPEED.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 2:
        return None
    dist, time = float(m.group(1)), float(m.group(2))
    if time == 0:
        return None
    return _format_number(dist / time)


_RECT_LW = re.compile(rf"length\s+(?:of\s+)?({_NUM})\b.*?\bwidth\s+(?:of\s+)?({_NUM})", re.I | re.S)
_RECT_WL = re.compile(rf"width\s+(?:of\s+)?({_NUM})\b.*?\blength\s+(?:of\s+)?({_NUM})", re.I | re.S)


def _rectangle(prompt: str) -> str | None:
    """Rectangle perimeter or area from an explicit length and width."""
    if not re.search(r"(?i)\brectangle\b", prompt):
        return None
    m = _RECT_LW.search(prompt) or _RECT_WL.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 2:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if re.search(r"(?i)\bperimeter\b", prompt):
        return _format_number(2 * (a + b))
    if re.search(r"(?i)\barea\b", prompt):
        return _format_number(a * b)
    return None




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


_PCT_TOKEN = re.compile(rf"({_NUM})\s*(?:percent|%)")


def _base_and_pct(prompt: str) -> tuple[float, float] | None:
    """Exactly two numbers, one clearly a percentage -> (base, pct). None if
    ambiguous. Base is the number that is NOT the percent token."""
    pm = _PCT_TOKEN.search(prompt)
    if not pm or len(re.findall(rf"(?i)percent|%", prompt)) != 1:
        return None
    if len(re.findall(_NUM, prompt)) != 2:
        return None
    pct = float(pm.group(1))
    base = None
    for nm in re.finditer(_NUM, prompt):
        if nm.start() == pm.start(1):
            continue
        base = float(nm.group())
    return (base, pct) if base is not None else None


def _discount(prompt: str) -> str | None:
    """'A jacket costs $140 and is 30% off. Final price?' Single base + pct."""
    low = prompt.lower()
    if not re.search(r"(?i)\b(off|discount|reduced|sale|markdown)\b", low):
        return None
    if "then" in low or "additional" in low or "another" in low:
        return None  # stacked -> different solver
    bp = _base_and_pct(prompt)
    if bp is None:
        return None
    base, pct = bp
    if re.search(r"(?i)how much.*(save|saved|discount|less)|discount amount|amount saved", low):
        return _format_number(base * pct / 100.0)
    return _format_number(base * (1 - pct / 100.0))


def _percent_increase(prompt: str) -> str | None:
    """'A $200 item increased by 15%. New price?' Single base + pct."""
    low = prompt.lower()
    if not re.search(r"(?i)\b(increase|increased|rises?|rose|grew|grows?|markup|marked up|more)\b", low):
        return None
    if "then" in low or "additional" in low or re.search(r"(?i)\b(off|discount)\b", low):
        return None
    bp = _base_and_pct(prompt)
    if bp is None:
        return None
    base, pct = bp
    return _format_number(base * (1 + pct / 100.0))


_STACK = re.compile(rf"(?i)({_NUM})\s*(?:percent|%)\s*off.*?(?:then|additional|another|plus)\s*(?:an\s+)?({_NUM})\s*(?:percent|%)\s*off", re.S)


def _stacked_discount(prompt: str) -> str | None:
    """'20% off, then an additional 10% off. Original $200. Final?'"""
    m = _STACK.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 3:
        return None
    d1, d2 = float(m.group(1)), float(m.group(2))
    base = None
    used = {m.group(1), m.group(2)}
    for nm in re.finditer(_NUM, prompt):
        if nm.group() not in used or nm.start() not in (m.start(1), m.start(2)):
            if nm.start() not in (m.start(1), m.start(2)):
                base = float(nm.group())
    if base is None:
        return None
    return _format_number(base * (1 - d1 / 100.0) * (1 - d2 / 100.0))


_COMPOUND = re.compile(
    rf"(?i)\$?({_NUM})\b.*?({_NUM})\s*(?:percent|%).*?(?:for|over)\s+({_NUM})\s*years?", re.S)


def _compound_interest(prompt: str) -> str | None:
    """'$1000 at 5% compounded annually for 2 years. Final amount?'"""
    if not re.search(r"(?i)compound", prompt):
        return None
    m = _COMPOUND.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 3:
        return None
    p, r, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if not t.is_integer() or t > 50:
        return None
    amount = p * (1 + r / 100.0) ** t
    if re.search(r"(?i)\binterest\s+earned|how much interest\b", prompt):
        return _format_number(amount - p)
    return _format_number(round(amount, 2))


_SIMPLE = re.compile(
    rf"(?i)\$?({_NUM})\b.*?({_NUM})\s*(?:percent|%).*?(?:for|over)\s+({_NUM})\s*years?", re.S)


def _simple_interest(prompt: str) -> str | None:
    """'$500 at 4% simple interest for 3 years. Interest earned?'"""
    if not re.search(r"(?i)simple\s+interest", prompt):
        return None
    m = _SIMPLE.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 3:
        return None
    p, r, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
    interest = p * r / 100.0 * t
    if re.search(r"(?i)\btotal|final amount|balance\b", prompt):
        return _format_number(p + interest)
    return _format_number(interest)


_AVG = re.compile(rf"(?i)\b(?:average|mean)\s+of\s+([-\d.,\sand]+?)(?:\?|$|\.)")


def _average(prompt: str) -> str | None:
    """'What is the average of 10, 20, and 30?'"""
    m = _AVG.search(prompt)
    if not m:
        return None
    vals = [float(x) for x in re.findall(_NUM, m.group(1))]
    if len(vals) < 2 or len(vals) != len(re.findall(_NUM, prompt)):
        return None
    return _format_number(sum(vals) / len(vals))


_FRACTION = re.compile(rf"(?i)\b({_NUM})\s*/\s*({_NUM})\s+of\s+\$?({_NUM})\b")


def _fraction_of(prompt: str) -> str | None:
    """'What is 3/4 of 200?'"""
    m = _FRACTION.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 3:
        return None
    num, den, whole = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if den == 0:
        return None
    return _format_number(num / den * whole)


_RADIUS = re.compile(rf"(?i)\bradius\s+(?:of\s+)?({_NUM})")
_DIAM = re.compile(rf"(?i)\bdiameter\s+(?:of\s+)?({_NUM})")


def _circle(prompt: str) -> str | None:
    """Circle area or circumference from radius or diameter (pi=3.14159)."""
    import math
    if not re.search(r"(?i)\bcircle|circular\b", prompt):
        return None
    if len(re.findall(_NUM, prompt)) != 1:
        return None
    rm, dm = _RADIUS.search(prompt), _DIAM.search(prompt)
    if rm:
        r = float(rm.group(1))
    elif dm:
        r = float(dm.group(1)) / 2.0
    else:
        return None
    if re.search(r"(?i)\barea\b", prompt):
        return _format_number(round(math.pi * r * r, 4))
    if re.search(r"(?i)\bcircumference|perimeter\b", prompt):
        return _format_number(round(2 * math.pi * r, 4))
    return None


_TRI = re.compile(rf"(?i)\bbase\s+(?:of\s+)?({_NUM})\b.*?\bheight\s+(?:of\s+)?({_NUM})", re.S)


def _triangle(prompt: str) -> str | None:
    """Triangle area from base and height."""
    if not re.search(r"(?i)\btriangle\b", prompt) or not re.search(r"(?i)\barea\b", prompt):
        return None
    m = _TRI.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 2:
        return None
    return _format_number(0.5 * float(m.group(1)) * float(m.group(2)))


_EARN = re.compile(
    rf"(?i)\b(?:earns?|paid|makes?|charges?)\s+\$?({_NUM})\s*(?:dollars?\s*)?"
    rf"(?:per|an?|each|/)\s*hour\b.*?\b({_NUM})\s*hours?\b", re.S)


def _earnings(prompt: str) -> str | None:
    """'earns $28 per hour and works 6 hours' -> wage x hours."""
    m = _EARN.search(prompt)
    if not m or len(re.findall(_NUM, prompt)) != 2:
        return None
    return _format_number(float(m.group(1)) * float(m.group(2)))


_DICE_SUM = re.compile(rf"(?i)\bsum\s+(?:of\s+|is\s+|equals?\s+)?({_NUM})\b")


def _dice_sum(prompt: str) -> str | None:
    """Probability that two fair six-sided dice sum to N."""
    if not re.search(r"(?i)\b(?:two|2|pair of)\b.*?\bdice\b", prompt, re.S):
        return None
    if not re.search(r"(?i)\b(probability|chance|likelihood|odds)\b", prompt):
        return None
    m = _DICE_SUM.search(prompt)
    if not m:
        return None
    target = float(m.group(1))
    if not target.is_integer():
        return None
    ways = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == int(target))
    if ways == 0:
        return "0"
    g = math.gcd(ways, 36)
    return f"{ways}/36 = {ways // g}/{36 // g} (about {ways / 36:.4f})"


def _single_die(prompt: str) -> str | None:
    """Probability of one specific face on a fair six-sided die."""
    if not re.search(r"(?i)\b(?:a|one|single|fair)\s+(?:six-sided\s+|6-sided\s+)?(?:die|dice)\b", prompt):
        return None
    if re.search(r"(?i)\btwo\b|\bpair\b|\bboth\b", prompt):
        return None
    if not re.search(r"(?i)\b(probability|chance|likelihood|odds)\b", prompt):
        return None
    if not re.search(r"(?i)\brolling\s+(?:a\s+|an\s+)?[1-6]\b", prompt):
        return None
    return "1/6 (about 0.1667)"


_MATH_SOLVERS = (_percent_of, _discount, _percent_increase, _stacked_discount,
                 _compound_interest, _simple_interest, _average, _fraction_of,
                 _circle, _triangle, _rate_time, _speed, _rectangle, _earnings,
                 _dice_sum, _single_die, _arithmetic)


def _syllogism_validity(prompt: str) -> str | None:
    from . import logic_search
    return logic_search.syllogism_validity(prompt)


def _ordering_search(prompt: str) -> str | None:
    from . import logic_search
    return logic_search.ordering_search(prompt)


_LOGIC_SOLVERS = (_syllogism, _ordering, _syllogism_validity, _ordering_search)
