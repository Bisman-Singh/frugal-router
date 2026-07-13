"""Search-based logic solvers: proven answers or None, never a guess.

Two engines:

1. Syllogism validity via finite model checking. Quantified statements over
   unary predicates ("All bloops are wumps") only depend on which predicate
   COMBINATIONS are inhabited, so enumerating every subset of the 2^k type
   space (k predicates -> 2^(2^k) worlds, k<=3 -> 256) is a sound AND
   complete decision procedure for this fragment: "yes" means the conclusion
   holds in every world satisfying the premises, "no" means a concrete
   counterexample world exists. Both are proofs, not heuristics.

2. Ordering puzzles via permutation search. Parse comparative relations,
   enumerate all orderings of the named entities, keep those satisfying
   every constraint, and answer only when all surviving orderings agree.
   If any sentence looks comparative but did not parse into a constraint,
   the solver defers — a missing constraint could flip the answer.
"""
from __future__ import annotations

import itertools
import re

# ------------------------------------------------ syllogism validity -------

_PLURAL = r"[a-z]+?"
_QUANT = re.compile(
    rf"(?i)\b(all|every|some|no)\s+({_PLURAL})s?\s+(?:are|is)\s+(not\s+)?(?:a\s+|an\s+)?({_PLURAL})s?\b")
_CONCLUSION_Q = re.compile(
    r"(?i)\b(?:must|do|does|are|is|can)\b[^?.]*?\b(all|every|some|any|no)\s+"
    rf"({_PLURAL})s?\b[^?.]*?\b(?:necessarily\s+)?(?:be|are|is)\s+(?:necessarily\s+)?"
    rf"(?:a\s+|an\s+)?({_PLURAL})s?\s*\?")
_FOLLOWS_Q = re.compile(
    rf"(?i)\bdoes\s+it\s+follow\s+that\s+(all|every|some|no)\s+({_PLURAL})s?\s+"
    rf"(?:are|is)\s+(?:a\s+|an\s+)?({_PLURAL})s?\s*\?")


def _norm(word: str) -> str:
    return word.lower().rstrip("s")


def _eval_stmt(world: frozenset, quant: str, a: int, negated: bool, b: int) -> bool:
    """Evaluate one quantified statement in a world (a set of inhabited
    predicate-bitmask types)."""
    a_bit, b_bit = 1 << a, 1 << b
    a_types = [t for t in world if t & a_bit]
    if quant in ("all", "every"):
        if negated:
            return all(not (t & b_bit) for t in a_types)
        return all(t & b_bit for t in a_types)
    if quant == "some":
        if negated:
            return any(not (t & b_bit) for t in a_types)
        return any(t & b_bit for t in a_types)
    if quant == "no":
        return all(not (t & b_bit) for t in a_types)
    return False


def _verdict(premises, concl, n_preds, satisfiability) -> str | None:
    """'yes'/'no' by exhaustive world enumeration, None if premises are
    contradictory (a parse artifact, never worth answering from)."""
    n_types = 1 << n_preds
    holds_all, consistent = True, False
    for bits in range(1 << n_types):
        world = frozenset(t for t in range(n_types) if bits & (1 << t))
        if not all(_eval_stmt(world, *p) for p in premises):
            continue
        consistent = True
        if _eval_stmt(world, *concl):
            if satisfiability:
                return "yes"
        else:
            holds_all = False
            if not satisfiability:
                return "no"
    if not consistent:
        return None
    if satisfiability:
        return "no"
    return "yes" if holds_all else "no"


def syllogism_validity(prompt: str) -> str | None:
    """Yes/no for 'must all X be Y'-style entailment questions."""
    q = _CONCLUSION_Q.search(prompt) or _FOLLOWS_Q.search(prompt)
    if not q:
        return None
    cq, ca, cb = q.group(1).lower(), _norm(q.group(2)), _norm(q.group(3))
    if cq == "every":
        cq = "all"

    # Premises = quantified statements before the question mark position.
    premises = []
    preds: dict[str, int] = {}

    def _pred(name: str) -> int:
        if name not in preds:
            preds[name] = len(preds)
        return preds[name]

    setup = prompt[:q.start()]
    parsed_spans = []
    for m in _QUANT.finditer(setup + " "):
        quant, a, neg, b = m.group(1).lower(), _norm(m.group(2)), bool(m.group(3)), _norm(m.group(4))
        if quant == "every":
            quant = "all"
        premises.append((quant, _pred(a), neg, _pred(b)))
        parsed_spans.append(m.span())
    if not premises or ca not in preds or cb not in preds:
        return None                      # unparsed setup -> defer
    if len(preds) > 4:
        return None                      # 2^(2^5) worlds is too many; defer

    # Safety gate: a quantified-looking sentence that produced no premise
    # means the model of the puzzle is incomplete -> defer.
    for m in re.finditer(r"(?i)\b(all|every|some|no|none)\b[^.?]*\b(?:are|is)\b", setup):
        if not any(s <= m.start() and m.end() <= e + 1 for s, e in parsed_spans):
            return None

    # "Can/any X be Y?" asks satisfiability, not entailment.
    satisfiability = cq in ("some", "any") or re.search(r"(?i)\bcan\b", q.group(0)) is not None
    concl = ("some" if satisfiability else "all", preds[ca], False, preds[cb])

    # 'All X are not Y' is ambiguous English ('no X are Y' vs 'some X are
    # not Y'). Evaluate every reading; commit only when the verdict is the
    # same under all of them — then the answer is proven under any parse.
    readings = [premises]
    if any(p[0] == "all" and p[2] for p in premises):
        alt = [("some", p[1], True, p[3]) if (p[0] == "all" and p[2]) else p
               for p in premises]
        readings.append(alt)
    verdicts = {_verdict(r, concl, len(preds), satisfiability) for r in readings}
    if len(verdicts) != 1 or None in verdicts:
        return None
    v = verdicts.pop()
    if satisfiability:
        return ("Yes. The premises allow it: there is a consistent scenario "
                "where this holds." if v == "yes" else
                "No. The premises rule it out in every consistent scenario.")
    return ("Yes. In every scenario consistent with the premises, the "
            "conclusion holds." if v == "yes" else
            "No. Not necessarily: the premises admit a scenario in which "
            "the conclusion fails, so it does not follow.")


# ------------------------------------------------ ordering by search -------

_NAME = r"[A-Z][a-z]+"
# comparative word -> (dimension, direction). +1: left entity is higher.
_CMP = {
    "taller": ("height", 1), "shorter": ("height", -1),
    "older": ("age", 1), "younger": ("age", -1),
    "faster": ("speed", 1), "slower": ("speed", -1),
    "heavier": ("weight", 1), "lighter": ("weight", -1),
    "bigger": ("size", 1), "larger": ("size", 1), "smaller": ("size", -1),
    "stronger": ("strength", 1), "weaker": ("strength", -1),
    "richer": ("wealth", 1), "poorer": ("wealth", -1),
    "higher": ("level", 1), "lower": ("level", -1),
    "more expensive": ("price", 1), "less expensive": ("price", -1),
    "cheaper": ("price", -1),
    "before": ("order", 1), "ahead of": ("order", 1),
    "after": ("order", -1), "behind": ("order", -1),
}
# superlative -> (dimension, want-max)
_SUP = {
    "tallest": ("height", True), "shortest": ("height", False),
    "oldest": ("age", True), "youngest": ("age", False),
    "fastest": ("speed", True), "slowest": ("speed", False),
    "heaviest": ("weight", True), "lightest": ("weight", False),
    "biggest": ("size", True), "largest": ("size", True), "smallest": ("size", False),
    "strongest": ("strength", True), "weakest": ("strength", False),
    "richest": ("wealth", True), "poorest": ("wealth", False),
    "highest": ("level", True), "lowest": ("level", False),
    "most expensive": ("price", True), "cheapest": ("price", False),
    "first": ("order", True), "last": ("order", False),
}
_REL = re.compile(
    rf"({_NAME})\s+(?:is|was|ran|runs)?\s*(?:much\s+)?"
    rf"({'|'.join(w for w in _CMP if w not in ('before', 'after', 'ahead of', 'behind'))})"
    rf"\s+than\s+({_NAME})")
_BEFORE = re.compile(rf"({_NAME})\s+(?:finished|arrived|came|ranks?|placed)\s+"
                     rf"(before|after|ahead of|behind)\s+({_NAME})")
_SUPER_Q = re.compile(rf"(?i)\bwho\s+(?:is|was)\s+the\s+({'|'.join(_SUP)})\b")
_POSITION_Q = re.compile(r"(?i)\bwho\s+(?:finished|arrived|came|placed)\s+(first|second|third|fourth|fifth|last)\b")
_COMPARATIVE_HINT = re.compile(
    rf"\b(?:{'|'.join(_CMP)})\b", re.I)

_ORDINAL = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "last": -1}


def ordering_search(prompt: str) -> str | None:
    """Unique answer to a superlative/position question over parsed
    comparative constraints, or None. Constraints are tracked per dimension:
    age relations are never evidence about height. Every entity named in any
    relation stays a candidate, so an entity unconstrained in the queried
    dimension keeps the answer ambiguous and the solver defers."""
    by_dim: dict[str, list] = {}
    entities: set[str] = set()
    parsed_spans = []

    for m in _REL.finditer(prompt):
        a, rel, b = m.group(1), m.group(2).lower(), m.group(3)
        dim, sign = _CMP[rel]
        entities.update((a, b))
        by_dim.setdefault(dim, []).append((a, b) if sign > 0 else (b, a))
        parsed_spans.append(m.span())
    for m in _BEFORE.finditer(prompt):
        a, rel, b = m.group(1), m.group(2).lower(), m.group(3)
        dim, sign = _CMP[rel]
        entities.update((a, b))
        by_dim.setdefault(dim, []).append((a, b) if sign > 0 else (b, a))
        parsed_spans.append(m.span())

    if len(entities) < 2 or len(entities) > 8 or not by_dim:
        return None

    # Safety gate: a comparative phrase that did NOT become a constraint
    # means the model of the puzzle is incomplete -> defer.
    for m in _COMPARATIVE_HINT.finditer(prompt):
        if not any(s <= m.start() and m.end() <= e for s, e in parsed_spans):
            return None

    sq = _SUPER_Q.search(prompt)
    pq = _POSITION_Q.search(prompt)
    if sq:
        dim, want_max = _SUP[sq.group(1).lower()]
    elif pq:
        dim, want_max = "order", True
    else:
        return None
    constraints = by_dim.get(dim)
    if not constraints:
        return None                      # no evidence in the asked dimension

    names = sorted(entities)
    answers = set()
    for perm in itertools.permutations(names):
        rank = {n: i for i, n in enumerate(perm)}   # 0 = top of scale / first
        if any(rank[w] >= rank[l] for w, l in constraints):
            continue
        if sq:
            answers.add(perm[0] if want_max else perm[-1])
        else:
            answers.add(perm[_ORDINAL[pq.group(1).lower()]])
        if len(answers) > 1:
            return None                  # ambiguous across valid orderings
    if len(answers) == 1:
        return answers.pop()
    return None
