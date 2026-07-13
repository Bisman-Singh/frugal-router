#!/usr/bin/env python3
"""Archetype-variant generator for the Track 1 public task shapes.

Final scoring re-randomizes the public prompts, so we measure the zero-token
tiers on randomized variants of those archetypes, never the fixed set. Every
task's ground truth is COMPUTED here, independently of the solver under test
(CSP puzzles are built from a fixed random assignment and the answer is read
back from it), so this is a real measurement, not a mirror of the code.

Deterministic under --seed; a new seed yields a fresh, disjoint sample.

Usage:
    python scripts/gen_variants.py --per 12 --seed 7 --out data/variants.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import permutations
from pathlib import Path

NAMES = ["Alice", "Bob", "Carol", "David", "Elena", "Farid", "Grace", "Hiro",
         "Ines", "Jamal", "Kira", "Liam", "Mona", "Noah", "Priya", "Quinn"]
ORGS = ["Northwind Labs", "Acme Corporation", "Zenith Bank", "Helios Motors",
        "Vertex Health", "Orion Logistics", "Bluepeak Energy"]
CITIES = ["Toronto", "Nairobi", "Oslo", "Jakarta", "Lima", "Prague", "Seoul"]
MONTHS = ["January", "March", "June", "September", "November"]
ITEMS = ["notebooks", "chairs", "phone cases", "mugs", "backpacks", "keyboards"]

# (category noun, base-form verb, [values]) for the assignment-CSP archetype.
# Verbs are base form so "{verb}s" ("owns", "picks") stays grammatical.
CSP_DOMAINS = [
    ("pet", "own", ["cat", "dog", "bird", "fish", "rabbit"]),
    ("color", "pick", ["red", "blue", "green", "yellow", "purple"]),
    ("tree", "plant", ["oak", "elm", "pine", "maple", "birch"]),
    ("drink", "order", ["tea", "coffee", "juice", "water", "cola"]),
    ("sport", "play", ["tennis", "chess", "rugby", "hockey", "golf"]),
]


def _num(v: float):
    return int(v) if float(v).is_integer() else round(v, 4)


# ------------------------------------------------------------------ math -----

def gen_math(rng, i):
    kind = rng.choice(["percent", "discount", "increase", "average", "fraction",
                       "rate", "interest", "rectangle"])
    if kind == "percent":
        p, b = rng.choice([5, 12, 15, 20, 25, 40]), rng.randrange(40, 900, 20)
        prompt, ans = f"What is {p}% of {b}?", b * p / 100
    elif kind == "discount":
        b, p = rng.randrange(40, 400, 5), rng.choice([10, 20, 25, 30, 50])
        prompt = f"A jacket costs ${b} and is {p}% off. What is the final price?"
        ans = b * (1 - p / 100)
    elif kind == "increase":
        b, p = rng.randrange(50, 500, 10), rng.choice([10, 15, 20, 25])
        prompt = f"A ${b} item increased by {p}%. What is the new price?"
        ans = b * (1 + p / 100)
    elif kind == "average":
        vals = [rng.randrange(2, 96) for _ in range(rng.choice([4, 5]))]
        while sum(vals) % len(vals):
            vals[-1] += 1
        prompt = f"What is the average of {', '.join(map(str, vals))}?"
        ans = sum(vals) / len(vals)
    elif kind == "fraction":
        d = rng.choice([2, 4, 5, 8]); n = rng.randrange(1, d)
        whole = rng.randrange(1, 20) * d
        prompt = f"What is {n}/{d} of {whole}?"
        ans = n / d * whole
    elif kind == "rate":
        s, h = rng.randrange(30, 90, 5), rng.choice([2, 3, 4, 5])
        prompt = f"A train travels at {s} km per hour for {h} hours. How far does it travel?"
        ans = s * h
    elif kind == "interest":
        pr, r, t = rng.randrange(200, 2000, 100), rng.choice([3, 4, 5, 6]), rng.choice([2, 3, 4])
        prompt = f"${pr} is invested at {r}% simple interest for {t} years. How much interest is earned?"
        ans = pr * r / 100 * t
    else:  # rectangle
        L, W = rng.randrange(4, 40), rng.randrange(3, 30)
        prompt = f"A rectangle has length {L} and width {W}. What is its perimeter?"
        ans = 2 * (L + W)
    return {"id": f"math-{i}", "category": "math", "prompt": prompt,
            "grader": {"type": "numeric", "expected": float(_num(ans))}}


# ----------------------------------------------------------------- logic -----

def _unique_solution(names, values, constraints):
    sols = []
    for perm in permutations(values):
        a = dict(zip(names, perm))
        if all((a[nm] == v) == pos for nm, v, pos in constraints):
            sols.append(a)
        if len(sols) > 1:
            return None
    return sols[0] if len(sols) == 1 else None


def gen_logic_csp(rng, i):
    n = rng.choice([3, 3, 4])
    noun, verb, pool = rng.choice(CSP_DOMAINS)
    names = rng.sample(NAMES, n)
    values = rng.sample(pool, n)
    truth = dict(zip(names, rng.sample(values, n)))
    # Build clues (mix of positive pins and negative exclusions) until unique.
    order = names[:]
    rng.shuffle(order)
    constraints, sentences = [], []
    pinned = set()
    for nm in order:
        if rng.random() < 0.55:
            v = truth[nm]
            constraints.append((nm, v, True))
            sentences.append(f"{nm} {verb}s the {v}.")
            pinned.add(nm)
        else:
            wrong = rng.choice([v for v in values if v != truth[nm]])
            constraints.append((nm, wrong, False))
            sentences.append(f"{nm} does not {verb} the {wrong}.")
        if _unique_solution(names, values, constraints) is not None:
            break
    # Negatives alone can under-determine the puzzle; add positive pins until it
    # is provably unique (full pinning always is), so this stays a logic task.
    for nm in names:
        if _unique_solution(names, values, constraints) is not None:
            break
        if nm not in pinned:
            constraints.append((nm, truth[nm], True))
            sentences.append(f"{nm} {verb}s the {truth[nm]}.")
    namelist = ", ".join(names[:-1]) + f", and {names[-1]}"
    asked_value = rng.choice(values)
    answer = next(nm for nm, v in truth.items() if v == asked_value)
    rng.shuffle(sentences)
    prompt = (f"{namelist} each {verb} a different {noun}: {', '.join(values)}. "
              + " ".join(sentences) + f" Who {verb}s the {asked_value}?")
    return {"id": f"logic-{i}", "category": "logic", "prompt": prompt,
            "grader": {"type": "string", "expected": answer}}


def gen_logic_order(rng, i):
    people = rng.sample(NAMES, rng.choice([3, 4]))
    pairs = " ".join(f"{people[j]} finished before {people[j + 1]},"
                     for j in range(len(people) - 1)).rstrip(",") + "."
    ask_last = rng.random() < 0.5
    answer = people[-1] if ask_last else people[0]
    prompt = (f"In a race, {pairs} Who finished {'last' if ask_last else 'first'}?")
    return {"id": f"logic-{i}", "category": "logic", "prompt": prompt,
            "grader": {"type": "string", "expected": answer}}


# -------------------------------------------------------------- code lanes ---

def gen_code_gen(rng, i):
    kind = rng.choice(["double", "sum_list", "count_vowels", "max_list"])
    if kind == "double":
        x = rng.randrange(2, 20)
        spec = ("Write a Python function `double` that returns its argument "
                f"times two. For example, double({x}) returns {x * 2}.")
        func, tests = "double", [[f"double({x})", x * 2], ["double(0)", 0]]
    elif kind == "sum_list":
        spec = ("Write a Python function `sum_list` that returns the sum of a "
                "list of numbers. For example, sum_list([1, 2, 3]) returns 6.")
        func, tests = "sum_list", [["sum_list([1, 2, 3])", 6], ["sum_list([])", 0]]
    elif kind == "count_vowels":
        spec = ("Write a Python function `count_vowels` that returns the number "
                "of vowels (aeiou) in a string. For example, count_vowels('hello') returns 2.")
        func, tests = "count_vowels", [["count_vowels('hello')", 2], ["count_vowels('xyz')", 0]]
    else:
        spec = ("Write a Python function `max_list` that returns the largest "
                "number in a list. For example, max_list([3, 1, 5]) returns 5.")
        func, tests = "max_list", [["max_list([3, 1, 5])", 5], ["max_list([-1, -4])", -1]]
    return {"id": f"code_gen-{i}", "category": "code_gen", "prompt": spec,
            "grader": {"type": "code", "func": func, "tests": tests}}


def gen_code_debug(rng, i):
    kind = rng.choice(["max_first", "off_by_one", "wrong_op"])
    if kind == "max_first":
        buggy = "def get_max(nums):\n    return nums[0]"
        prompt = ("The function `get_max` should return the largest number in a "
                  "list. For example, get_max([3, 1, 5]) returns 5. Find and fix the bug.")
        func, tests = "get_max", [["get_max([3, 1, 5])", 5], ["get_max([2, 9, 4])", 9]]
    elif kind == "off_by_one":
        buggy = "def total(n):\n    return sum(range(n))"
        prompt = ("The function `total` should return the sum of 1..n inclusive. "
                  "For example, total(5) returns 15. Find and fix the bug.")
        func, tests = "total", [["total(5)", 15], ["total(3)", 6]]
    else:
        buggy = "def area(w, h):\n    return w + h"
        prompt = ("The function `area` should return the area of a rectangle. "
                  "For example, area(3, 4) returns 12. Find and fix the bug.")
        func, tests = "area", [["area(3, 4)", 12], ["area(5, 2)", 10]]
    full = f"{prompt}\n```python\n{buggy}\n```"
    return {"id": f"code_debug-{i}", "category": "code_debug", "prompt": full,
            "grader": {"type": "code", "func": func, "tests": tests}}


# ---------------------------------------------- sentiment / ner / summ / fact -

def gen_sentiment(rng, i):
    label = rng.choice(["positive", "negative", "neutral"])
    x = rng.choice(ITEMS)[:-1]
    text = {
        "positive": f"I absolutely love this {x} — it exceeded every expectation.",
        "negative": f"This {x} broke within a day and support never replied. Awful.",
        "neutral": f"The {x} arrived on schedule and matches the listed specifications.",
    }[label]
    prompt = (f'Classify the sentiment of this review as positive, negative, or '
              f'neutral: "{text}"')
    return {"id": f"sentiment-{i}", "category": "sentiment", "prompt": prompt,
            "grader": {"type": "label", "expected": label}}


def gen_ner(rng, i):
    person = rng.choice(NAMES) + " " + rng.choice(["Meier", "Okafor", "Tan", "Silva"])
    org, city = rng.choice(ORGS), rng.choice(CITIES)
    date = f"{rng.choice(MONTHS)} {rng.randrange(1, 28)}, 20{rng.randrange(10, 24)}"
    text = f"On {date}, {person} of {org} spoke at a conference in {city}."
    prompt = ("Extract the named entities (person, organization, location, date) "
              f'from this text, one per line as \'label: value\': "{text}"')
    return {"id": f"ner-{i}", "category": "ner", "prompt": prompt,
            "grader": {"type": "ner",
                       "entities": [["person", person], ["organization", org],
                                    ["location", city], ["date", date]]}}


def gen_summary(rng, i):
    org, city = rng.choice(ORGS), rng.choice(CITIES)
    n = rng.randrange(20, 400, 5)
    passage = (f"{org} announced today that it will open a new facility in {city}, "
               f"creating {n} jobs over the next two years. The company said the "
               f"site will focus on research and development.")
    prompt = f"Summarize the following text in one sentence: {passage}"
    return {"id": f"summary-{i}", "category": "summarization", "prompt": prompt,
            "grader": {"type": "summary", "max_sentences": 1,
                       "must_mention_any": [org, city]}}


FACTS = [
    ("photosynthesis", ["light", "energy", "plant"]),
    ("gravity", ["force", "mass", "attract"]),
    ("inflation", ["price", "money", "rise"]),
    ("evaporation", ["liquid", "gas", "vapor"]),
]


def gen_factual(rng, i):
    topic, keys = rng.choice(FACTS)
    prompt = f"Explain what {topic} is in two to three sentences."
    return {"id": f"factual-{i}", "category": "factual", "prompt": prompt,
            "grader": {"type": "rubric", "must_mention_any": keys}}


ARCHETYPES = {
    "math": [gen_math],
    "logic": [gen_logic_csp, gen_logic_order],
    "code_gen": [gen_code_gen],
    "code_debug": [gen_code_debug],
    "sentiment": [gen_sentiment],
    "ner": [gen_ner],
    "summarization": [gen_summary],
    "factual": [gen_factual],
}


def generate(per: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out, idx = [], 0
    for gens in ARCHETYPES.values():
        for _ in range(per):
            idx += 1
            out.append(rng.choice(gens)(rng, idx))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12, help="variants per category")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/variants.jsonl")
    args = ap.parse_args()
    tasks = generate(args.per, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks across {len(ARCHETYPES)} categories to {args.out}")


if __name__ == "__main__":
    main()
