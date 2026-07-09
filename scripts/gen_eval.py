#!/usr/bin/env python3
"""Generate a labeled eval set across the 8 Track 1 categories.

Every task is produced from a seeded template whose ground truth is COMPUTED
during generation (arithmetic evaluated, puzzle constraints solved, entities
recorded as they are inserted), so the set can be regenerated endlessly with
fresh values — measuring the agent, never memorization.

Usage:
    python scripts/gen_eval.py --n-per-cat 25 --seed 7 --out data/eval_gen.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIRST = ["Alice", "Bob", "Carol", "David", "Elena", "Farid", "Grace", "Hiro",
         "Ines", "Jamal", "Kira", "Liam", "Mona", "Noah", "Priya", "Quinn"]
ORGS = ["Northwind Labs", "Acme Corporation", "Zenith Bank", "Bluepeak Energy",
        "Talwar Industries", "Helios Motors", "Vertex Health", "Orion Logistics"]
CITIES = ["Toronto", "Nairobi", "Oslo", "Jakarta", "Lima", "Prague", "Seoul",
          "Casablanca", "Adelaide", "Boston"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
ITEMS = ["notebooks", "chairs", "phone cases", "coffee mugs", "backpacks",
         "posters", "keyboards", "plants"]

# ---------------------------------------------------------------- math ------

def gen_math(rng: random.Random, i: int) -> dict:
    kind = rng.choice(["percent", "discount", "rate", "average", "ratio", "two_step"])
    if kind == "percent":
        p, base = rng.choice([5, 8, 12, 15, 20, 25, 30, 40, 60, 75]), rng.randrange(40, 960, 20)
        ans = base * p / 100
        prompt = f"What is {p}% of {base}?"
    elif kind == "discount":
        price = rng.randrange(20, 400, 5)
        p = rng.choice([10, 15, 20, 25, 30, 50])
        ans = round(price * (100 - p) / 100, 2)
        prompt = (f"A jacket costs ${price} and is on sale at {p}% off. "
                  f"What is the final price in dollars?")
    elif kind == "rate":
        per, hours = rng.randrange(8, 45), rng.randrange(3, 12)
        ans = per * hours
        prompt = (f"A worker earns ${per} per hour and works {hours} hours. "
                  f"How much do they earn in total, in dollars?")
    elif kind == "average":
        nums = [rng.randrange(2, 98) for _ in range(rng.choice([4, 5]))]
        while sum(nums) % len(nums):  # force an integer mean: unambiguous truth
            nums[-1] += 1
        ans = sum(nums) // len(nums)
        prompt = f"Calculate the average of {', '.join(map(str, nums))}."
    elif kind == "ratio":
        a, b = rng.choice([(3, 2), (2, 5), (4, 3), (5, 4), (1, 3)])
        mult = rng.randrange(3, 15)
        prompt = (f"The ratio of cats to dogs in a shelter is {a}:{b}. "
                  f"If there are {a * mult} cats, how many dogs are there?")
        ans = b * mult
    else:  # two_step
        n, each = rng.randrange(3, 9), rng.randrange(4, 25)
        fee = rng.randrange(5, 30)
        ans = n * each + fee
        prompt = (f"An order contains {n} {rng.choice(ITEMS)} at ${each} each plus a "
                  f"${fee} delivery fee. What is the total cost in dollars?")
    return {"id": f"math-{i}", "category": "math", "prompt": prompt,
            "grader": {"type": "numeric", "expected": float(ans)}}

# ----------------------------------------------------------- sentiment ------

_POS = ["Absolutely loved it — {x} exceeded every expectation and I would buy it again in a heartbeat.",
        "The {x} was fantastic: quick delivery, great build quality, and superb value.",
        "Five stars. {x} works flawlessly and the support team was wonderful."]
_NEG = ["Terrible experience — the {x} broke within two days and support never replied.",
        "Deeply disappointed: the {x} arrived late, scratched, and missing parts.",
        "Would not recommend. The {x} is flimsy and the manual is useless."]
_NEU = ["The {x} arrived on Tuesday in standard packaging and matches the listed specifications.",
        "The {x} performs as described in the manual; installation took about ten minutes.",
        "I received the {x} yesterday. It includes a cable, a stand, and a warranty card."]

def gen_sentiment(rng: random.Random, i: int) -> dict:
    label = rng.choice(["positive", "negative", "neutral"])
    tpl = rng.choice({"positive": _POS, "negative": _NEG, "neutral": _NEU}[label])
    review = tpl.format(x=rng.choice(ITEMS)[:-1])
    prompt = f"Classify the sentiment of this review and justify briefly: \"{review}\""
    return {"id": f"sentiment-{i}", "category": "sentiment", "prompt": prompt,
            "grader": {"type": "label", "expected": label,
                       "others": [l for l in ("positive", "negative", "neutral") if l != label]}}

# -------------------------------------------------------- summarization -----

def gen_summarization(rng: random.Random, i: int) -> dict:
    org, city = rng.choice(ORGS), rng.choice(CITIES)
    person = rng.choice(FIRST)
    n = rng.randrange(20, 60, 5)
    year = rng.randrange(2019, 2026)
    passage = (
        f"{org} announced on Monday that it will open a new facility in {city}. "
        f"The project, led by {person} Watanabe, is expected to create {n} jobs "
        f"over the next two years. The company was founded in {year} and has "
        f"grown steadily since, expanding into three regional markets. Analysts "
        f"said the move reflects rising demand in the sector, though some warned "
        f"about integration costs. Local officials welcomed the announcement and "
        f"said permitting is already underway."
    )
    prompt = f"Summarize the following text in one sentence: {passage}"
    return {"id": f"summarization-{i}", "category": "summarization", "prompt": prompt,
            "grader": {"type": "summary", "max_sentences": 1,
                       "must_mention_any": [org.split()[0], city]}}

# ------------------------------------------------------------------ ner -----

def gen_ner(rng: random.Random, i: int) -> dict:
    person = f"{rng.choice(FIRST)} {rng.choice(['Okafor', 'Petrov', 'Sandoval', 'Meier', 'Tanaka'])}"
    org, city = rng.choice(ORGS), rng.choice(CITIES)
    day, month, year = rng.randrange(1, 28), rng.choice(MONTHS), rng.randrange(2018, 2026)
    date = f"{month} {day}, {year}"
    sent = (f"On {date}, {person} of {org} announced a partnership during a "
            f"press event held in {city}.")
    prompt = ("Extract the named entities (person, organization, location, date) "
              f"from this text, one per line as 'label: value': \"{sent}\"")
    return {"id": f"ner-{i}", "category": "ner", "prompt": prompt,
            "grader": {"type": "contains_all",
                       "expected": [person, org, city, str(year)]}}

# ---------------------------------------------------------------- logic -----

def gen_logic(rng: random.Random, i: int) -> dict:
    kind = rng.choice(["taller", "order", "syllogism"])
    if kind == "taller":
        a, b, c = rng.sample(FIRST, 3)
        prompt = (f"{a} is taller than {b}. {b} is taller than {c}. "
                  f"Who is the shortest?")
        return {"id": f"logic-{i}", "category": "logic", "prompt": prompt,
                "grader": {"type": "choice", "expected": c,
                           "others": [a, b]}}
    if kind == "order":
        a, b, c, d = rng.sample(FIRST, 4)
        # fixed true order a<b<c<d by finish time (a first)
        prompt = (f"In a race, {a} finished before {b}, {b} finished before {c}, "
                  f"and {c} finished before {d}. Who finished last?")
        return {"id": f"logic-{i}", "category": "logic", "prompt": prompt,
                "grader": {"type": "choice", "expected": d, "others": [a, b, c]}}
    # syllogism -> yes/no
    x, y, z = rng.sample(["bloops", "razzies", "lazzies", "wumps", "glorks"], 3)
    truthy = rng.random() < 0.5
    if truthy:
        prompt = (f"All {x} are {y}. All {y} are {z}. "
                  f"Must all {x} necessarily be {z}? Answer yes or no, briefly explaining why.")
        expected = "yes"
    else:
        prompt = (f"All {x} are {y}. Some {y} are {z}. "
                  f"Must all {x} necessarily be {z}? Answer yes or no, briefly explaining why.")
        expected = "no"
    return {"id": f"logic-{i}", "category": "logic", "prompt": prompt,
            "grader": {"type": "yesno", "expected": expected}}

# -------------------------------------------------------------- factual -----

_FACTS = [
    ("Explain briefly what photosynthesis is.", ["light", "plant"], []),
    ("What is the capital of Australia?", ["Canberra"], ["Sydney"]),
    ("What is the capital of Canada?", ["Ottawa"], ["Toronto"]),
    ("Define the term 'gravity' in simple terms.", ["force"], []),
    ("What does HTTP stand for?", ["hypertext", "transfer", "protocol"], []),
    ("What does CPU stand for?", ["central", "processing", "unit"], []),
    ("Which planet is known as the Red Planet?", ["Mars"], []),
    ("How many continents are there on Earth?", ["seven"], ["six", "eight"]),
    # no must_not here: a correct answer legitimately mentions oxygen as the byproduct
    ("What gas do plants primarily absorb from the atmosphere?", ["carbon dioxide"], []),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"], []),
    ("What is the chemical symbol for water?", ["H2O"], []),
    ("What is the largest ocean on Earth?", ["Pacific"], ["Atlantic"]),
    ("What is the boiling point of water at sea level in Celsius?", ["100"], []),
    ("Explain what an API is in one or two sentences.", ["interface"], []),
    ("What is the primary language spoken in Brazil?", ["Portuguese"], ["Spanish"]),
    ("Which organ pumps blood through the human body?", ["heart"], []),
    ("What does RAM stand for in computing?", ["random", "access", "memory"], []),
    ("Which country is home to the kangaroo?", ["Australia"], []),
    ("What force keeps planets in orbit around the Sun?", ["gravity"], []),
    ("What is the freezing point of water in Fahrenheit?", ["32"], []),
]

def gen_factual(rng: random.Random, i: int) -> dict:
    q, must, must_not = _FACTS[i % len(_FACTS)]
    return {"id": f"factual-{i}", "category": "factual", "prompt": q,
            "grader": {"type": "contains_all", "expected": must,
                       "must_not": must_not}}

# ----------------------------------------------------------------- code -----

_CODE_SPECS = [
    ("fib", "Write a Python function fib(n) that returns the nth Fibonacci number, with fib(0)=0 and fib(1)=1.",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     [("fib(0)", 0), ("fib(1)", 1), ("fib(7)", 13), ("fib(10)", 55)]),
    ("is_palindrome", "Write a Python function is_palindrome(s) that returns True if the string s reads the same forwards and backwards, ignoring case.",
     "def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]",
     [("is_palindrome('Level')", True), ("is_palindrome('python')", False), ("is_palindrome('abba')", True)]),
    ("count_vowels", "Write a Python function count_vowels(s) that returns the number of vowels (a, e, i, o, u, case-insensitive) in the string s.",
     "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')",
     [("count_vowels('Hello World')", 3), ("count_vowels('xyz')", 0), ("count_vowels('AEIOU')", 5)]),
    ("factorial", "Write a Python function factorial(n) that returns n! for a non-negative integer n.",
     "def factorial(n):\n    out = 1\n    for k in range(2, n + 1):\n        out *= k\n    return out",
     [("factorial(0)", 1), ("factorial(5)", 120), ("factorial(7)", 5040)]),
    ("sum_even", "Write a Python function sum_even(nums) that returns the sum of the even numbers in the list nums.",
     "def sum_even(nums):\n    return sum(x for x in nums if x % 2 == 0)",
     [("sum_even([1,2,3,4,5,6])", 12), ("sum_even([1,3,5])", 0), ("sum_even([])", 0)]),
    ("reverse_words", "Write a Python function reverse_words(s) that returns the string s with the order of its words reversed (words are separated by single spaces).",
     "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
     [("reverse_words('the quick brown fox')", "fox brown quick the"), ("reverse_words('hello')", "hello")]),
    ("second_largest", "Write a Python function second_largest(nums) that returns the second largest distinct value in the list nums (the list has at least two distinct values).",
     "def second_largest(nums):\n    return sorted(set(nums))[-2]",
     [("second_largest([3,1,4,1,5])", 4), ("second_largest([10, 10, 9])", 9)]),
]

_BUGS = [  # (broken_fragment, fixed_fragment) applied to reference implementations
    ("a, b = b, a + b", "b, a = a + b, b"),          # swapped update
    ("t == t[::-1]", "t == t[::1]"),                  # missing negative step
    ("ch in 'aeiou'", "ch in 'aeio'"),                # dropped vowel
    ("range(2, n + 1)", "range(2, n)"),               # off-by-one
    ("x % 2 == 0", "x % 2 == 1"),                     # inverted predicate
    ("s.split()[::-1]", "s.split()"),                  # missing reverse
    ("sorted(set(nums))[-2]", "sorted(set(nums))[-1]"),  # wrong index
]

def gen_code_gen(rng: random.Random, i: int) -> dict:
    name, spec, _ref, tests = _CODE_SPECS[i % len(_CODE_SPECS)]
    return {"id": f"code_gen-{i}", "category": "code_gen", "prompt": spec,
            "grader": {"type": "code", "func": name, "tests": tests}}

def gen_code_debug(rng: random.Random, i: int) -> dict:
    name, _spec, ref, tests = _CODE_SPECS[i % len(_CODE_SPECS)]
    good, bad = _BUGS[i % len(_BUGS)][1], _BUGS[i % len(_BUGS)][0]
    # invert: reference contains the GOOD fragment; inject its broken twin
    broken = ref.replace(_BUGS[i % len(_BUGS)][0], _BUGS[i % len(_BUGS)][1]) \
        if _BUGS[i % len(_BUGS)][0] in ref else ref
    if broken == ref:  # bug template not applicable to this spec: swap one op
        broken = ref.replace("+", "-", 1) if "+" in ref else ref.replace("==", "!=", 1)
    prompt = ("This Python function has a bug. Identify it and provide the "
              f"corrected function:\n```python\n{broken}\n```")
    return {"id": f"code_debug-{i}", "category": "code_debug", "prompt": prompt,
            "grader": {"type": "code", "func": name, "tests": tests}}

# ----------------------------------------------------------------- main -----

GENERATORS = {
    "math": gen_math,
    "sentiment": gen_sentiment,
    "summarization": gen_summarization,
    "ner": gen_ner,
    "logic": gen_logic,
    "factual": gen_factual,
    "code_gen": gen_code_gen,
    "code_debug": gen_code_debug,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/eval_gen.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = []
    for cat, gen in GENERATORS.items():
        for i in range(args.n_per_cat):
            tasks.append(gen(rng, i))
    rng.shuffle(tasks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    per = {c: sum(1 for t in tasks if t["category"] == c) for c in GENERATORS}
    print(f"wrote {len(tasks)} tasks -> {out}  {per}")


if __name__ == "__main__":
    main()
