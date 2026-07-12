"""Adversarial solver cases: prove-or-defer means every ambiguous, unit-mixed,
negated, or under-specified prompt must return None (defer to the model), and
every claimed hit must be exactly right. A wrong zero-token answer is worse
than no solver at all."""
from frugal_router.solvers import solve_any


DEFER_CASES = [
    # unit mixing / conversion traps
    "A car travels at 60 km/h for 90 minutes. How many miles does it cover?",
    "Convert 2.5 kilograms to pounds and round to one decimal.",
    # wording ambiguity
    "What is fifteen percent, roughly, of a bit more than 240?",
    "How much is a dozen and a half plus a score?",
    # multiple quantities where the target is unclear
    "A shop sells pens at $3 and pencils at $2. Tom buys 7 items. How much?",
    # negation flips
    "All bloops are NOT razzies. Some razzies are lazzies. Must all bloops be lazzies?",
    "Alice is not taller than Bob. Bob is not taller than Carol. Who is tallest?",
    # ordering with an unconsumed relation
    "Priya finished before Quinn but after Ravi in one heat and after Quinn in another. Who won overall?",
    # percent-of with a follow-up twist the solver cannot see
    "What is 15% of 240, minus the number of days in February 2024?",
    # assignment CSP traps: under-constrained, so no unique proof
    ("Sam, Jo, and Lee each own a different pet: cat, dog, bird. "
     "Sam does not own the bird. Who owns the cat?"),
    # a clue relating two entities is not a pairwise pin -> unparsed -> defer
    ("Ana, Ben, and Cara each chose a different color: red, blue, green. "
     "Ana chose a brighter color than Ben. Ben chose red. Who chose green?"),
    # no bijection signal ("different" absent): setup unproven
    ("Sam, Jo, and Lee own a cat, a dog, and a bird. Jo owns the dog. "
     "Who owns the cat?"),
    # contradictory clues -> zero solutions -> defer
    ("Sam, Jo, and Lee each own a different pet: cat, dog, bird. "
     "Sam owns the dog. Jo owns the dog. Who owns the cat?"),
]

EXACT_CASES = [
    ("What is 15% of 240?", "36"),
    ("Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?", "Carol"),
]


def test_adversarial_prompts_defer():
    answered = []
    for prompt in DEFER_CASES:
        hit = solve_any(prompt)
        if hit is not None:
            answered.append((prompt, hit))
    assert not answered, (
        "solver answered prompts it cannot prove:\n" +
        "\n".join(f"  {p!r} -> {h}" for p, h in answered)
    )


def test_known_hits_stay_exact():
    for prompt, expected in EXACT_CASES:
        hit = solve_any(prompt)
        assert hit is not None, f"expected a solver hit for {prompt!r}"
        assert hit[0] == expected
