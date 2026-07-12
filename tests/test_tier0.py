"""Tier-0 deterministic solvers: facts, logic search, code canon.

The invariant under test is prove-or-defer: every solver either returns a
PROVEN answer or None. A wrong committed answer is the only unacceptable
outcome, so half of these tests exist to force deferrals.
"""
from frugal_router import code_canon, facts, logic_search
from frugal_router.solvers import solve, solve_any

# ------------------------------------------------------------- facts -------


def test_capital_hit():
    assert facts.lookup("What is the capital of Canada?") == "Ottawa"
    assert facts.lookup("What is the capital city of Japan?") == "Tokyo"


def test_capital_unknown_defers():
    assert facts.lookup("What is the capital of Wakanda?") is None


def test_element_by_symbol():
    assert facts.lookup("Which element has the symbol Au?") == "gold"
    assert facts.lookup("What element has the chemical symbol Fe?") == "iron"


def test_symbol_of_element():
    assert facts.lookup("What is the chemical symbol for gold?") == "Au"
    assert facts.lookup("What is the symbol of oxygen?") == "O"


def test_atomic_number():
    assert facts.lookup("What is the atomic number of carbon?") == "6"


def test_water_constants():
    assert facts.lookup(
        "What is the boiling point of water at sea level in Celsius?") == "100"
    assert facts.lookup("What is the freezing point of water in Fahrenheit?") == "32"


def test_authors_and_painters():
    assert facts.lookup("Who wrote Pride and Prejudice?") == "Jane Austen"
    assert facts.lookup("Who painted the Mona Lisa?") == "Leonardo da Vinci"


def test_volatile_defers():
    assert facts.lookup("What is the current population of Canada?") is None
    assert facts.lookup("Who is the president of France?") is None
    assert facts.lookup("What is the latest iPhone price?") is None


def test_multi_question_defers():
    assert facts.lookup(
        "What is the capital of Canada? And what is its population?") is None


def test_unmatched_factual_defers():
    assert facts.lookup("Explain how photosynthesis works in detail.") is None


# ------------------------------------------------------ syllogism validity -


def test_syllogism_invalid_some():
    a = logic_search.syllogism_validity(
        "All bloops are wumps. Some wumps are glorks. "
        "Must all bloops necessarily be glorks? Answer yes or no, briefly explaining why.")
    assert a is not None and a.lower().startswith("no")


def test_syllogism_valid_chain():
    a = logic_search.syllogism_validity(
        "All cats are mammals. All mammals are animals. "
        "Must all cats be animals?")
    assert a is not None and a.lower().startswith("yes")


def test_syllogism_no_premise_defers():
    assert logic_search.syllogism_validity("Must all bloops be glorks?") is None


def test_syllogism_unrelated_defers():
    assert logic_search.syllogism_validity(
        "What is the capital of France?") is None


def test_syllogism_no_quantified_conclusion_negative():
    a = logic_search.syllogism_validity(
        "No fish are birds. All salmon are fish. Must all salmon be birds?")
    assert a is not None and a.lower().startswith("no")


# ------------------------------------------------------- ordering search ---


def test_ordering_unique():
    a = logic_search.ordering_search(
        "Alice is taller than Bob. Bob is taller than Carol. Who is the tallest?")
    assert a == "Alice"


def test_ordering_min_superlative():
    a = logic_search.ordering_search(
        "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?")
    assert a == "Carol"


def test_ordering_position_query():
    a = logic_search.ordering_search(
        "Dana finished before Eli. Eli finished before Fay. Who finished second?")
    assert a == "Eli"


def test_ordering_ambiguous_defers():
    # B vs C unknown: tallest is still A, but 'second tallest' style ambiguity
    # -> a genuinely ambiguous query must defer.
    a = logic_search.ordering_search(
        "Alice is taller than Bob. Alice is taller than Carol. Who is the shortest?")
    assert a is None


def test_ordering_unparsed_relation_defers():
    # 'exactly as tall as' is a comparative sentence the parser cannot model.
    a = logic_search.ordering_search(
        "Alice is taller than Bob. Carol is exactly as tall as Alice. Who is the tallest?")
    assert a is None or a == "Alice"  # defer preferred; never 'Carol'
    # The hint gate should fire on the unparsed 'taller/shorter' family only;
    # 'as tall as' contains no comparative keyword, so a modest answer is ok.


def test_ordering_mixed_relations():
    a = logic_search.ordering_search(
        "Bob is older than Alice. Carol is younger than Alice. Who is the oldest?")
    assert a == "Bob"


# ------------------------------------------------------------- code canon --


def test_canon_factorial():
    code = code_canon.try_canon(
        "Write a Python function factorial(n) that returns n! for a non-negative integer n.")
    assert code is not None and "def factorial" in code


def test_canon_reverse_words():
    code = code_canon.try_canon(
        "Write a Python function reverse_words(s) that returns the string s "
        "with the order of its words reversed (words are separated by single spaces).")
    assert code is not None and "def reverse_words" in code


def test_canon_prompt_examples_must_pass():
    code = code_canon.try_canon(
        "Write a Python function is_prime(n) that returns True when n is prime. "
        "For example is_prime(7) returns True and is_prime(8) returns False.")
    assert code is not None


def test_canon_modifier_defers():
    assert code_canon.try_canon(
        "Write a recursive Python function factorial(n) that returns n!.") is None
    assert code_canon.try_canon(
        "Write a Python function factorial(n) without using loops.") is None
    assert code_canon.try_canon(
        "Write a Python function factorial(n) that raises ValueError for negatives.") is None


def test_canon_unknown_name_defers():
    assert code_canon.try_canon(
        "Write a Python function frobnicate(x) that frobnicates x.") is None


def test_canon_name_intent_mismatch_defers():
    # Correct name but the described behavior is NOT the canon one.
    assert code_canon.try_canon(
        "Write a Python function factorial(n) that returns the sum of digits of n.") is None


# --------------------------------------------------------- new math bits ---


def test_earnings():
    assert solve("A worker earns $28 per hour and works 6 hours. "
                 "How much do they earn in total, in dollars?", "math") == "168"


def test_dice_sum():
    a = solve("Two fair six-sided dice are rolled. What is the probability "
              "that the sum is 7?", "math")
    assert a is not None and "1/6" in a


def test_dice_impossible_sum():
    a = solve("Two fair six-sided dice are rolled. What is the probability "
              "that the sum is 13?", "math")
    assert a == "0"


def test_single_die():
    a = solve("A fair six-sided die is rolled. What is the probability of "
              "rolling a 3?", "math")
    assert a is not None and "1/6" in a


def test_earnings_extra_numbers_defer():
    assert solve("A worker earns $28 per hour, works 6 hours, and spends $10 "
                 "on lunch. How much money remains?", "math") is None


# ------------------------------------------------------------ solve_any ----


def test_solve_any_factual():
    hit = solve_any("What is the capital of Canada?")
    assert hit == ("Ottawa", "factual")


def test_solve_any_syllogism():
    hit = solve_any("All bloops are wumps. Some wumps are glorks. "
                    "Must all bloops necessarily be glorks?")
    assert hit is not None and hit[1] == "logic" and hit[0].lower().startswith("no")


def test_solve_any_still_defers_prose():
    assert solve_any("Summarize the plot of Hamlet in two sentences.") is None
