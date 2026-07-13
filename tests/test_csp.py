"""Assignment-CSP logic solver: prove-or-defer over bijective puzzles.

Every positive case is exact; every ambiguous, oversized, or unparsed case
defers (None). Variants randomize names/values/order to prove the solver is
parametric, not memorized.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frugal_router.solvers import solve, solve_any  # noqa: E402


def test_canonical_pet_puzzle():
    p = ("Sam, Jo, and Lee each own a different pet: cat, dog, bird. "
         "Sam does not own the bird. Jo owns the dog. Who owns the cat?")
    assert solve(p, "logic") == "Sam"
    assert solve_any(p) == ("Sam", "logic")


def test_variant_names_values_order_changed():
    # Tom=oak; Maria != pine -> Maria=elm; Priya=pine.
    p = ("Maria, Tom, and Priya each planted a different tree: oak, elm, pine. "
         "Maria did not plant the pine. Tom planted the oak. Who planted the pine?")
    assert solve(p, "logic") == "Priya"


def test_variant_which_question_returns_value():
    p = ("Ana, Ben, and Cara each chose a different color: red, blue, green. "
         "Ben chose green. Ana did not choose blue. Which color did Cara choose?")
    # Ben=green; Ana != blue -> Ana=red; Cara=blue.
    assert solve(p, "logic") == "blue"


def test_four_entities():
    p = ("Al, Bo, Cy, and Di each drive a different car: sedan, coupe, van, jeep. "
         "Al drives the van. Bo does not drive the sedan. Bo does not drive the coupe. "
         "Cy drives the coupe. Who drives the sedan?")
    # Al=van, Cy=coupe; Bo not sedan/coupe -> Bo=jeep; Di=sedan.
    assert solve(p, "logic") == "Di"


def test_defer_multiple_solutions():
    # Only one clue -> elm/pine ambiguous between Maria and Priya.
    p = ("Maria, Tom, and Priya each planted a different tree: oak, elm, pine. "
         "Tom planted the oak. Who planted the pine?")
    assert solve(p, "logic") is None


def test_defer_unparsed_relational_clue():
    # A clue mentioning two names is not a pairwise pin/forbid -> defer.
    p = ("Maria, Tom, and Priya each planted a different tree: oak, elm, pine. "
         "Tom planted the oak. Maria planted a taller tree than Priya. "
         "Who planted the pine?")
    assert solve(p, "logic") is None


def test_defer_no_different_keyword():
    # Without the bijection signal ("different"), the setup is not proven.
    p = ("Sam, Jo, and Lee own a cat, a dog, and a bird. "
         "Jo owns the dog. Who owns the cat?")
    assert solve(p, "logic") is None


def test_defer_too_many_entities():
    p = ("A, Bo, Cy, Di, Ed, and Fi each hold a different card: one, two, three, "
         "four, five, six. Bo holds the two. Who holds the one?")
    assert solve(p, "logic") is None


def test_defer_contradictory_no_solution():
    p = ("Sam, Jo, and Lee each own a different pet: cat, dog, bird. "
         "Sam owns the dog. Jo owns the dog. Who owns the cat?")
    assert solve(p, "logic") is None
