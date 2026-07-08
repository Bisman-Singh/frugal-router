from frugal_router.solvers import solve


def test_bare_arithmetic():
    assert solve("What is 124 + 387?", "math") == "511"
    assert solve("Calculate 12 * 7.", "math") == "84"


def test_arithmetic_defers_on_word_problems():
    # Numbers outside the expression mean context the solver cannot prove.
    assert solve("Tom had 23 marbles. He lost 9 and found 4. 23 - 9 + 4 friends?", "math") is None
    assert solve("A bakery sells muffins for $3 each. Maria buys 7. Total?", "math") is None


def test_percent_of():
    assert solve("What is 15 percent of 200?", "math") == "30"
    assert solve("What is 12.5% of 80?", "math") == "10"


def test_percent_defers_on_change_language():
    assert solve("A $12 subscription increases by 25 percent. New price?", "math") is None


def test_rate_time():
    assert solve("A train travels at 60 km per hour for 2.5 hours. How far?", "math") == "150"


def test_speed_from_distance_time():
    assert solve("A car travels 240 kilometers in 3 hours. What is its average speed?", "math") == "80"
    # no speed keyword -> defer (could be asking something else)
    assert solve("A car travels 240 km in 3 hours then stops.", "math") is None


def test_rectangle_perimeter_and_area():
    assert solve("A rectangle has a length of 8 meters and a width of 6 meters. What is its perimeter?", "math") == "28"
    assert solve("A rectangle with length 8 and width 6. What is its area?", "math") == "48"


def test_cost_word_problems_defer_to_model():
    # Cost phrasing varies too much to parse safely; prove-or-defer hands these
    # to the model rather than risk a misfire.
    assert solve("A store sells notebooks at $4 each. Priya buys 9 notebooks. Total cost?", "math") is None
    assert solve("Priya buys 9 notebooks at $4 each and pays with a $50 bill. Change?", "math") is None


def test_ordering_unique_extremum():
    prompt = "Ali is taller than Ben. Ben is taller than Carl. Who is the shortest?"
    assert solve(prompt, "logic") == "Carl"


def test_ordering_elliptical_subject():
    race = "In a race, Priya finished before Quinn but after Ravi. Who finished first?"
    assert solve(race, "logic") == "Ravi"


def test_ordering_defers_when_ambiguous():
    prompt = "Ali is taller than Ben. Dana is taller than Carl. Who is the tallest?"
    assert solve(prompt, "logic") is None


def test_ordering_defers_on_unparsed_comparison_language():
    # 'much taller than everyone else' has a 'taller' the parser cannot consume.
    prompt = "Ali is taller than Ben. Omar is much taller than everyone else. Who is the tallest?"
    assert solve(prompt, "logic") is None


def test_ordering_ignores_other_dimensions():
    # Age relations are not height evidence; with only one height relation the
    # tallest is not provable over three people.
    prompt = "Ali is taller than Ben. Carl is older than Ali. Who is the tallest?"
    assert solve(prompt, "logic") is None


def test_syllogism():
    prompt = (
        "If all bloops are razzies and all razzies are lazzies, "
        "are all bloops definitely lazzies? Answer yes or no."
    )
    assert solve(prompt, "logic") == "yes"


def test_syllogism_defers_on_some():
    prompt = "If some bloops are razzies and all razzies are lazzies, are all bloops lazzies?"
    assert solve(prompt, "logic") is None


def test_wrong_category_defers():
    assert solve("What is 124 + 387?", "factual") is None
