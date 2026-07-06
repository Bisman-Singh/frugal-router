from frugal_router.extract import extract_answer, is_valid, normalize


def test_extract_prefers_last_answer_line():
    text = "Let me think.\nAnswer: 12\nWait, correction.\nAnswer: 14"
    assert extract_answer(text) == "14"


def test_extract_falls_back_to_last_line():
    assert extract_answer("some reasoning\n42") == "42"


def test_extract_empty():
    assert extract_answer("") is None
    assert extract_answer(None) is None


def test_normalize_math():
    assert normalize("$1,234.50", "math") == "1234.5"
    assert normalize("42%", "math") == "42"
    assert normalize("the total is 18 apples", "math") == "18"
    assert normalize("no digits here", "math") is None


def test_normalize_mcq():
    assert normalize("(b)", "mcq") == "B"
    assert normalize("The answer is C.", "mcq") == "C"
    assert normalize("banana", "mcq") is None


def test_normalize_general():
    assert normalize("  Positive.  ", "classification") == "positive"
    assert normalize('"Paris"', "general") == "paris"


def test_is_valid():
    assert is_valid("42", "math")
    assert not is_valid("about 42", "math")
    assert is_valid("B", "mcq")
    assert not is_valid("AB", "mcq")
    assert not is_valid("", "general")
    assert not is_valid(None, "general")
    assert is_valid("positive", "classification")
