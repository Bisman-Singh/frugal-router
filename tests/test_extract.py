from frugal_router.extract import (
    extract_answer,
    final_answer,
    is_valid_answer,
    normalize_number,
    text_key,
    vote_key,
)


def test_extract_prefers_last_answer_line():
    text = "Let me think.\nAnswer: 12\nWait, correction.\nAnswer: 14"
    assert extract_answer(text) == "14"


def test_extract_falls_back_to_last_line():
    assert extract_answer("some reasoning\n42") == "42"


def test_extract_empty():
    assert extract_answer("") is None
    assert extract_answer(None) is None


def test_normalize_number():
    assert normalize_number("$1,234.50") == "1234.5"
    assert normalize_number("42%") == "42"
    assert normalize_number("the total is 18 apples") == "18"
    assert normalize_number("no digits here") is None
    assert normalize_number(None) is None


def test_vote_key_math():
    assert vote_key("Step 1: 3 * 7 = 21.\nAnswer: $21.00", "math") == "21"
    assert vote_key("The sum is 15 + 6 = 21", "math") == "21"  # last number wins
    assert vote_key("Answer: twenty-one", "math") is None


def test_vote_key_sentiment_is_the_label():
    a = vote_key("Answer: Positive - the reviewer says they loved it.", "sentiment")
    b = vote_key("Answer: positive - enthusiastic wording throughout.", "sentiment")
    assert a == b == "positive"


def test_vote_key_full_style_uses_whole_text():
    assert vote_key("- item one\n- item two", "ner") == "- item one - item two"


def test_final_answer_math_is_the_number():
    assert final_answer("Working...\nAnswer: 21 dollars", "math") == "21"


def test_final_answer_line_style_keeps_raw_line():
    text = "Reasoning here.\nAnswer: Positive - the reviewer loved it."
    assert final_answer(text, "sentiment") == "Positive - the reviewer loved it."


def test_final_answer_full_style_keeps_everything():
    code = "The bug is a missing colon.\n```python\ndef f():\n    return 1\n```"
    assert final_answer(code, "code_debug") == code


def test_is_valid_answer():
    assert is_valid_answer("21", "math")
    assert not is_valid_answer("about right", "math")
    assert is_valid_answer("Positive - the reviewer says they loved it.", "sentiment")
    assert not is_valid_answer("Positive", "sentiment")  # label without justification
    assert is_valid_answer("```python\ndef f(): pass\n```", "code_gen")
    assert not is_valid_answer("I cannot write code.", "code_gen")
    assert not is_valid_answer("", "factual")
    assert not is_valid_answer(None, "factual")
    assert is_valid_answer("Canberra", "factual")


def test_text_key():
    assert text_key("  Paris.  ") == "paris"
    assert text_key('"Paris"') == "paris"
    assert text_key("") is None
