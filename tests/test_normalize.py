"""Normalization must repair shape without ever destroying content."""
from frugal_router.normalize import normalize


def test_sentiment_label_leads():
    out = normalize("sentiment", "The review is glowing, so this is positive overall.")
    assert out.lower().startswith("positive")


def test_sentiment_conflicting_labels_untouched():
    text = "It reads positive in parts but negative in others."
    assert normalize("sentiment", text) == text


def test_math_answer_line_appended():
    out = normalize("math", "15% of 240 is 36, since 240 * 0.15 = 36")
    assert out.splitlines()[-1] == "Answer: 36"


def test_math_existing_answer_line_kept():
    text = "Step 1: compute.\nAnswer: 42"
    assert normalize("math", text) == text


def test_logic_yesno_answer_line():
    out = normalize("logic", "All bloops are lazzies follows transitively, so yes.")
    assert out.splitlines()[-1] == "Answer: Yes"


def test_logic_names_not_guessed():
    text = "Carol is shorter than Bob, who is shorter than Alice."
    assert "Answer:" not in normalize("logic", text)


def test_code_gen_reduced_to_fence():
    text = "Here's the function you asked for:\n```python\ndef f():\n    return 1\n```\nHope that helps!"
    out = normalize("code_gen", text)
    assert out.startswith("```python") and out.endswith("```")
    assert "Hope" not in out


def test_code_debug_keeps_explanation():
    text = "The bug is an off-by-one.\n```python\ndef f():\n    return 1\n```"
    assert "off-by-one" in normalize("code_debug", text)


def test_preamble_stripped():
    out = normalize("factual", "Sure, here's the answer: Paris is the capital of France.")
    assert out.startswith("Paris")


def test_ner_json_converted_to_lines():
    out = normalize("ner", '{"person": ["Marie Curie"], "location": ["Paris"]}')
    assert "person: Marie Curie" in out and "location: Paris" in out


def test_never_empty():
    assert normalize("math", "no numbers here at all") == "no numbers here at all"
    assert normalize("sentiment", "") == ""


def test_summary_prefix_stripped():
    out = normalize("summarization", "Summary: The company expanded to Oslo.")
    assert out.startswith("The company")
