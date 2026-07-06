from frugal_router.classify import classify
from frugal_router.tasks import Task


def make(text, **kwargs):
    return Task(id="t", input=text, **kwargs)


def test_explicit_type_wins():
    assert classify(make("What is 2 + 2?", type="general")) == "general"


def test_choices_imply_mcq():
    assert classify(make("Pick one", choices=["a", "b"])) == "mcq"


def test_math():
    assert classify(make("What is 12 + 7?")) == "math"
    assert classify(make("How many apples are left if you eat 3 of 10?")) == "math"


def test_mcq_from_text():
    assert classify(make("Which of the following is a mammal?")) == "mcq"


def test_classification():
    assert classify(make("Classify the sentiment of this review: great!")) == "classification"


def test_extraction_with_context():
    assert classify(make("Who wrote it?", context="Some long passage.")) == "extraction"


def test_summarization():
    assert classify(make("Summarize this paragraph: once upon a time...")) == "summarization"


def test_general_fallback():
    assert classify(make("What is the capital of France?")) == "general"
