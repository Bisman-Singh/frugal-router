from frugal_router.classify import classify
from frugal_router.tasks import Task


def make(text, **kwargs):
    return Task(id="t", input=text, **kwargs)


def test_explicit_type_wins():
    assert classify(make("What is 2 + 2?", type="factual")) == "factual"


def test_sentiment():
    assert classify(make("What is the sentiment of this review: 'great product'?")) == "sentiment"


def test_summarization():
    assert classify(make("Summarize this paragraph in two sentences: once upon a time...")) == "summarization"


def test_ner():
    assert classify(make("Extract all named entities from the following text.")) == "ner"
    assert classify(make("Extract the people, organizations and dates mentioned below.")) == "ner"


def test_code_debug():
    prompt = "This function crashes with an error. Fix it:\ndef f(x):\n    return x +"
    assert classify(make(prompt)) == "code_debug"


def test_code_gen():
    assert classify(make("Write a Python function that reverses a string.")) == "code_gen"


def test_logic():
    assert classify(make("Ali is taller than Ben. Ben is taller than Carl. Who is the shortest?")) == "logic"


def test_math():
    assert classify(make("What is 12 + 7?")) == "math"
    assert classify(make("How many apples are left if you eat 3 of 10?")) == "math"


def test_factual_fallback():
    assert classify(make("What is the capital of France?")) == "factual"
