"""Classification coverage: borderline phrasings must land in the right bucket.

Each case is a real-world surface form of a category that a thin keyword matcher
would drop into 'factual'. A mis-route means the wrong answer contract/format,
which the judge scores as wrong even when the model knew the answer.
"""
from frugal_router.classify import classify
from frugal_router.tasks import Task


CASES = [
    # (prompt, expected_category)
    ("What is 15% of 240?", "math"),
    ("A shirt costs $40 with a 25% discount. What's the final price?", "math"),
    ("How much interest accrues on $1000 at 5% over 3 years?", "math"),
    ("Round 3.14159 to two decimal places.", "math"),
    ("The ratio of cats to dogs is 3:2. If there are 15 cats, how many dogs?", "math"),

    ("What's the mood of this review: 'The plot dragged and I nearly left.'", "sentiment"),
    ("Classify the emotional tone of this tweet.", "sentiment"),
    ("Is the author happy or upset in this message?", "sentiment"),
    ("Rate the sentiment of the following comment.", "sentiment"),

    ("Give me the key points of this article.", "summarization"),
    ("Summarize the passage in two sentences.", "summarization"),
    ("What's the gist of the text below?", "summarization"),
    ("Boil this paragraph down to one line.", "summarization"),
    ("TL;DR the following.", "summarization"),

    ("List all the people mentioned in the passage below.", "ner"),
    ("Extract every organization and location from this text.", "ner"),
    ("Identify the named entities in the following.", "ner"),
    ("Pull out all the dates referenced here.", "ner"),

    ("Who owns the zebra? Each house has a different pet and exactly one owner.", "logic"),
    ("If all bloops are razzies and all razzies are lazzies, must all bloops be lazzies?", "logic"),
    ("Alice is taller than Bob and Bob is taller than Carol. Who is shortest?", "logic"),
    ("Solve this riddle: the knight always tells the truth, the knave always lies.", "logic"),

    ("Write a function that returns the nth Fibonacci number.", "code_gen"),
    ("Implement a Python script to reverse a linked list.", "code_gen"),
    ("Generate code that checks if a string is a palindrome.", "code_gen"),

    ("Fix this code: def add(a,b): return a-b", "code_debug"),
    ("Why doesn't my loop terminate?\n```\nwhile x>0: print(x)\n```", "code_debug"),
    ("There's a bug in this function, find it.", "code_debug"),

    ("Explain how photosynthesis works.", "factual"),
    ("What is the capital of Australia?", "factual"),
    ("Define entropy in thermodynamics.", "factual"),
]


def test_classification_coverage():
    misses = []
    for prompt, expected in CASES:
        got = classify(Task(id="t", input=prompt))
        if got != expected:
            misses.append((prompt, expected, got))
    assert not misses, "mis-routed:\n" + "\n".join(
        f"  [{exp} -> got {got}] {p}" for p, exp, got in misses
    )
