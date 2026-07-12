"""Canonical solutions for classic code_gen asks, verified before commit.

Three gates, all mandatory:
  1. The prompt's requested function name exactly matches a canon entry and
     the prompt mentions the entry's keywords (intent confirmation).
  2. No requirement-modifier appears ("recursive", "without using", "raise",
     "handle", "one line", ...): those change the spec, so canon defers.
  3. The canon code passes its built-in self-tests AND any examples
     extracted from the prompt, executed in an isolated subprocess.

A canon hit costs zero tokens and zero model time; anything unproven falls
through to the local-generate + execution-verify path unchanged.
"""
from __future__ import annotations

import re

from . import code_verify

_C = {
    "factorial": {
        "kw": ("factorial", "n!"),
        "code": ("def factorial(n):\n"
                 "    result = 1\n"
                 "    for i in range(2, n + 1):\n"
                 "        result *= i\n"
                 "    return result"),
        "self": [("factorial(0)", "1"), ("factorial(5)", "120"), ("factorial(7)", "5040")],
    },
    "fibonacci": {
        "kw": ("fibonacci",),
        "code": ("def fibonacci(n):\n"
                 "    a, b = 0, 1\n"
                 "    for _ in range(n):\n"
                 "        a, b = b, a + b\n"
                 "    return a"),
        "self": [("fibonacci(0)", "0"), ("fibonacci(1)", "1"), ("fibonacci(10)", "55")],
    },
    "is_prime": {
        "kw": ("prime",),
        "code": ("def is_prime(n):\n"
                 "    if n < 2:\n"
                 "        return False\n"
                 "    i = 2\n"
                 "    while i * i <= n:\n"
                 "        if n % i == 0:\n"
                 "            return False\n"
                 "        i += 1\n"
                 "    return True"),
        "self": [("is_prime(1)", "False"), ("is_prime(2)", "True"),
                 ("is_prime(97)", "True"), ("is_prime(100)", "False")],
    },
    "is_palindrome": {
        "kw": ("palindrome",),
        "code": ("def is_palindrome(s):\n"
                 "    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
                 "    return cleaned == cleaned[::-1]"),
        "self": [("is_palindrome('racecar')", "True"),
                 ("is_palindrome('hello')", "False"),
                 ("is_palindrome('A man, a plan, a canal: Panama')", "True")],
    },
    "reverse_string": {
        "kw": ("revers",),
        "code": "def reverse_string(s):\n    return s[::-1]",
        "self": [("reverse_string('abc')", "'cba'"), ("reverse_string('')", "''")],
    },
    "reverse_words": {
        "kw": ("revers", "word"),
        "code": "def reverse_words(s):\n    return ' '.join(s.split(' ')[::-1])",
        "self": [("reverse_words('the quick brown fox')", "'fox brown quick the'"),
                 ("reverse_words('hello')", "'hello'")],
    },
    "count_vowels": {
        "kw": ("vowel",),
        "code": ("def count_vowels(s):\n"
                 "    return sum(1 for ch in s.lower() if ch in 'aeiou')"),
        "self": [("count_vowels('hello world')", "3"), ("count_vowels('xyz')", "0")],
    },
    "gcd": {
        "kw": ("greatest common", "gcd"),
        "kw_any": True,
        "code": ("def gcd(a, b):\n"
                 "    while b:\n"
                 "        a, b = b, a % b\n"
                 "    return abs(a)"),
        "self": [("gcd(12, 18)", "6"), ("gcd(7, 13)", "1"), ("gcd(0, 5)", "5")],
    },
    "is_anagram": {
        "kw": ("anagram",),
        "code": ("def is_anagram(a, b):\n"
                 "    return sorted(a.lower()) == sorted(b.lower())"),
        "self": [("is_anagram('listen', 'silent')", "True"),
                 ("is_anagram('hello', 'world')", "False")],
    },
    "remove_duplicates": {
        "kw": ("duplicate",),
        "code": ("def remove_duplicates(items):\n"
                 "    seen = set()\n"
                 "    out = []\n"
                 "    for x in items:\n"
                 "        if x not in seen:\n"
                 "            seen.add(x)\n"
                 "            out.append(x)\n"
                 "    return out"),
        "self": [("remove_duplicates([1, 2, 2, 3, 1])", "[1, 2, 3]"),
                 ("remove_duplicates([])", "[]")],
    },
    "second_largest": {
        "kw": ("second largest", "second-largest"),
        "kw_any": True,
        "code": ("def second_largest(nums):\n"
                 "    uniq = sorted(set(nums))\n"
                 "    return uniq[-2] if len(uniq) >= 2 else None"),
        "self": [("second_largest([4, 1, 7, 7, 3])", "4"),
                 ("second_largest([5, 9])", "5")],
    },
    "celsius_to_fahrenheit": {
        "kw": ("celsius", "fahrenheit"),
        "code": ("def celsius_to_fahrenheit(c):\n"
                 "    return c * 9 / 5 + 32"),
        "self": [("celsius_to_fahrenheit(0)", "32.0"), ("celsius_to_fahrenheit(100)", "212.0")],
    },
    "fahrenheit_to_celsius": {
        "kw": ("celsius", "fahrenheit"),
        "code": ("def fahrenheit_to_celsius(f):\n"
                 "    return (f - 32) * 5 / 9"),
        "self": [("fahrenheit_to_celsius(32)", "0.0"), ("fahrenheit_to_celsius(212)", "100.0")],
    },
    "count_words": {
        "kw": ("count", "word"),
        "code": "def count_words(s):\n    return len(s.split())",
        "self": [("count_words('one two three')", "3"), ("count_words('')", "0")],
    },
    "sum_list": {
        "kw": ("sum",),
        "code": ("def sum_list(nums):\n"
                 "    total = 0\n"
                 "    for x in nums:\n"
                 "        total += x\n"
                 "    return total"),
        "self": [("sum_list([1, 2, 3])", "6"), ("sum_list([])", "0")],
    },
    "find_max": {
        "kw": ("max", "largest"),
        "kw_any": True,
        "code": ("def find_max(nums):\n"
                 "    biggest = nums[0]\n"
                 "    for x in nums[1:]:\n"
                 "        if x > biggest:\n"
                 "            biggest = x\n"
                 "    return biggest"),
        "self": [("find_max([3, 9, 2])", "9"), ("find_max([-5, -1])", "-1")],
    },
    "char_frequency": {
        "kw": ("frequenc", "character"),
        "code": ("def char_frequency(s):\n"
                 "    freq = {}\n"
                 "    for ch in s:\n"
                 "        freq[ch] = freq.get(ch, 0) + 1\n"
                 "    return freq"),
        "self": [("char_frequency('aab')", "{'a': 2, 'b': 1}")],
    },
    "capitalize_words": {
        "kw": ("capitaliz",),
        "code": ("def capitalize_words(s):\n"
                 "    return ' '.join(w.capitalize() for w in s.split(' '))"),
        "self": [("capitalize_words('hello world')", "'Hello World'")],
    },
}

# Requirement modifiers that change the spec beyond the vanilla canon: the
# canonical implementation might not satisfy them, so canon must defer.
_MODIFIERS = re.compile(
    r"(?i)\b(recursi|without\s+using|one[- ]?lin|lambda|memoiz|generator|"
    r"raise|except|error|handle|edge\s+case|type\s+hint|docstring|"
    r"o\(1\)|o\(n\s*log|in[- ]?place|thread|async|class\b|unit\s+test)")

_FUNC = re.compile(r"(?i)function\s+(?:named\s+|called\s+)?`?([A-Za-z_]\w*)`?")


def try_canon(prompt: str) -> str | None:
    """Verified canonical code for a vanilla classic ask, else None."""
    m = _FUNC.search(prompt)
    if not m:
        return None
    name = m.group(1)
    entry = _C.get(name)
    if entry is None:
        return None
    if _MODIFIERS.search(prompt):
        return None
    low = prompt.lower()
    kws = entry["kw"]
    if entry.get("kw_any"):
        if not any(k in low for k in kws):
            return None
    elif not all(k in low for k in kws):
        return None

    tests = list(entry["self"])
    tests.extend(code_verify.extract_tests(prompt, name))
    if not code_verify._run(entry["code"], tests):
        return None
    return f"```python\n{entry['code']}\n```"
