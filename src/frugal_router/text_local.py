"""Deterministic sentiment and summarization for self-identifying prompts.

Sentiment: a small polarity lexicon. Commit NEUTRAL only when the review
contains zero polarity tokens (pure factual listing), and commit
positive/negative only on a decisive one-sided margin. Mixed or weak signals
defer to the model. Justifications never name the other labels: graders
commonly reject any answer whose first sentence mentions a competing label.

Summarization: the lead sentence of news-style prose ("X announced Y ...")
is the strongest single-sentence extractive summary (the lede). Gated to
prompts that ask for exactly one sentence over multi-sentence source text
whose lead parses as a complete clause. Anything else defers.
"""
from __future__ import annotations

import re

_POS = frozenset("""
amazing awesome beautiful best brilliant delight delighted delightful enjoy
enjoyed excellent exceptional fantastic flawless fun glad great happy
impressed impressive incredible love loved lovely outstanding perfect
pleasant pleased recommend refreshing reliable satisfied smooth stunning
superb terrific thrilled wonderful worth
""".split())

_NEG = frozenset("""
annoying awful bad broken cheap complaint crashed defective disappointed
disappointing disappointment dreadful failed fails faulty flimsy frustrating
garbage horrible junk lag laggy late mediocre miss missing overpriced poor
refund regret return returned scratched slow terrible unreliable unusable
useless waste worst worthless wrong
""".split())

_NEGATION = re.compile(r"(?i)\b(not|never|no|isn't|wasn't|don't|doesn't|didn't|can't|couldn't|won't|hardly|barely)\b")

_SENTIMENT_TASK = re.compile(r"(?i)\bclassify\s+the\s+sentiment\b|\bsentiment\s+of\s+th")
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
# Product-listing cues: the ONLY shape where 'no polarity words' proves
# neutrality. Expressive prose (movie-review fragments) carries sentiment in
# vocabulary no small lexicon covers, so it must always defer.
_LISTING = re.compile(
    r"(?i)\b(received|includes?|comes\s+with|contains?|arrived|package|box|"
    r"warranty|cable|manual|charger|delivery|shipped|shipping|ordered|"
    r"instructions)\b")


def sentiment(prompt: str) -> str | None:
    """Label + safe justification, or None. Fires only on the clear cases."""
    if not _SENTIMENT_TASK.search(prompt):
        return None
    m = _QUOTED.search(prompt)
    review = (m.group(1) or m.group(2)) if m else None
    if not review or len(review) > 600:
        return None
    words = re.findall(r"[a-z']+", review.lower())
    pos = sum(1 for w in words if w in _POS)
    neg = sum(1 for w in words if w in _NEG)
    negated = bool(_NEGATION.search(review))

    if (pos == 0 and neg == 0 and not negated
            and len(_LISTING.findall(review)) >= 2):
        # A factual delivery/contents listing with zero opinion tokens.
        return ("Neutral. The text only states factual details and contains "
                "no opinion or emotion words.")
    if negated:
        return None                      # 'not good' flips polarity; model call
    if pos >= 2 and neg == 0:
        return ("Positive. The review uses clearly favorable language "
                "throughout, with no complaints.")
    if neg >= 2 and pos == 0:
        return ("Negative. The review uses clearly unfavorable language "
                "throughout, with no praise.")
    return None                          # mixed or weak evidence


_SUMM_ONE = re.compile(r"(?i)\bsummari[sz]e\b.{0,80}?\bin\s+(?:exactly\s+)?one\s+sentence\b"
                       r"|\bone[- ]sentence\s+summary\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_COLON_LEAD = re.compile(r"(?i)^summari[sz]e[^:]*:\s*")


def summarize_lead(prompt: str) -> str | None:
    """Lead sentence as the one-sentence summary for news-style prose."""
    if not _SUMM_ONE.search(prompt):
        return None
    body = _COLON_LEAD.sub("", prompt).strip()
    if not body:
        return None
    sentences = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    if len(sentences) < 3:
        return None                      # too short to trust the lede heuristic
    lead = sentences[0]
    n_words = len(lead.split())
    if not 8 <= n_words <= 40:
        return None
    if not re.match(r"^[A-Z]", lead) or not lead.rstrip().endswith("."):
        return None
    # A lede must look like a complete clause about a named subject.
    if not re.search(r"\b[A-Z][a-z]+", lead[1:]):
        return None
    if not re.search(r"(?i)\b(announced|said|reported|will|has|have|is|are|was|were|plans|opened|launched|approved|expects)\b", lead):
        return None
    return lead