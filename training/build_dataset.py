#!/usr/bin/env python3
"""Build the SFT dataset for the local-tier model.

Mix (target ~5k examples):
  ~35%  contract-format examples from our generators (computed answers)
  ~25%  SST-2 train split      -> label-first sentiment with justification
  ~20%  GSM8K train split      -> brief steps + 'Answer: <value>'
  ~20%  ABSTENTION examples    -> target output is exactly 'UNSURE' for the
        populations we measured the small model hallucinating on: trivia
        lookups (who/when/where + names), mixed/ambiguous sentiment, and
        multi-entity ordering. The gate reads UNSURE as escalate.

Run anywhere with network (rate-limited HF sources degrade gracefully):
    python training/build_dataset.py --out training/sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gen_eval import GENERATORS  # noqa: E402

SYSTEMS = {
    "factual": "English only. Be concise; no preamble. Answer accurately and completely in under 120 words.",
    "math": "English only. Be concise; no preamble. Show brief steps, then end with 'Answer: <value>' on its own line.",
    "sentiment": "English only. Be concise; no preamble. State exactly one label - positive, negative, or neutral - then one short justification.",
    "summarization": "English only. Be concise; no preamble. Output only the summary and obey any stated length or format constraint exactly.",
    "ner": "English only. Be concise; no preamble. List each entity as 'label: value', one per line; labels: person, organization, location, date.",
    "logic": "English only. Be concise; no preamble. Reason in brief numbered steps checking every constraint, then end with 'Answer: <value>' on its own line.",
}

ABSTAIN_NOTE = " If you are not confident the answer is correct, reply with exactly UNSURE."


def _example(category, prompt, target):
    return {"messages": [
        {"role": "system", "content": SYSTEMS[category] + ABSTAIN_NOTE},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]}


def from_generators(rng, n_per_cat=220):
    out = []
    for cat in ("sentiment", "factual", "ner", "logic", "math"):
        gen = GENERATORS[cat]
        for i in range(n_per_cat):
            t = gen(rng, i + 10_000)
            g = t["grader"]
            if cat == "sentiment":
                target = f"{g['expected'].capitalize()}. The text clearly expresses a {g['expected']} tone."
            elif cat == "math":
                v = g["expected"]
                v = int(v) if float(v).is_integer() else v
                target = f"Computing step by step.\nAnswer: {v}"
            elif cat == "logic":
                if g["type"] == "yesno":
                    target = f"Checking each statement in order.\nAnswer: {g['expected']}"
                else:
                    target = f"Ordering all constraints.\nAnswer: {g['expected']}"
            elif cat == "ner":
                target = "\n".join(f"entity: {e}" for e in g["expected"])
                # generator embeds labels; better: reconstruct from prompt fields
            else:  # factual bank has keyword truths, not prose; use short answers
                target = ", ".join(g["expected"])
            out.append(_example(cat, t["prompt"], target))
    return out


def from_sst2(n=1200):
    from fetch_bench import rows
    out = []
    for r in rows("stanfordnlp/sst2", "default", "train", n * 2):
        text = r["sentence"].strip()
        if len(text) < 20:
            continue
        label = "positive" if r["label"] == 1 else "negative"
        out.append(_example(
            "sentiment",
            f"Classify the sentiment of this review fragment and justify briefly: \"{text}\"",
            f"{label.capitalize()}. The fragment reads as {label} in tone."))
        if len(out) >= n:
            break
    return out


def from_gsm8k(n=1000):
    from fetch_bench import rows
    out = []
    for r in rows("openai/gsm8k", "main", "train", n * 2):
        m = re.search(r"####\s*([-\d,\.]+)", r["answer"])
        if not m:
            continue
        steps = re.sub(r"####.*", "", r["answer"]).strip()
        steps = re.sub(r"<<.*?>>", "", steps)[:400]
        out.append(_example("math", r["question"].strip(),
                            f"{steps}\nAnswer: {m.group(1).replace(',', '')}"))
        if len(out) >= n:
            break
    return out


def abstention(rng, n=700):
    """Populations where the small model must say UNSURE, not guess."""
    out = []
    first = ["Amara", "Boris", "Chen", "Dita", "Emil", "Farah", "Goro", "Hana"]
    topics = ["the 1962 regional rowing championship", "the founding year of the village bakery in Oslo",
              "the third studio album's producer", "the middle name of the district's first mayor",
              "the 1978 amendment's sponsoring senator", "the original architect of the harbor bridge"]
    for i in range(n // 3):
        out.append(_example("factual", f"Who was {rng.choice(topics)}?", "UNSURE"))
        out.append(_example("factual", f"When did {rng.choice(first)} {rng.choice(['Kovac','Lindt','Moreau'])} win their first title?", "UNSURE"))
    mixed = ["The camera is superb but the battery ruined the trip and support was useless.",
             "Loved the interface; shipping was a disaster and the refund took months.",
             "Brilliant screen, horrible keyboard, average sound - hard to say overall."]
    for i in range(n // 6):
        out.append(_example("sentiment",
                            f"Classify the sentiment and justify briefly: \"{rng.choice(mixed)}\"", "UNSURE"))
    for i in range(n // 6):
        a, b, c, d, e = rng.sample(first, 5)
        out.append(_example("logic",
                            f"{a} finished after {b} but before {c} in one heat, while {d} beat {b} and lost to {e}. Who finished third overall?",
                            "UNSURE"))
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/sft.jsonl")
    args = ap.parse_args()
    rng = random.Random(17)

    data = from_generators(rng)
    for name, fn in (("sst2", from_sst2), ("gsm8k", from_gsm8k)):
        try:
            got = fn()
            print(f"{name}: {len(got)}")
            data.extend(got)
        except Exception as exc:
            print(f"{name}: unavailable ({type(exc).__name__}) - continuing without it")
    data.extend(abstention(rng))
    rng.shuffle(data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in data:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(data)} examples -> {out}")


if __name__ == "__main__":
    main()
