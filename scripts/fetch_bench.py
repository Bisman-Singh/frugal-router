#!/usr/bin/env python3
"""Build a diverse eval set from REAL public benchmarks.

The generated eval set shares its author with the pipeline, so it cannot catch
distribution surprises. This pulls real tasks via the HuggingFace
datasets-server REST API (public, no auth) and adapts them to our grader
format:

  math          GSM8K            numeric final answer
  sentiment     SST-2            real movie-review fragments, binary labels
  ner           CoNLL-2003       real newswire, gold entity spans
  summarization XSum             real articles with one-sentence references
  factual       NQ-Open          real questions with answer aliases
  code_gen      MBPP             specs with ready-made assert tests
  code_debug    MBPP (mutated)   real solutions with injected bugs
  logic         generated        (no clean public source fits the grader)

Usage: python scripts/fetch_bench.py --per-cat 20 --out data/eval_bench.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"


def rows(dataset: str, config: str, split: str, length: int, offset: int = 0):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                "split": split, "offset": offset, "length": length})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "eval-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [r["row"] for r in json.load(resp)["rows"]]


def fetch_gsm8k(n):
    out = []
    for i, r in enumerate(rows("openai/gsm8k", "main", "test", n * 2)):
        m = re.search(r"####\s*([-\d,\.]+)", r["answer"])
        if not m:
            continue
        out.append({"id": f"bench-math-{i}", "category": "math",
                    "prompt": r["question"].strip(),
                    "grader": {"type": "numeric",
                               "expected": float(m.group(1).replace(",", ""))}})
        if len(out) >= n:
            break
    return out


def fetch_sst2(n):
    out = []
    for i, r in enumerate(rows("stanfordnlp/sst2", "default", "validation", n * 2)):
        text = r["sentence"].strip()
        if len(text) < 25:
            continue
        label = "positive" if r["label"] == 1 else "negative"
        out.append({"id": f"bench-sent-{i}", "category": "sentiment",
                    "prompt": f"Classify the sentiment of this review fragment and justify briefly: \"{text}\"",
                    "grader": {"type": "label", "expected": label,
                               "others": [l for l in ("positive", "negative") if l != label]}})
        if len(out) >= n:
            break
    return out


_CONLL_TAGS = {1: "person", 2: "person", 3: "organization", 4: "organization",
               5: "location", 6: "location"}


def fetch_conll(n):
    out = []
    for i, r in enumerate(rows("eriktks/conll2003", "conll2003", "test", n * 6)):
        toks, tags = r["tokens"], r["ner_tags"]
        ents, cur = [], []
        for t, g in zip(toks, tags):
            if g in (1, 3, 5):          # B- tags we care about
                if cur:
                    ents.append(" ".join(cur))
                cur = [t]
            elif g in (2, 4, 6) and cur:
                cur.append(t)
            else:
                if cur:
                    ents.append(" ".join(cur))
                cur = []
        if cur:
            ents.append(" ".join(cur))
        ents = [e for e in dict.fromkeys(ents) if len(e) > 2]
        if len(ents) < 2 or len(toks) < 12:
            continue
        sent = " ".join(toks)
        out.append({"id": f"bench-ner-{i}", "category": "ner",
                    "prompt": ("Extract the named entities (person, organization, "
                               f"location, date) from this text, one per line as 'label: value': \"{sent}\""),
                    "grader": {"type": "contains_all", "expected": ents[:4]}})
        if len(out) >= n:
            break
    return out


def fetch_xsum(n):
    out = []
    for i, r in enumerate(rows("EdinburghNLP/xsum", "default", "validation", n * 2)):
        doc = " ".join(r["document"].split())
        if len(doc.split()) < 60:
            continue
        doc = " ".join(doc.split()[:220])
        ref_words = [w.strip(".,'\"") for w in r["summary"].split() if len(w) > 5][:4]
        if len(ref_words) < 2:
            continue
        out.append({"id": f"bench-summ-{i}", "category": "summarization",
                    "prompt": f"Summarize the following text in one sentence: {doc}",
                    "grader": {"type": "summary", "max_sentences": 1,
                               "must_mention_any": ref_words}})
        if len(out) >= n:
            break
    return out


def fetch_nq(n):
    out = []
    for i, r in enumerate(rows("google-research-datasets/nq_open", "nq_open",
                               "validation", n * 2)):
        answers = [a for a in r["answer"] if len(a) > 1]
        if not answers:
            continue
        out.append({"id": f"bench-fact-{i}", "category": "factual",
                    "prompt": r["question"].strip().rstrip("?") + "?",
                    "grader": {"type": "contains_any", "expected": answers}})
        if len(out) >= n:
            break
    return out


def fetch_mbpp(n):
    gen, dbg = [], []
    mutations = [("==", "!="), ("<", "<="), ("+", "-"), ("return ", "return not "),
                 ("range(", "range(1, ")]
    for i, r in enumerate(rows("google-research-datasets/mbpp", "full", "test", n * 3)):
        tests = r.get("test_list") or []
        code = r.get("code") or ""
        if not tests or not code or len(code) > 600:
            continue
        checks = [(t.replace("assert ", ""), True) for t in tests[:3]
                  if t.startswith("assert ") and "==" in t]
        # our code grader compares call == expected; reuse assert strings directly
        tests_spec = []
        for t in tests[:3]:
            m = re.match(r"assert\s+(.+?)\s*==\s*(.+)$", t.strip())
            if m:
                try:
                    tests_spec.append((m.group(1), eval(m.group(2), {}, {})))  # noqa: S307
                except Exception:
                    pass
        if len(tests_spec) < 2:
            continue
        fname = re.search(r"def\s+(\w+)", code)
        if not fname:
            continue
        if len(gen) < n:
            gen.append({"id": f"bench-cgen-{i}", "category": "code_gen",
                        "prompt": f"{r['text'].strip()} Write it as a Python function named {fname.group(1)}.",
                        "grader": {"type": "code", "func": fname.group(1),
                                   "tests": tests_spec}})
        elif len(dbg) < n:
            broken = code
            for a, b in mutations:
                if a in broken:
                    broken = broken.replace(a, b, 1)
                    break
            if broken == code:
                continue
            dbg.append({"id": f"bench-cdbg-{i}", "category": "code_debug",
                        "prompt": ("This Python function has a bug. Identify it and give the "
                                   f"complete corrected function:\n```python\n{broken}\n```"),
                        "grader": {"type": "code", "func": fname.group(1),
                                   "tests": tests_spec}})
        if len(gen) >= n and len(dbg) >= n:
            break
    return gen + dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--out", default="data/eval_bench.jsonl")
    args = ap.parse_args()

    sources = [("gsm8k", fetch_gsm8k), ("sst2", fetch_sst2), ("conll", fetch_conll),
               ("xsum", fetch_xsum), ("nq_open", fetch_nq), ("mbpp", fetch_mbpp)]
    tasks = []
    for name, fn in sources:
        try:
            got = fn(args.per_cat)
            print(f"  {name}: {len(got)} tasks")
            tasks.extend(got)
        except Exception as exc:
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc}) — skipped")

    random.Random(3).shuffle(tasks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    cats = {}
    for t in tasks:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    print(f"wrote {len(tasks)} -> {out}  {cats}")


if __name__ == "__main__":
    main()
