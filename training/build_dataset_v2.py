#!/usr/bin/env python3
"""Full 8-category SFT dataset for the local model (v2, full-local ambition).

Sources (train splits only — never evaluation data):
  GSM8K train        -> math: brief steps + 'Answer: <value>'
  SST-2 train        -> sentiment: label-first + justification
  XSum train         -> summarization: gold one-sentence summaries
  MBPP train         -> code_gen: spec -> canonical solution in one fenced block
  MBPP train (mutated)-> code_debug: buggy code -> bug sentence + fixed block
  our generators     -> every category's contract format (computed answers)
  abstention (~12%)  -> UNSURE on measured-hallucination populations

    python build_dataset_v2.py --out sft.jsonl --target 9000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import (ABSTAIN_NOTE, SYSTEMS, _example, abstention,  # noqa: E402
                           from_generators, from_gsm8k, from_sst2)

SYSTEMS.setdefault("code_gen", "English only. Be concise; no preamble. Output only the complete, correct, self-contained code in one fenced block.")
SYSTEMS.setdefault("code_debug", "English only. Be concise; no preamble. Name the bug in one sentence, then give the complete corrected code in one fenced block.")


def from_xsum(n=900):
    from fetch_bench import rows
    out = []
    for r in rows("EdinburghNLP/xsum", "default", "train", n * 2):
        doc = " ".join(r["document"].split())
        summ = r["summary"].strip()
        if len(doc.split()) < 60 or len(summ.split()) < 6:
            continue
        doc = " ".join(doc.split()[:220])
        out.append(_example("summarization",
                            f"Summarize the following text in one sentence: {doc}", summ))
        if len(out) >= n:
            break
    return out


_MUTATIONS = [("==", "!="), ("<", "<="), ("+", "-"), ("return ", "return not "),
              ("range(", "range(1, "), ("and", "or")]


def from_mbpp(n_gen=700, n_dbg=700):
    from fetch_bench import rows
    gen, dbg = [], []
    for r in rows("google-research-datasets/mbpp", "full", "train", (n_gen + n_dbg) * 2):
        code = (r.get("code") or "").strip()
        text = (r.get("text") or "").strip()
        if not code or not text or len(code) > 500:
            continue
        m = re.search(r"def\s+(\w+)", code)
        if not m:
            continue
        fenced = f"```python\n{code}\n```"
        if len(gen) < n_gen:
            gen.append(_example("code_gen",
                                f"{text} Write it as a Python function named {m.group(1)}.",
                                fenced))
            continue
        if len(dbg) < n_dbg:
            broken = code
            for a, b in _MUTATIONS:
                if a in broken:
                    broken = broken.replace(a, b, 1)
                    break
            if broken == code:
                continue
            dbg.append(_example(
                "code_debug",
                ("This Python function has a bug. Identify it and give the complete "
                 f"corrected function:\n```python\n{broken}\n```"),
                f"The bug is an incorrect operator or boundary in the highlighted logic.\n{fenced}"))
        if len(gen) >= n_gen and len(dbg) >= n_dbg:
            break
    return gen + dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--target", type=int, default=9000)
    args = ap.parse_args()
    rng = random.Random(23)

    data = from_generators(rng, n_per_cat=260)
    for name, fn in (("sst2", lambda: from_sst2(1400)),
                     ("gsm8k", lambda: from_gsm8k(1600)),
                     ("xsum", from_xsum),
                     ("mbpp", from_mbpp)):
        try:
            got = fn()
            print(f"{name}: {len(got)}")
            data.extend(got)
        except Exception as exc:
            print(f"{name}: UNAVAILABLE ({type(exc).__name__}) — mix will be weaker")
    data.extend(abstention(rng, n=max(400, int(len(data) * 0.14))))
    rng.shuffle(data)
    data = data[: args.target]

    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in data:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    unsure = sum(1 for ex in data if ex["messages"][2]["content"] == "UNSURE")
    print(f"wrote {len(data)} examples -> {args.out}  (UNSURE: {unsure} = {unsure/len(data):.0%})")


if __name__ == "__main__":
    main()
