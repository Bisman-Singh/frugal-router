#!/usr/bin/env python3
"""Big verified dataset via the HuggingFace `datasets` library (full splits).

Unlike the REST-API fetcher (rate-limited, small), this pulls large train
splits reliably with an HF token: SST-2, GSM8K, XSum, MBPP, Alpaca — plus our
computed-answer generators and the grader-verified distilled gold. Targets
~60-90k examples. Every non-generator label is the dataset's own ground truth.

    HF_TOKEN=... python build_dataset_big.py --out sft.jsonl --target 80000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import ABSTAIN_NOTE, SYSTEMS, _example, abstention, from_generators  # noqa

SYSTEMS.setdefault("code_gen", "English only. Be concise; no preamble. Output only the complete, correct, self-contained code in one fenced block.")
SYSTEMS.setdefault("code_debug", "English only. Be concise; no preamble. Name the bug in one sentence, then give the complete corrected code in one fenced block.")

_MUT = [("==", "!="), ("<", "<="), ("+", "-"), ("return ", "return not "), ("range(", "range(1, ")]


def _ld(name, config=None, split="train"):
    from datasets import load_dataset
    try:
        return load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
    except TypeError:
        return load_dataset(name, config, split=split, streaming=True)


def sst2(n):
    out = []
    for r in _ld("stanfordnlp/sst2"):
        t = (r.get("sentence") or "").strip()
        if len(t) < 20:
            continue
        lab = "positive" if r["label"] == 1 else "negative"
        out.append(_example("sentiment",
            f"Classify the sentiment of this review fragment and justify briefly: \"{t}\"",
            f"{lab.capitalize()}. The fragment reads as {lab} in tone."))
        if len(out) >= n:
            break
    return out


def gsm8k(n):
    out = []
    for r in _ld("openai/gsm8k", "main"):
        m = re.search(r"####\s*([-\d,\.]+)", r["answer"])
        if not m:
            continue
        steps = re.sub(r"<<.*?>>", "", re.sub(r"####.*", "", r["answer"])).strip()[:400]
        out.append(_example("math", r["question"].strip(),
                            f"{steps}\nAnswer: {m.group(1).replace(',', '')}"))
        if len(out) >= n:
            break
    return out


def xsum(n):
    out = []
    for r in _ld("EdinburghNLP/xsum"):
        doc, summ = " ".join(r["document"].split()), r["summary"].strip()
        if len(doc.split()) < 60 or len(summ.split()) < 6:
            continue
        out.append(_example("summarization",
            f"Summarize the following text in one sentence: {' '.join(doc.split()[:220])}", summ))
        if len(out) >= n:
            break
    return out


def alpaca(n):
    out = []
    for r in _ld("yahma/alpaca-cleaned"):
        ins, inp, tgt = (r.get("instruction") or "").strip(), (r.get("input") or "").strip(), (r.get("output") or "").strip()
        if not ins or not tgt or len(tgt) > 600 or (inp and len(inp) > 400):
            continue
        prompt = ins if not inp else f"{ins}\n\n{inp}"
        out.append({"messages": [
            {"role": "system", "content": "English only. Be concise; no preamble." + ABSTAIN_NOTE},
            {"role": "user", "content": prompt}, {"role": "assistant", "content": tgt}]})
        if len(out) >= n:
            break
    return out


def mbpp(n_gen, n_dbg):
    gen, dbg = [], []
    for r in _ld("google-research-datasets/mbpp", "full"):
        code, text = (r.get("code") or "").strip(), (r.get("text") or "").strip()
        m = re.search(r"def\s+(\w+)", code)
        if not code or not text or len(code) > 500 or not m:
            continue
        fenced = f"```python\n{code}\n```"
        if len(gen) < n_gen:
            gen.append(_example("code_gen", f"{text} Write it as a Python function named {m.group(1)}.", fenced))
        elif len(dbg) < n_dbg:
            broken = code
            for a, b in _MUT:
                if a in broken:
                    broken = broken.replace(a, b, 1); break
            if broken != code:
                dbg.append(_example("code_debug",
                    f"This Python function has a bug. Identify it and give the complete corrected function:\n```python\n{broken}\n```",
                    f"The bug is an incorrect operator or boundary in the logic.\n{fenced}"))
        if len(gen) >= n_gen and len(dbg) >= n_dbg:
            break
    return gen + dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--target", type=int, default=80000)
    args = ap.parse_args()
    rng = random.Random(29)

    data = from_generators(rng, n_per_cat=5000)   # 5 cats -> 25k computed-answer FLOOR
    for name, fn in (("sst2", lambda: sst2(18000)), ("gsm8k", lambda: gsm8k(7000)),
                     ("xsum", lambda: xsum(15000)), ("alpaca", lambda: alpaca(18000)),
                     ("mbpp", lambda: mbpp(400, 400))):
        try:
            g = fn(); print(f"{name}: {len(g)}", flush=True); data.extend(g)
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e})", flush=True)
    # merge grader-verified distilled gold if present next to this script
    dp = Path(__file__).with_name("distill.jsonl")
    if dp.exists():
        gold = [json.loads(l) for l in dp.read_text().splitlines() if l.strip()]
        data.extend(gold); print(f"distilled gold: {len(gold)}")
    data.extend(abstention(rng, n=max(1500, int(len(data) * 0.12))))
    rng.shuffle(data)
    data = data[: args.target]

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    from collections import Counter
    cats = Counter()
    for ex in data:
        sysm = ex["messages"][0]["content"]
        for c, key in [("code","fenced"),("code","corrected"),("math","Answer:"),("sentiment","sentiment label"),
                       ("ner","label: value"),("summarization","summary"),("logic","constraint"),("factual","120 words"),("general","Be concise; no preamble. If")]:
            if key in sysm: cats[c]+=1; break
        else: cats["?"]+=1
    unsure = sum(1 for ex in data if ex["messages"][2]["content"] == "UNSURE")
    print(f"wrote {len(data)} -> {args.out}  (UNSURE {unsure} = {unsure/max(1,len(data)):.0%})")
    print(f"category mix: {dict(cats)}")
    assert len(data) >= 20000, f"DATASET TOO SMALL ({len(data)}) - a source failed; not training on this"


if __name__ == "__main__":
    main()
