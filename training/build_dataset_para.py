#!/usr/bin/env python3
"""Core dataset for the paraphrase-augmented retrain.

Fixes the two data-level failures of the first 4B run:
  1. NER answers are TYPED (person/organization/location/date), never 'entity:'.
  2. Answer scaffolds rotate (no single canned opener for the judge to see 19x).

Emits two files:
  core.jsonl    - messages-format training examples (generators + benchmarks)
  to_para.jsonl - generator items to paraphrase: {pid, category, instruction,
                  payload, target}; payload (quoted text) is NEVER paraphrased.

    HF_TOKEN=... python build_dataset_para.py --gens-per-cat 2500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_dataset import ABSTAIN_NOTE, SYSTEMS, _example  # noqa: E402
from build_dataset_big import alpaca, gsm8k, mbpp, sst2, xsum  # noqa: E402
from gen_eval import GENERATORS  # noqa: E402

_NER_RE = re.compile(
    r"On (\w+ \d{1,2}, \d{4}), (.+?) of (.+?) announced a partnership "
    r"during a press event held in (.+?)\.")

_MATH_OPEN = ["Computing step by step.", "Working through the arithmetic.",
              "Setting up the calculation.", "Solving directly."]
_LOGIC_OPEN = ["Checking each statement in order.", "Testing every constraint.",
               "Walking through the conditions.", "Evaluating the premises."]


def _split(prompt: str) -> tuple[str, str]:
    """Instruction vs verbatim payload: quoted text is never paraphrased."""
    q = prompt.find('"')
    if q > 20:
        return prompt[:q].rstrip(), " " + prompt[q:]
    return prompt, ""


def gen_items(rng, n_per_cat):
    """Generator items with judge-grade targets. Returns (examples, para_records)."""
    examples, para = [], []
    pid = 0
    for cat in ("sentiment", "factual", "ner", "logic", "math"):
        gen = GENERATORS[cat]
        for i in range(n_per_cat):
            t = gen(rng, i + 10_000)
            g = t["grader"]
            if cat == "sentiment":
                lab = g["expected"]
                target = f"{lab.capitalize()}. The text conveys a {lab} tone overall."
            elif cat == "math":
                v = g["expected"]
                v = int(v) if float(v).is_integer() else v
                target = f"{rng.choice(_MATH_OPEN)}\nAnswer: {v}"
            elif cat == "logic":
                target = f"{rng.choice(_LOGIC_OPEN)}\nAnswer: {g['expected']}"
            elif cat == "ner":
                m = _NER_RE.search(t["prompt"])
                if not m:          # never teach an untyped format
                    continue
                date, person, org, city = m.groups()
                target = (f"person: {person}\norganization: {org}\n"
                          f"location: {city}\ndate: {date}")
            else:  # factual: short keyword truths
                target = ", ".join(g["expected"])
            examples.append(_example(cat, t["prompt"], target))
            instruction, payload = _split(t["prompt"])
            para.append({"pid": pid, "category": cat, "instruction": instruction,
                         "payload": payload, "target": target})
            pid += 1
    return examples, para


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens-per-cat", type=int, default=2500)
    ap.add_argument("--core", default="core.jsonl")
    ap.add_argument("--para-out", default="to_para.jsonl")
    args = ap.parse_args()
    rng = random.Random(31)

    examples, para = gen_items(rng, args.gens_per_cat)
    print(f"generators: {len(examples)} (ner typed, scaffolds rotated)", flush=True)

    for name, fn in (("sst2", lambda: sst2(6000)), ("gsm8k", lambda: gsm8k(6000)),
                     ("xsum", lambda: xsum(6000)), ("alpaca", lambda: alpaca(6000)),
                     ("mbpp", lambda: mbpp(400, 400))):
        try:
            g = fn()
            print(f"{name}: {len(g)}", flush=True)
            examples.extend(g)
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e})", flush=True)

    with open(args.core, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.para_out, "w", encoding="utf-8") as f:
        for r in para:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"core: {len(examples)} -> {args.core}")
    print(f"to paraphrase: {len(para)} -> {args.para_out}")
    assert len(examples) >= 15000, "CORE TOO SMALL - a benchmark source failed"


if __name__ == "__main__":
    main()
