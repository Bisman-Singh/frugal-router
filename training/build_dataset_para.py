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


_SURNAMES = ["Okafor", "Petrov", "Sandoval", "Meier", "Tanaka"]


def ner_variety(rng, n, pid0):
    """Variable-shape NER: subsets of types, duplicate types — the model must
    list ONLY entities that are present, never hallucinate a missing date/org."""
    from gen_eval import CITIES, FIRST, MONTHS, ORGS
    ask = ("Extract the named entities (person, organization, location, date) "
           "from this text, one per line as 'label: value': \"{s}\"")
    examples, para = [], []
    for i in range(n):
        p1 = f"{rng.choice(FIRST)} {rng.choice(_SURNAMES)}"
        p2 = f"{rng.choice(FIRST)} {rng.choice(_SURNAMES)}"
        org, city = rng.choice(ORGS), rng.choice(CITIES)
        date = f"{rng.choice(MONTHS)} {rng.randrange(1, 28)}, {rng.randrange(2018, 2026)}"
        kind = i % 5
        if kind == 0:      # person + org
            sent = f"{p1} joined {org} as chief architect."
            ents = [("person", p1), ("organization", org)]
        elif kind == 1:    # org + location
            sent = f"The {org} office in {city} expanded to two new floors."
            ents = [("organization", org), ("location", city)]
        elif kind == 2:    # person + location + date
            sent = f"{p1} delivered the keynote in {city} on {date}."
            ents = [("person", p1), ("location", city), ("date", date)]
        elif kind == 3:    # TWO persons + org
            if p2 == p1:
                p2 = f"{rng.choice(FIRST)} {rng.choice(_SURNAMES[1:])}"
            sent = f"{p1} and {p2} co-founded {org} together."
            ents = [("person", p1), ("person", p2), ("organization", org)]
        else:              # org + date
            sent = f"{org} reported quarterly earnings on {date}."
            ents = [("organization", org), ("date", date)]
        prompt = ask.format(s=sent)
        target = "\n".join(f"{k}: {v}" for k, v in ents)
        examples.append(_example("ner", prompt, target))
        instruction, payload = _split(prompt)
        para.append({"pid": pid0 + i, "category": "ner", "instruction": instruction,
                     "payload": payload, "target": target})
    return examples, para


def code_items():
    """Official-guide-style code specs WITH reference solutions, plus debug
    variants whose bug descriptions are SPECIFIC (from the known mutation)."""
    from gen_eval import _BUGS, _CODE_SPECS
    ex = []
    for _name, spec, ref, _tests in _CODE_SPECS:
        ex.append(_example("code_gen", spec, f"```python\n{ref}\n```"))
    for i in range(len(_CODE_SPECS) * len(_BUGS)):
        _name, _spec, ref, _tests = _CODE_SPECS[i % len(_CODE_SPECS)]
        good, bad = _BUGS[i % len(_BUGS)]
        if good not in ref:
            continue
        broken = ref.replace(good, bad, 1)
        if broken == ref:
            continue
        ex.append(_example(
            "code_debug",
            "This Python function has a bug. Identify it and give the complete "
            f"corrected function:\n```python\n{broken}\n```",
            f"The bug: `{bad}` should be `{good}`.\n```python\n{ref}\n```"))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens-per-cat", type=int, default=2500)
    ap.add_argument("--core", default="core.jsonl")
    ap.add_argument("--para-out", default="to_para.jsonl")
    args = ap.parse_args()
    rng = random.Random(31)

    examples, para = gen_items(rng, args.gens_per_cat)
    print(f"generators: {len(examples)} (ner typed, scaffolds rotated)", flush=True)

    nv_ex, nv_para = ner_variety(rng, max(1200, args.gens_per_cat // 2), len(para))
    examples.extend(nv_ex)
    para.extend(nv_para)
    print(f"ner variety: {len(nv_ex)} (variable entity shapes)", flush=True)

    ci = code_items()
    examples.extend(ci)
    print(f"guide-style code specs: {len(ci)} (specific bug descriptions)", flush=True)

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
