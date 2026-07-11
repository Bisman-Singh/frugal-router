#!/usr/bin/env python3
"""Assemble the final paraphrase-augmented SFT set.

core (originals, judge formats) + paraphrase variants (same verified target)
+ distilled gold + abstention (ABSTAIN_FRAC, default 8% for hybrid gates).

    ABSTAIN_FRAC=0.08 python assemble_para.py --target 90000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import ABSTAIN_NOTE, SYSTEMS, abstention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="core.jsonl")
    ap.add_argument("--para-src", default="to_para.jsonl")
    ap.add_argument("--para", default="para.jsonl")
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--target", type=int, default=90000)
    args = ap.parse_args()
    rng = random.Random(29)

    data = [json.loads(l) for l in open(args.core, encoding="utf-8")]
    print(f"core: {len(data)}")

    src = {r["pid"]: r for r in
           (json.loads(l) for l in open(args.para_src, encoding="utf-8"))}
    n_var = 0
    for line in open(args.para, encoding="utf-8"):
        rec = json.loads(line)
        base = src.get(rec["pid"])
        if not base:
            continue
        system = SYSTEMS[base["category"]] + ABSTAIN_NOTE
        for v in rec["variants"]:
            prompt = (v.rstrip() + base["payload"]) if base["payload"] else v
            data.append({"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": base["target"]}]})
            n_var += 1
    print(f"paraphrase variants: {n_var}")

    dp = Path(__file__).with_name("distill.jsonl")
    if dp.exists():
        gold = [json.loads(l) for l in dp.read_text().splitlines() if l.strip()]
        data.extend(gold)
        print(f"distilled gold: {len(gold)}")

    frac = float(os.environ.get("ABSTAIN_FRAC", "0.08"))
    n_abs = max(600, int(len(data) * frac))
    data.extend(abstention(rng, n=n_abs))
    print(f"abstention: {n_abs} ({frac:.0%})")

    rng.shuffle(data)
    data = data[: args.target]
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    unsure = sum(1 for ex in data if ex["messages"][2]["content"] == "UNSURE")
    sysc = Counter(ex["messages"][0]["content"][:40] for ex in data)
    print(f"wrote {len(data)} -> {args.out}  (UNSURE {unsure} = {unsure / max(1, len(data)):.0%})")
    for k, v in sysc.most_common():
        print(f"  {v:>6}  {k}...")
    assert len(data) >= 20000, f"DATASET TOO SMALL ({len(data)})"


if __name__ == "__main__":
    main()
