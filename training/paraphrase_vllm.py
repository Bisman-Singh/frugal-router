#!/usr/bin/env python3
"""Mass-paraphrase task instructions on the MI300X.

Primary path: vLLM + Qwen2.5-32B-Instruct (fast batch inference).
Fallback:     transformers + Qwen2.5-14B-Instruct (if vLLM unavailable).

Each instruction gets K diverse rewrites. A rewrite is ACCEPTED only if it
preserves every number and >=80% of proper names, differs from the original,
and is length-sane. Payloads (quoted text) are reattached verbatim later.

    python paraphrase_vllm.py --k 3 [--fallback]
"""
from __future__ import annotations

import argparse
import json
import re

SYS = ("You rewrite task instructions. Rewrite the user's instruction in {k} "
       "genuinely different ways: vary sentence structure, register, and word "
       "choice (formal, casual, imperative, question...). STRICT RULES: keep "
       "the exact same task and expected answer; preserve every number, every "
       "name, and every explicit requirement; do not add or remove constraints; "
       "output ONLY the {k} rewrites, one per numbered line.")

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
_LINE = re.compile(r"^\s*\d+\s*[).:\-]\s*(.+?)\s*$")


def ok(orig: str, var: str) -> bool:
    if not var or var.strip().casefold() == orig.strip().casefold():
        return False
    if not (0.4 * len(orig) <= len(var) <= 3.0 * len(orig)):
        return False
    if set(_NUM.findall(orig)) - set(_NUM.findall(var)):
        return False                       # every number must survive
    names = set(_NAME.findall(orig))
    if names:
        kept = sum(1 for n in names if n in var)
        if kept < 0.8 * len(names):
            return False                   # names mostly preserved
    return True


def parse(text: str, orig: str, k: int) -> list[str]:
    out = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            v = m.group(1).strip().strip('"')
            if ok(orig, v):
                out.append(v)
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="to_para.jsonl")
    ap.add_argument("--out", default="para.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fallback", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    print(f"paraphrasing {len(rows)} instructions, k={args.k}", flush=True)
    sys_prompt = SYS.format(k=args.k)

    use_vllm = not args.fallback
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            print(f"vllm unavailable ({type(e).__name__}) -> transformers fallback", flush=True)
            use_vllm = False

    if use_vllm:
        model_id = "Qwen/Qwen2.5-32B-Instruct"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        llm = LLM(model=model_id, max_model_len=2048, gpu_memory_utilization=0.92)
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": r["instruction"]}],
            tokenize=False, add_generation_prompt=True) for r in rows]
        sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=300)
        outs = llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "Qwen/Qwen2.5-14B-Instruct"
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        texts = []
        B = 32
        for s in range(0, len(rows), B):
            chunk = rows[s:s + B]
            batch = [tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": r["instruction"]}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=300, do_sample=True,
                                 temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            for j in range(len(chunk)):
                texts.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True))
            if (s // B) % 10 == 0:
                print(f"  {s + len(chunk)}/{len(rows)}", flush=True)

    n_var = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, text in zip(rows, texts):
            variants = parse(text, r["instruction"], args.k)
            n_var += len(variants)
            f.write(json.dumps({"pid": r["pid"], "variants": variants},
                               ensure_ascii=False) + "\n")
    print(f"accepted {n_var} variants ({n_var / max(1, len(rows)):.2f}/instruction) -> {args.out}")
    assert n_var >= len(rows), "PARAPHRASE YIELD TOO LOW - inspect model output"


if __name__ == "__main__":
    main()
