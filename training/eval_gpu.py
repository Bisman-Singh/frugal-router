#!/usr/bin/env python3
"""GPU batch evaluation of the merged model over thousands of graded tasks.

Builds a large eval pool (multi-seed generators + real benchmark validation
splits fetched fresh — none of it in the training data), runs batched
generation on the merged fp16 model, grades deterministically, and prints the
acceptance report including a FULL-LOCAL projection: the accuracy a 0-token
entry would score if this model answered every category by itself (UNSURE and
wrong both count against it; UNSURE would escalate only in hybrid mode).

    python eval_gpu.py --model ./tuned/merged --n 2000
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SYSTEMS  # noqa: E402
from gen_eval import GENERATORS  # noqa: E402

_FENCE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANS = re.compile(r"(?im)^\s*\**answer\**\s*[:=]\s*(.+?)\s*$")


def grade(task, answer):
    g, text = task["grader"], (answer or "").strip()
    if not text or text.upper().startswith("UNSURE"):
        return False
    low = text.lower()
    kind = g["type"]
    if kind == "numeric":
        m = _ANS.search(text)
        scope = m.group(1) if m else text[-90:]
        nums = [float(x.replace(",", "")) for x in _NUM.findall(scope)]
        return any(abs(v - g["expected"]) < 0.01 for v in nums)
    if kind == "label":
        pos = low.find(g["expected"])
        rivals = [low.find(o) for o in g.get("others", []) if o in low]
        return pos >= 0 and all(pos <= r for r in rivals)
    if kind == "yesno":
        m = _ANS.search(text)
        scope = (m.group(1) if m else text).lower()
        want = g["expected"]
        other = "no" if want == "yes" else "yes"
        w = re.search(rf"\b{want}\b", scope)
        o = re.search(rf"\b{other}\b", scope)
        return bool(w) and (not o or w.start() <= o.start())
    if kind == "choice":
        if g["expected"].lower() not in low:
            return False
        rivals = [low.rfind(o.lower()) for o in g.get("others", []) if o.lower() in low]
        return all(low.rfind(g["expected"].lower()) >= r for r in rivals)
    if kind == "contains_any":
        return any(k.lower() in low for k in g["expected"])
    if kind == "contains_all":
        return all(k.lower() in low for k in g["expected"]) and \
            not any(k.lower() in low for k in g.get("must_not", []))
    if kind == "summary":
        sents = [s for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()]
        if len(sents) > g.get("max_sentences", 1) + 1:
            return False
        return any(k.lower() in low for k in g["must_mention_any"])
    if kind == "code":
        m = _FENCE.search(text)
        code = m.group(1) if m else text
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        checks = "\n".join(f"assert {c} == {e!r}, {c!r}" for c, e in g["tests"])
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code + "\n\n" + checks + "\nprint('PASS')\n")
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, "-I", path], capture_output=True,
                                  text=True, timeout=6, env={})
            return proc.returncode == 0 and "PASS" in proc.stdout
        except Exception:
            return False
    return False


def build_pool(n):
    tasks = []
    rng = random.Random(101)
    per = max(20, n // 16)
    for seed in (31, 57, 83, 111):
        r2 = random.Random(seed)
        for cat, gen in GENERATORS.items():
            for i in range(per // 4):
                t = gen(r2, i + seed * 100)
                t["id"] = f"g{seed}-{t['id']}"
                tasks.append(t)
    # real validation-split items, fetched fresh (never trained on)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fetch_bench import (fetch_gsm8k, fetch_mbpp, fetch_nq, fetch_sst2,
                                 fetch_xsum)
        for fn, k in ((fetch_gsm8k, 60), (fetch_sst2, 60), (fetch_xsum, 50),
                      (fetch_nq, 50), (fetch_mbpp, 60)):
            try:
                tasks.extend(fn(k))
            except Exception as e:
                print(f"  bench source {fn.__name__}: {type(e).__name__} (skipped)")
    except Exception:
        pass
    rng.shuffle(tasks)
    return tasks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./tuned/merged")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    tasks = build_pool(args.n)
    print(f"pool: {len(tasks)} tasks "
          f"({sum(1 for t in tasks if t['id'].startswith('bench'))} real-benchmark)")

    stats = defaultdict(lambda: {"n": 0, "ok": 0, "unsure": 0})
    for i in range(0, len(tasks), args.batch):
        chunk = tasks[i:i + args.batch]
        prompts = []
        for t in chunk:
            msgs = [{"role": "system",
                     "content": SYSTEMS.get(t["category"], SYSTEMS["factual"]) +
                     " If you are not confident the answer is correct, reply with exactly UNSURE."},
                    {"role": "user", "content": t["prompt"]}]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=220, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for t, seq in zip(chunk, out):
            answer = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            s = stats[t["category"]]
            s["n"] += 1
            if answer.strip().upper().startswith("UNSURE"):
                s["unsure"] += 1
            elif grade(t, answer):
                s["ok"] += 1
        done = min(i + args.batch, len(tasks))
        if done % (args.batch * 5) < args.batch:
            print(f"  {done}/{len(tasks)}", flush=True)

    print(f"\n{'category':<15}{'n':>5}{'ok':>6}{'unsure':>8}{'acc(ans)':>10}{'acc(all)':>10}")
    tot = {"n": 0, "ok": 0, "unsure": 0}
    for cat, s in sorted(stats.items()):
        for k in tot:
            tot[k] += s[k]
        answered = s["n"] - s["unsure"]
        print(f"{cat:<15}{s['n']:>5}{s['ok']:>6}{s['unsure']:>8}"
              f"{(s['ok'] / max(1, answered)):>10.0%}{(s['ok'] / max(1, s['n'])):>10.0%}")
    answered = tot["n"] - tot["unsure"]
    full_local = tot["ok"] / max(1, tot["n"])
    print(f"{'TOTAL':<15}{tot['n']:>5}{tot['ok']:>6}{tot['unsure']:>8}"
          f"{(tot['ok'] / max(1, answered)):>10.0%}{full_local:>10.0%}")
    print(f"\nFULL-LOCAL PROJECTION (0-token entry on a fresh set): {full_local:.1%}")
    print("HYBRID KEPT-ACCURACY (what the gate sees on answered): "
          f"{tot['ok'] / max(1, answered):.1%}  | escalation rate: {tot['unsure'] / tot['n']:.1%}")
    Path(args.out).write_text(json.dumps({"stats": {k: dict(v) for k, v in stats.items()},
                                          "full_local": full_local}, indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
