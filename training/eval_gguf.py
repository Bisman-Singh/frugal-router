#!/usr/bin/env python3
"""Sized post-quantization gate: evaluate the EXACT shipped artifact.

Runs the quantized GGUF via llama.cpp over >=300 graded tasks and compares
against the bf16 eval report. Exported models most commonly regress through
chat-template or EOS mismatches, which only this test catches. Acceptance:
the GGUF must land within 2 accuracy points of bf16.

    python eval_gguf.py --gguf tuned-final-q4km.gguf --n 300 --threads 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SYSTEMS  # noqa: E402
from eval_gpu import build_pool, grade  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bf16-report", default="eval_report.json")
    ap.add_argument("--max-tokens", type=int, default=220)
    args = ap.parse_args()

    from llama_cpp import Llama

    llm = Llama(model_path=args.gguf, n_ctx=2048, n_threads=args.threads,
                verbose=False)
    tasks = build_pool(args.n)
    print(f"pool: {len(tasks)} tasks | threads={args.threads}")

    stats = defaultdict(lambda: {"n": 0, "ok": 0, "unsure": 0})
    lat = []
    for i, t in enumerate(tasks, 1):
        system = SYSTEMS.get(t["category"], SYSTEMS["factual"]) + \
            " If you are not confident the answer is correct, reply with exactly UNSURE."
        t0 = time.monotonic()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": t["prompt"]}],
                max_tokens=args.max_tokens, temperature=0.0)
            answer = (out["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            answer = ""
        lat.append(time.monotonic() - t0)
        s = stats[t["category"]]
        s["n"] += 1
        if answer.upper().startswith("UNSURE"):
            s["unsure"] += 1
        elif grade(t, answer):
            s["ok"] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)}  (p50 {sorted(lat)[len(lat)//2]:.1f}s/task)",
                  flush=True)

    tot = {"n": 0, "ok": 0, "unsure": 0}
    print(f"\n{'category':<15}{'n':>5}{'ok':>6}{'unsure':>8}{'acc(all)':>10}")
    for cat, s in sorted(stats.items()):
        for k in tot:
            tot[k] += s[k]
        print(f"{cat:<15}{s['n']:>5}{s['ok']:>6}{s['unsure']:>8}"
              f"{s['ok'] / max(1, s['n']):>10.0%}")
    gguf_acc = tot["ok"] / max(1, tot["n"])
    lat.sort()
    print(f"{'TOTAL':<15}{tot['n']:>5}{tot['ok']:>6}{tot['unsure']:>8}{gguf_acc:>10.0%}")
    print(f"latency: p50 {lat[len(lat)//2]:.1f}s  p95 {lat[int(len(lat)*0.95)]:.1f}s per task")

    bf16 = None
    p = Path(args.bf16_report)
    if p.exists():
        bf16 = json.loads(p.read_text()).get("full_local")
    if bf16 is not None:
        delta = (bf16 - gguf_acc) * 100
        verdict = "PASS" if delta <= 2.0 else "FAIL"
        print(f"\nbf16 full-local: {bf16:.1%} | GGUF: {gguf_acc:.1%} | "
              f"quant drop: {delta:.1f} pts -> {verdict} (bar: <=2.0)")
        sys.exit(0 if verdict == "PASS" else 1)
    print("\n(no bf16 report found; absolute numbers only)")


if __name__ == "__main__":
    main()
