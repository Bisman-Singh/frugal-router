#!/usr/bin/env python3
"""Rejection-sampled teacher distillation for the local model.

Strong teacher models answer thousands of tasks from OUR 8-category
distribution; the deterministic graders keep only provably-correct answers
(rejection sampling with an exact judge, not an LLM's opinion). The output is
gold SFT data in the same messages format the trainers consume.

    FIREWORKS_API_KEY=... python training/distill_teacher.py \
        --n 5000 --out training/distill.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import ABSTAIN_NOTE, SYSTEMS  # noqa: E402
from eval_gpu import grade  # noqa: E402  (deterministic graders incl. code)
from gen_eval import GENERATORS  # noqa: E402

SYSTEMS.setdefault("code_gen", "English only. Be concise; no preamble. Output only the complete, correct, self-contained code in one fenced block.")
SYSTEMS.setdefault("code_debug", "English only. Be concise; no preamble. Name the bug in one sentence, then give the complete corrected code in one fenced block.")

TEACHERS = {
    "code_gen": "accounts/fireworks/models/kimi-k2p6",
    "code_debug": "accounts/fireworks/models/kimi-k2p6",
    "math": "accounts/fireworks/models/deepseek-v4-pro",
    "logic": "accounts/fireworks/models/deepseek-v4-pro",
}
DEFAULT_TEACHER = "accounts/fireworks/models/gpt-oss-120b"

_LOCK = threading.Lock()
_STATS = {"asked": 0, "kept": 0}
_THINK = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")


def make_pool(n):
    tasks = []
    per = max(1, n // (8 * 6))
    for seed in (211, 223, 227, 229, 233, 239):
        rng = random.Random(seed)
        for cat, gen in GENERATORS.items():
            for i in range(per):
                t = gen(rng, i + seed * 1000)
                t["id"] = f"d{seed}-{t['id']}"
                tasks.append(t)
    random.Random(0).shuffle(tasks)
    return tasks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="training/distill.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--samples", type=int, default=2, help="teacher attempts per task")
    args = ap.parse_args()

    from openai import OpenAI

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY required")
    client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1",
                    timeout=45.0, max_retries=1)

    tasks = make_pool(args.n)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")

    def work(t):
        cat = t["category"]
        system = SYSTEMS.get(cat, SYSTEMS["factual"])
        model = TEACHERS.get(cat, DEFAULT_TEACHER)
        for attempt in range(args.samples):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": t["prompt"]}],
                    temperature=0.0 if attempt == 0 else 0.7,
                    max_tokens=700,
                    extra_body={"reasoning_effort": "low"},
                )
                text = _THINK.sub("", (resp.choices[0].message.content or "")).strip()
            except Exception:
                continue
            with _LOCK:
                _STATS["asked"] += 1
            if not text or len(text) > 1600:
                continue
            if grade(t, text):  # rejection sampling: exact graders only
                ex = {"messages": [
                    {"role": "system", "content": system + ABSTAIN_NOTE},
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": text},
                ]}
                with _LOCK:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    fh.flush()
                    _STATS["kept"] += 1
                return

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(work, tasks), 1):
            if i % 250 == 0:
                print(f"  {i}/{len(tasks)}  kept={_STATS['kept']}  "
                      f"({time.monotonic() - t0:.0f}s)", flush=True)
    fh.close()
    print(f"DONE: kept {_STATS['kept']} graded-correct teacher answers "
          f"of {len(tasks)} tasks -> {out_path}")


if __name__ == "__main__":
    main()
