#!/usr/bin/env python3
"""Per-category eval harness for the generated task set.

Runs the agent's real answering pieces — deterministic solvers, per-category
prompts, answer normalization — against a chosen backend, then grades every
answer deterministically (numeric compare, label match, entity coverage,
sandboxed code tests). The point is the per-category accuracy table: it shows
WHERE the pipeline leaks instead of one opaque number per submission.

Backends:
  none    solvers only, no network (sanity-check the deterministic layer)
  fw      Fireworks dev key, stand-in models (fast, reasoning family)
  gemma   gemma-4-31b-it via the Gemini OpenAI-compat endpoint (real T1
          workhorse weights; free tier, paced, intermittent 5xx tolerated)

Usage:
    python scripts/run_eval2.py --backend fw --tasks data/eval_gen.jsonl --limit 48
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frugal_router.normalize import normalize          # noqa: E402
from frugal_router.solvers import solve_any            # noqa: E402
from frugal_router.simple import _LEAN, _THINK         # noqa: E402

# ------------------------------------------------------------- backends -----

FW_BASE = "https://api.fireworks.ai/inference/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

FW_MODELS = {  # dev-key stand-ins for the real ALLOWED_MODELS tiers
    "general": "accounts/fireworks/models/gpt-oss-120b",
    "reason": "accounts/fireworks/models/deepseek-v4-pro",
    "code": "accounts/fireworks/models/kimi-k2p6",
}
GEMMA_MODEL = "gemma-4-31b-it"

_TOKENS = {"prompt": 0, "completion": 0}


def _client(backend: str):
    from openai import OpenAI
    if backend == "fw":
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            sys.exit("FIREWORKS_API_KEY required for --backend fw")
        return OpenAI(api_key=key, base_url=FW_BASE, timeout=40.0, max_retries=1)
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY required for --backend gemma")
    return OpenAI(api_key=key, base_url=GEMINI_BASE, timeout=40.0, max_retries=0)


def _model_for(backend: str, category: str) -> str:
    if backend == "gemma":
        return GEMMA_MODEL
    if category in ("code_gen", "code_debug"):
        return FW_MODELS["code"]
    if category in ("math", "logic"):
        return FW_MODELS["reason"]
    return FW_MODELS["general"]


_last_call = 0.0

def ask(client, backend: str, category: str, prompt: str) -> str:
    global _last_call
    system, cap = _LEAN.get(category, _LEAN["factual"])
    model = _model_for(backend, category)
    if backend == "fw":
        cap = max(cap, 2500)  # stand-ins are reasoning models: thought eats budget
    attempts = 3 if backend == "gemma" else 2
    for attempt in range(attempts):
        if backend == "gemma":  # free tier: 15 RPM per model
            wait = 4.6 - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
        try:
            kwargs = dict(model=model,
                          messages=[{"role": "system", "content": system},
                                    {"role": "user", "content": prompt}],
                          temperature=0.0, max_tokens=cap)
            if backend == "fw":
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            resp = client.chat.completions.create(**kwargs)
            _last_call = time.monotonic()
            if getattr(resp, "usage", None):
                _TOKENS["prompt"] += resp.usage.prompt_tokens or 0
                _TOKENS["completion"] += resp.usage.completion_tokens or 0
            text = (resp.choices[0].message.content or "").strip()
            stripped = _THINK.sub("", text).strip()
            out = stripped if stripped else text
            if out:
                return out
        except Exception as exc:  # 5xx/429/timeout: brief backoff, retry
            _last_call = time.monotonic()
            if attempt + 1 == attempts:
                print(f"  ! {category} call failed: {type(exc).__name__}", file=sys.stderr)
            time.sleep(2.0 * (attempt + 1))
    return ""

# -------------------------------------------------------------- grading -----

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANS_LINE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+)$")
_FENCE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)


def _numbers(text: str) -> list[float]:
    return [float(m.group(0).replace(",", "")) for m in _NUM.finditer(text)]


def grade(task: dict, answer: str) -> bool:
    g, text = task["grader"], (answer or "").strip()
    if not text:
        return False
    low = text.lower()
    kind = g["type"]

    if kind == "numeric":
        m = _ANS_LINE.search(text)
        pool = _numbers(m.group(1)) if m else []
        if not pool:
            pool = _numbers(text)[-2:]  # tolerate a trailing unit repeat
        return any(abs(v - g["expected"]) < 0.01 for v in pool)

    if kind == "label":
        pos = low.find(g["expected"])
        if pos < 0:
            return False
        rivals = [low.find(o) for o in g.get("others", []) if o in low]
        return all(pos <= r for r in rivals) if rivals else True

    if kind == "yesno":
        m = _ANS_LINE.search(text)
        scope = (m.group(1) if m else text).lower()
        want, other = g["expected"], ("no" if g["expected"] == "yes" else "yes")
        w = re.search(rf"\b{want}\b", scope)
        o = re.search(rf"\b{other}\b", scope)
        return bool(w) and (not o or w.start() <= o.start())

    if kind == "choice":
        pos = low.find(g["expected"].lower())
        if pos < 0:
            return False
        m = _ANS_LINE.search(text)
        if m and g["expected"].lower() in m.group(1).lower():
            return True
        rivals = [low.rfind(o.lower()) for o in g.get("others", []) if o.lower() in low]
        return all(low.rfind(g["expected"].lower()) >= r for r in rivals) if rivals else True

    if kind == "contains_all":
        ok = all(k.lower() in low for k in g["expected"])
        bad = any(k.lower() in low for k in g.get("must_not", []))
        return ok and not bad

    if kind == "summary":
        sentences = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
        if len(sentences) > g.get("max_sentences", 1) + 0:  # small tolerance below
            if len(sentences) > g.get("max_sentences", 1) + 1:
                return False
        return any(k.lower() in low for k in g["must_mention_any"])

    if kind == "code":
        m = _FENCE.search(text)
        code = m.group(1) if m else text
        checks = "\n".join(
            f"assert {call} == {expected!r}, {call!r}" for call, expected in g["tests"]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code + "\n\n" + checks + "\nprint('PASS')\n")
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, timeout=6)
            return proc.returncode == 0 and "PASS" in proc.stdout
        except Exception:
            return False
        finally:
            os.unlink(path)

    raise ValueError(f"unknown grader {kind}")

# ----------------------------------------------------------------- main -----

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/eval_gen.jsonl")
    ap.add_argument("--backend", choices=["none", "fw", "gemma"], default="none")
    ap.add_argument("--limit", type=int, default=0, help="cap total tasks (0 = all)")
    ap.add_argument("--cats", default="", help="comma list to filter categories")
    ap.add_argument("--no-solvers", action="store_true")
    ap.add_argument("--dump", default="", help="write per-task results jsonl here")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if args.cats:
        keep = set(args.cats.split(","))
        tasks = [t for t in tasks if t["category"] in keep]
    if args.limit:
        # even per-category slice, not the first N of a shuffled file
        by_cat: dict[str, list] = defaultdict(list)
        for t in tasks:
            by_cat[t["category"]].append(t)
        per = max(1, args.limit // max(1, len(by_cat)))
        tasks = [t for cat in sorted(by_cat) for t in by_cat[cat][:per]]

    client = _client(args.backend) if args.backend != "none" else None
    stats = defaultdict(lambda: {"n": 0, "solver": 0, "correct": 0})
    rows = []

    for k, t in enumerate(tasks, 1):
        cat = t["category"]
        answer, via = "", "model"
        if not args.no_solvers:
            hit = solve_any(t["prompt"])
            if hit is not None:
                answer, via = hit[0], "solver"
        if not answer and client is not None:
            answer = normalize(cat, ask(client, args.backend, cat, t["prompt"]))
        ok = grade(t, answer) if answer else False
        s = stats[cat]
        s["n"] += 1
        s["correct"] += ok
        s["solver"] += via == "solver"
        rows.append({"id": t["id"], "category": cat, "via": via, "ok": ok,
                     "answer": answer[:400], "prompt": t["prompt"][:200]})
        if k % 20 == 0:
            print(f"  {k}/{len(tasks)} done", file=sys.stderr)

    print(f"\n{'category':<15}{'n':>4}{'solver':>8}{'correct':>9}{'acc':>8}")
    tot_n = tot_c = 0
    for cat in sorted(stats):
        s = stats[cat]
        tot_n += s["n"]; tot_c += s["correct"]
        print(f"{cat:<15}{s['n']:>4}{s['solver']:>8}{s['correct']:>9}"
              f"{s['correct'] / max(1, s['n']):>8.0%}")
    print(f"{'TOTAL':<15}{tot_n:>4}{'':>8}{tot_c:>9}{tot_c / max(1, tot_n):>8.0%}")
    print(f"tokens: prompt={_TOKENS['prompt']} completion={_TOKENS['completion']} "
          f"total={_TOKENS['prompt'] + _TOKENS['completion']}")

    if args.dump:
        Path(args.dump).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"per-task dump -> {args.dump}")


if __name__ == "__main__":
    main()
