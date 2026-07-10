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

# --- faithful ensemble replica (mirrors simple.py's shipped path) ------------

def ask_ensemble(client, backend: str, category: str, prompt: str,
                 reason_effort: str = "on") -> str:
    """Primary + second opinion + referee with the same system prompts and
    budget shape the shipped ensemble uses, on this backend's models."""
    from frugal_router.simple import SYSTEM, REFEREE_SYSTEM

    def call(model: str, sys_prompt: str, user: str, cap: int, effort_none: bool) -> str:
        kwargs = dict(model=model,
                      messages=[{"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user}],
                      temperature=0.0, max_tokens=cap)
        if effort_none:
            kwargs["extra_body"] = {"reasoning_effort": "none"}
        try:
            resp = client.chat.completions.create(**kwargs)
            if getattr(resp, "usage", None):
                _TOKENS["prompt"] += resp.usage.prompt_tokens or 0
                _TOKENS["completion"] += resp.usage.completion_tokens or 0
            text = (resp.choices[0].message.content or "").strip()
            stripped = _THINK.sub("", text).strip()
            return stripped if stripped else text
        except Exception as exc:
            print(f"  ! ensemble call failed ({model}): {type(exc).__name__}", file=sys.stderr)
            return ""

    is_reason = category in ("math", "logic")
    is_code = category in ("code_gen", "code_debug")
    primary_model = _model_for(backend, category)
    second_model = FW_MODELS["general"] if primary_model != FW_MODELS["general"] else FW_MODELS["reason"]
    budget = 8000 if is_reason else (6000 if is_code else 4000)
    effort_none_primary = is_reason and reason_effort == "none"

    primary = call(primary_model, SYSTEM, prompt, budget, effort_none_primary)
    second = call(second_model, SYSTEM, prompt, budget, False)
    final = primary or second
    if primary and second and primary != second:
        user = f"TASK:\n{prompt}\n\nCANDIDATE 1:\n{primary}\n\nCANDIDATE 2:\n{second}\n\nFinal answer:"
        final = call(FW_MODELS["general"], REFEREE_SYSTEM, user, 8000, False) or final
    return final


_JUDGE_SYSTEM = (
    "You are an evaluation judge for an AI benchmark. You will see a TASK and "
    "an ANSWER. Decide whether the answer correctly and completely fulfills "
    "the task's expected intent: the content must be correct, any explicit "
    "format or length constraints in the task must be obeyed, and the answer "
    "must be in English. Judge only what is asked; extra correct detail is "
    "acceptable unless it violates a stated constraint. Reply with exactly one "
    "line: 'VERDICT: correct' or 'VERDICT: incorrect'."
)


def judge_answer(judge_client, task_prompt: str, answer: str) -> bool | None:
    """LLM-judge simulation of the harness's accuracy gate. None = judge failed."""
    user = f"TASK:\n{task_prompt}\n\nANSWER:\n{answer}\n\nVerdict:"
    for _ in range(2):
        try:
            resp = judge_client.chat.completions.create(
                model="accounts/fireworks/models/glm-5p2",
                messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=1200,
                extra_body={"reasoning_effort": "low"},
            )
            text = _THINK.sub("", (resp.choices[0].message.content or "")).strip().lower()
            if "verdict: correct" in text:
                return True
            if "verdict: incorrect" in text:
                return False
        except Exception:
            time.sleep(1.5)
    return None


_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")


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

        def _limits():  # model-generated code is untrusted: cage it
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))

        try:
            # -I: isolated mode (no site, no env hooks); env={}: no secrets;
            # rlimits: no CPU/memory/file abuse. Network isolation still needs
            # a container - do not run this against untrusted models outside one.
            proc = subprocess.run([sys.executable, "-I", path],
                                  capture_output=True, text=True, timeout=6,
                                  env={}, cwd=tempfile.gettempdir(),
                                  preexec_fn=_limits)
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
    ap.add_argument("--ensemble", action="store_true",
                    help="use the shipped ensemble path (primary+second+referee)")
    ap.add_argument("--reason-effort", choices=["on", "none"], default="on",
                    help="reasoning effort for the math/logic primary in ensemble mode")
    ap.add_argument("--judge", action="store_true",
                    help="also grade every answer with an LLM judge (glm-5p2, dev key)")
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
    judge_client = _client("fw") if args.judge else None
    stats = defaultdict(lambda: {"n": 0, "solver": 0, "correct": 0,
                                 "judged": 0, "judge_n": 0, "blank": 0, "cjk": 0})
    rows = []

    for k, t in enumerate(tasks, 1):
        cat = t["category"]
        answer, via = "", "model"
        if not args.no_solvers:
            hit = solve_any(t["prompt"])
            if hit is not None:
                answer, via = hit[0], "solver"
        if not answer and client is not None:
            if args.ensemble:
                answer = ask_ensemble(client, args.backend, cat, t["prompt"],
                                      reason_effort=args.reason_effort)
            else:
                answer = ask(client, args.backend, cat, t["prompt"])
        ok = grade(t, answer) if answer else False
        verdict = None
        if judge_client is not None:
            verdict = judge_answer(judge_client, t["prompt"], answer or "(no answer)")
        s = stats[cat]
        s["n"] += 1
        s["correct"] += ok
        s["solver"] += via == "solver"
        s["blank"] += not (answer or "").strip()
        s["cjk"] += bool(_CJK.search(answer or ""))
        if verdict is not None:
            s["judge_n"] += 1
            s["judged"] += verdict
        rows.append({"id": t["id"], "category": cat, "via": via, "ok": ok,
                     "judge": verdict, "cjk": bool(_CJK.search(answer or "")),
                     "answer": answer[:600], "prompt": t["prompt"][:200]})
        if k % 20 == 0:
            print(f"  {k}/{len(tasks)} done", file=sys.stderr)

    hdr = f"\n{'category':<15}{'n':>4}{'solver':>8}{'det-acc':>9}"
    if args.judge:
        hdr += f"{'judged':>9}"
    hdr += f"{'blank':>7}{'cjk':>5}"
    print(hdr)
    tot_n = tot_c = tot_j = tot_jn = 0
    for cat in sorted(stats):
        s = stats[cat]
        tot_n += s["n"]; tot_c += s["correct"]; tot_j += s["judged"]; tot_jn += s["judge_n"]
        line = (f"{cat:<15}{s['n']:>4}{s['solver']:>8}"
                f"{s['correct'] / max(1, s['n']):>9.0%}")
        if args.judge:
            line += f"{s['judged'] / max(1, s['judge_n']):>9.0%}"
        line += f"{s['blank']:>7}{s['cjk']:>5}"
        print(line)
    total = f"{'TOTAL':<15}{tot_n:>4}{'':>8}{tot_c / max(1, tot_n):>9.0%}"
    if args.judge:
        total += f"{tot_j / max(1, tot_jn):>9.0%}"
    print(total)
    # disagreements are the diagnostic gold: det-correct but judge-incorrect =
    # format/verbosity penalty; det-wrong but judge-correct = grader artifact.
    if args.judge:
        fmt_penalty = [r for r in rows if r["ok"] and r["judge"] is False]
        grader_artifact = [r for r in rows if not r["ok"] and r["judge"]]
        print(f"det-correct/judge-INCORRECT (format penalty): {len(fmt_penalty)}"
              f" -> {sorted(set(r['category'] for r in fmt_penalty))}")
        print(f"det-wrong/judge-correct (grader artifact):   {len(grader_artifact)}"
              f" -> {sorted(set(r['category'] for r in grader_artifact))}")
    print(f"tokens: prompt={_TOKENS['prompt']} completion={_TOKENS['completion']} "
          f"total={_TOKENS['prompt'] + _TOKENS['completion']}")

    if args.dump:
        Path(args.dump).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"per-task dump -> {args.dump}")


if __name__ == "__main__":
    main()
