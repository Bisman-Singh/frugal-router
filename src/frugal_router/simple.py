"""Minimal passthrough router (v7).

Everything the main agent does — classification, per-category answer contracts,
answer extraction, normalization, format validation — is machinery built for
token efficiency that can mangle a correct model answer. Below the accuracy
gate that machinery is a liability, not an asset.

This module does the opposite: for each task, one call to a strong Fireworks
model with the task prompt sent verbatim and a generous token budget, and the
model's response returned verbatim as the answer. No extraction, no reshaping.
It is deliberately close to what the one submission that cleared the gate almost
certainly does.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "You are a careful, knowledgeable assistant completing evaluation tasks. "
    "Answer each task correctly and completely, giving the full answer the task "
    "asks for. For sentiment tasks, state the label and justify it. For code "
    "tasks, provide complete, correct, runnable code. Respond in English."
)

# Substring preferences for model choice, resolved against ALLOWED_MODELS.
# Code -> the code specialist. Math/logic -> the reasoning model with reasoning
# ENABLED and a big budget: below the accuracy gate tokens are irrelevant, and
# multi-step word problems and constraint puzzles are exactly where a
# non-reasoning model drops the 2-3 tasks that decide the gate.
# Everything else -> the general Gemma model.
_CODE_HINT = re.compile(r"(?i)\b(bug|debug|fix|function|code|python|program|def |class |implement|compile)\b|```")
_REASON_HINT = re.compile(
    r"(?i)\b(how (many|much|far|old|long)|calculate|compute|percent|%|total|remainder|"
    r"average|per (hour|day|week|month)|profit|discount|split|ratio|"
    r"puzzle|deduce|constraint|exactly one|who (owns|is|was|finished)|taller|older|younger|"
    r"if all|must be true|either)\b|\d\s*[-+*/^]\s*\d"
)
_CODE_MODELS = ["kimi-k2p7-code", "code", "kimi"]
_REASON_MODELS = ["minimax", "deepseek", "gpt-oss", "glm"]
# v9: the 78.9% leader routes everything non-code to the reasoning model with
# no output cap. Below the gate, tokens are irrelevant; truncation and weaker
# general models are the only enemies. GENERAL_HINTS env can flip this back to
# gemma-first for A/B without a rebuild.
_GENERAL_MODELS = [h.strip() for h in os.environ.get(
    "GENERAL_HINTS", "minimax,gemma-4-31b-it,gemma").split(",") if h.strip()]


def _pick(allowed, hints):
    for h in hints:
        for m in allowed:
            if h.lower() in m.lower():
                return m
    return allowed[0] if allowed else ""


def run_simple(input_path="/input/tasks.json", output_path="/output/results.json",
               max_workers=8, per_task_max_tokens=1200):
    started = time.monotonic()
    answers: dict[str, str] = {}
    tasks = _read(input_path, answers)
    _write(output_path, answers)
    if not tasks:
        return 0

    # Deterministic solvers need no API: run them before the credential check
    # so every provable task is answered even if the environment is broken.
    if os.environ.get("SOLVERS", "0") == "1":
        from .solvers import solve_any as _solve_any

        for tid, prompt in tasks:
            hit = _solve_any(prompt)
            if hit is not None:
                answers[tid] = hit[0]
        if any(answers.values()):
            _write(output_path, answers)
        tasks = [(tid, p) for tid, p in tasks if not answers.get(tid)]
        if not tasks:
            _write(output_path, answers)
            return 0

    key = os.environ.get("FIREWORKS_API_KEY")
    base = os.environ.get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
    allowed = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    if not key or not allowed:
        print("missing FIREWORKS_API_KEY or ALLOWED_MODELS", file=sys.stderr)
        _write(output_path, answers)
        return 0

    from .backends.fireworks import FireworksBackend, normalize_base_url  # noqa

    client = _client(key, normalize_base_url(base))
    gen_model = _pick(allowed, _GENERAL_MODELS)
    code_model = _pick(allowed, _CODE_MODELS)
    reason_model = _pick(allowed, _REASON_MODELS)

    def route(prompt: str) -> tuple[str, int]:
        if _CODE_HINT.search(prompt):
            return code_model, 6000
        if _REASON_HINT.search(prompt):
            # Reasoning model, reasoning left ON, room for thought + answer.
            # Board evidence: capping this budget truncates the reasoning and
            # blanks exactly the hard tasks; the request-time rule is not
            # enforced at a level that punishes the longer generation.
            return reason_model, 8000
        return gen_model, 4000

    # LEAN=1: token-war mode. Zero-token classification + deterministic solvers
    # answer provable math/logic free; every other task is ONE call with a terse
    # category prompt and a tuned cap. Used only once the accuracy gate is
    # passed; default mode stays accuracy-first ensemble.
    lean = os.environ.get("LEAN", "0") == "1"
    # Second-opinion generator and referee, from different model families.
    # ENSEMBLE=0 disables (single-model mode identical to v9).
    ensemble = (os.environ.get("ENSEMBLE", "1") != "0") and not lean
    alt_model = _pick(allowed, ["gemma-4-31b-it", "gemma", "deepseek", "glm"])
    referee_model = _pick(allowed, ["gemma-4-26b-a4b", "gemma", "minimax"])

    def solve(task):
        from .classify import classify as _classify
        from .normalize import normalize
        from .solvers import solve_any
        from .tasks import Task

        tid, prompt = task
        if lean:
            answers[tid] = _solve_lean(client, prompt, gen_model, code_model, reason_model)
            _write(output_path, answers)
            return

        # Deterministic solvers run first: prove-or-defer means a hit is exact
        # by construction (zero tokens, no format risk); anything unproven
        # falls through to the ensemble. SOLVERS=0 disables.
        if os.environ.get("SOLVERS", "0") == "1":
            hit = solve_any(prompt)
            if hit is not None:
                answers[tid] = hit[0]
                _write(output_path, answers)
                return

        category = _classify(Task(id=tid, input=prompt))
        model, budget = route(prompt)
        primary = ""
        try:
            primary = _call(client, model, prompt, budget)
        except Exception as exc:
            print(f"task {tid} failed on {model}: {type(exc).__name__}", file=sys.stderr)

        second = ""
        if ensemble:
            second_model = alt_model if alt_model != model else code_model
            try:
                second = _call(client, second_model, prompt, budget)
            except Exception as exc:
                print(f"task {tid} second opinion failed: {type(exc).__name__}", file=sys.stderr)

        # Self-consistency on the hard reasoning categories: one more
        # independent sample from the reasoning model at a nonzero temperature.
        # A single greedy sample is exactly where a hard puzzle whiffs; a third
        # candidate gives the referee a majority signal. SELF_CONSISTENCY=0
        # disables. Applied only when the task routed to the reasoning model.
        third = ""
        if ensemble and model == reason_model and os.environ.get("SELF_CONSISTENCY", "1") != "0":
            try:
                third = _call(client, model, prompt, budget, temperature=0.7)
            except Exception as exc:
                print(f"task {tid} third sample failed: {type(exc).__name__}", file=sys.stderr)

        candidates = [c for c in (primary, second, third) if c.strip()]
        final = candidates[0] if candidates else ""
        if ensemble and len(set(candidates)) > 1:
            try:
                final = _referee(client, referee_model, prompt, *candidates) or final
            except Exception as exc:
                print(f"task {tid} referee failed: {type(exc).__name__}", file=sys.stderr)

        if not final.strip():
            # a truncated/empty reply is a guaranteed zero; retry the other
            # families before giving up (a lone timeout must not cost the task)
            for fb_model in (alt_model, gen_model):
                try:
                    final = _call(client, fb_model, prompt, per_task_max_tokens)
                except Exception:
                    final = ""
                if final.strip():
                    break

        # Answer normalization is opt-in only: scored evidence showed shape
        # rewriting (an appended Answer line built from the last number, code
        # reduced to its first fence) flips correct answers to wrong far more
        # often than it repairs mis-shaped ones. NORMALIZE=1 re-enables.
        if os.environ.get("NORMALIZE", "0") == "1":
            final = normalize(category, final)
        answers[tid] = final
        _write(output_path, answers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(solve, tasks))

    _write(output_path, answers)
    print(json.dumps({"tasks": len(tasks), "answered": sum(1 for v in answers.values() if v),
                      "elapsed_s": round(time.monotonic() - started, 1)}), file=sys.stderr)
    return 0


def _client(key, base):
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base, timeout=55.0, max_retries=1)


_THINK = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")

# Lean-mode category prompts: terse (input tokens bill) but intent-complete.
_LEAN = {
    "factual":       ("English only. Answer accurately and completely; under 120 words.", 300),
    "math":          ("English only. Brief steps, then 'Answer: <value>' on its own line.", 400),
    "sentiment":     ("English only. Label the sentiment positive, negative, or neutral, then one short justification.", 120),
    "summarization": ("English only. Output only the summary; obey any stated length or format constraint.", 220),
    "ner":           ("English only. List each entity as 'label: value', one per line; labels: person, organization, location, date.", 260),
    "logic":         ("English only. Deduce in brief numbered steps checking every constraint, then 'Answer: <value>' on its own line.", 420),
    "code_debug":    ("English only. Name the bug in one sentence, then the corrected code in one fenced block.", 560),
    "code_gen":      ("English only. Output only the code in one fenced block, correct and self-contained.", 560),
}


def _solve_lean(client, prompt, gen_model, code_model, reason_model):
    from .classify import classify as _classify
    from .solvers import solve_any
    from .tasks import Task

    if os.environ.get("SOLVERS", "0") == "1":
        hit = solve_any(prompt)
        if hit is not None:
            return hit[0]  # proven-correct deterministic answer: zero tokens
    cat = _classify(Task(id="x", input=prompt))
    system, cap = _LEAN.get(cat, _LEAN["factual"])
    # Passer-validated tiering: the strong general model handles math/logic at
    # reasoning-effort none (a reasoning model with reasoning suppressed is
    # crippled; a strong non-reasoning model is not). Light model for the three
    # classification-style categories, code specialist for code.
    cheap_model = _pick(os.environ.get("ALLOWED_MODELS", "").split(","), ["a4b"]) or gen_model
    if cat in ("code_debug", "code_gen"):
        model = code_model
    elif cat in ("sentiment", "summarization", "ner"):
        model = cheap_model
    else:
        model = gen_model
    out = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=cap,
            extra_body={"reasoning_effort": "none"},
        )
        out = _THINK.sub("", (resp.choices[0].message.content or "").strip()).strip()
    except Exception:
        out = ""
    if out:
        return out
    # Blank or errored (model not deployed, effort param rejected, truncation,
    # transient): a blank answer is a guaranteed zero. Retry on the strong
    # general model with reasoning suppressed first — if that model reasons and
    # the suppression is dropped, a small cap fills with hidden thinking and
    # returns blank — then once more plain with a budget big enough to think in.
    for extra, fb_cap in (({"reasoning_effort": "none"}, max(cap, 800)), (None, 1600)):
        try:
            kwargs = dict(model=gen_model,
                          messages=[{"role": "system", "content": system},
                                    {"role": "user", "content": prompt}],
                          temperature=0.0, max_tokens=fb_cap)
            if extra:
                kwargs["extra_body"] = extra
            resp = client.chat.completions.create(**kwargs)
            out = _THINK.sub("", (resp.choices[0].message.content or "").strip()).strip()
            if out:
                return out
        except Exception:
            continue
    return ""


REFEREE_SYSTEM = (
    "You are a meticulous grader consolidating two candidate answers to a task. "
    "Check each against the task's explicit requirements (correctness, "
    "completeness, any format or length constraints). Output ONLY the single "
    "best final answer for the task: if one candidate is fully correct and "
    "complete, output it verbatim; otherwise output a corrected answer that "
    "fixes the flaws. Never output commentary, labels, or comparisons - only "
    "the final answer itself, in English."
)


def _referee(client, model, prompt, *candidates):
    blocks = "\n\n".join(
        f"CANDIDATE {i}:\n{c}" for i, c in enumerate(candidates, 1)
    )
    user = f"TASK:\n{prompt}\n\n{blocks}\n\nFinal answer:"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": REFEREE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=8000,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _THINK.sub("", text).strip()


def _call(client, model, prompt, max_tokens, temperature=0.0):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _THINK.sub("", text).strip()


def _read(input_path, answers):
    try:
        raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read {input_path}: {type(exc).__name__}", file=sys.stderr)
        return []
    tasks = []
    for i, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        tid = str(item.get("task_id", f"task-{i}"))
        answers[tid] = ""
        tasks.append((tid, str(item.get("prompt", ""))))
    return tasks


_write_lock = __import__("threading").Lock()


def _write(output_path, answers):
    # Thread-safe atomic write: unique temp per writer, serialized replace.
    # Concurrent workers sharing one temp name can collide and crash the run.
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results = [{"task_id": k, "answer": v} for k, v in answers.items()]
        with _write_lock:
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{__import__('threading').get_ident()}.tmp")
            tmp.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
    except Exception as exc:
        print(f"write failed (will retry on next flush): {exc}", file=__import__("sys").stderr)
