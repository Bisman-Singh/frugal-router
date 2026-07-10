"""The scored path: accuracy-first gate baseline.

This module is what the submitted image runs — one consolidated pipeline, no
dead branches. Design rules, in order:

1. Known-available models only in the critical path. The general workhorse and
   the code specialist have answered every scored run; the Gemma tiers are
   opportunistic last-resort fallbacks, never load-bearing (they are on-demand
   deployments that can 404).
2. Deterministic solvers first: prove-or-defer, exact by construction.
3. One primary call per task with a category contract (terse instruction +
   budget), reasoning suppressed.
4. VALIDATE the answer against the category's acceptance checks. Never rewrite
   an answer; on failure RE-ASK — first the same model with the requirement
   spelled out, then a different model family, then (hard categories) the
   reasoning mode. Escalation only on validated failure.
5. Never exit non-zero; never blow the 10-minute wall; record every request in
   an inference ledger written next to the results.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --------------------------------------------------------------- contracts --

_BASE = "English only. Be concise; no preamble."

# category -> (system instruction, max_tokens for the primary attempt)
CONTRACTS: dict[str, tuple[str, int]] = {
    "factual": (f"{_BASE} Answer accurately and completely in under 120 words.", 700),
    "math": (f"{_BASE} Show brief steps, then end with 'Answer: <value>' on its own line.", 1500),
    "sentiment": (f"{_BASE} State exactly one label - positive, negative, or neutral - then one short justification.", 300),
    "summarization": (f"{_BASE} Output only the summary and obey any stated length or format constraint exactly.", 400),
    "ner": (f"{_BASE} List each entity as 'label: value', one per line; labels: person, organization, location, date.", 500),
    "logic": (f"{_BASE} Reason in brief numbered steps checking every constraint, then end with 'Answer: <value>' on its own line.", 1800),
    "code_debug": (f"{_BASE} Name the bug in one sentence, then give the complete corrected code in one fenced block.", 2500),
    "code_gen": (f"{_BASE} Output only the complete, correct, self-contained code in one fenced block.", 2500),
}

# Back-compat exports for the eval harness (same (instruction, cap) shape).
_LEAN = CONTRACTS
SYSTEM = (
    "You are a careful, knowledgeable assistant completing evaluation tasks. "
    "Answer each task correctly and completely, giving the full answer the task "
    "asks for. Respond in English."
)
REFEREE_SYSTEM = (
    "You are a meticulous grader consolidating candidate answers to a task. "
    "Output ONLY the single best final answer for the task, in English."
)

_CODE_MODELS = ["kimi-k2p7-code", "code", "kimi"]
_GENERAL_MODELS = [h.strip() for h in os.environ.get(
    "GENERAL_HINTS", "minimax,gemma-4-31b-it,gemma").split(",") if h.strip()]
_OPPORTUNISTIC = ["gemma-4-31b-it", "gemma"]  # never load-bearing

_FENCE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")
_UNCLOSED_THINK = re.compile(r"(?i)<(?:think|thought)>(?!.*</(?:think|thought)>)", re.DOTALL)
_TRAILING_CLOSE = re.compile(r"(?is)^.*?</(?:think|thought)>\s*")

# ----------------------------------------------------------------- ledger ---

LEDGER: list[dict] = []
_ledger_lock = threading.Lock()


def _record(task_id: str, model: str, status: str, attempt: int, usage) -> None:
    entry = {"task_id": task_id, "model": model, "status": status, "attempt": attempt,
             "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}
    with _ledger_lock:
        LEDGER.append(entry)

# -------------------------------------------------------------- validation --


def validate(category: str, prompt: str, answer: str) -> str | None:
    """Acceptance check. Returns None when the answer is acceptable, else a
    short requirement string used to re-ask. Never mutates the answer."""
    a = (answer or "").strip()
    if not a:
        return "a non-empty answer is required"
    low = a.lower()

    if category == "math":
        if not re.search(r"-?\d", a):
            return "end with 'Answer: <numeric value>' on its own line"

    elif category == "sentiment":
        labels = {l for l in ("positive", "negative", "neutral")
                  if re.search(rf"\b{l}\b", low)}
        if len(labels) != 1:
            return ("state exactly one sentiment label - positive, negative, or "
                    "neutral - followed by a brief justification")

    elif category == "ner":
        hits = sum(1 for l in ("person", "organization", "location", "date") if l in low)
        if hits < 2:
            return ("list each entity as 'label: value' on its own line using the "
                    "labels person, organization, location, date")

    elif category == "summarization":
        m = re.search(r"(?i)\bin\s+(one|a single|two|three|\d+)\s+sentences?\b", prompt)
        if m:
            word = m.group(1).lower()
            want = {"one": 1, "a single": 1, "two": 2, "three": 3}.get(word)
            want = want if want is not None else int(word)
            sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", a) if s.strip()]
            if len(sentences) > want:
                return f"the summary must be at most {want} sentence(s)"
        m = re.search(r"(?i)\bin\s+(\d+)\s+words?\b", prompt)
        if m and len(a.split()) > int(m.group(1)) * 1.5:
            return f"the summary must be about {m.group(1)} words"

    elif category in ("code_gen", "code_debug"):
        block = _FENCE.search(a)
        code = block.group(1) if block else a
        is_python = bool(re.search(r"(?i)\bpython\b", prompt)) or bool(
            re.search(r"^\s*(def |import |from |class )", code, re.MULTILINE))
        if block is None and category == "code_gen":
            return "provide the code in one fenced code block"
        if is_python:
            try:
                ast.parse(code)
            except SyntaxError:
                return "provide complete, syntactically valid code in one fenced block"

    elif category == "logic":
        if re.search(r"(?i)\b(yes or no|true or false)\b", prompt) and not \
                re.search(r"\b(yes|no|true|false)\b", low):
            return "state the final yes/no answer explicitly"

    return None

# ------------------------------------------------------------------ client --


def _client(key, base):
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base,
                  timeout=float(os.environ.get("TIMEOUT_S", "45")), max_retries=1)


def _pick(allowed, hints):
    for h in hints:
        for m in allowed:
            if h.lower() in m.lower():
                return m
    return ""


def _call(client, task_id, model, system, prompt, max_tokens, attempt,
          temperature=0.0, effort="none"):
    """One completion with reasoning suppressed by default. Returns cleaned
    text ('' for an all-thought reply). Raises on transport errors so the
    caller can walk its model chain."""
    kwargs = dict(model=model,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
                  temperature=temperature, max_tokens=max_tokens)
    if effort:
        kwargs["extra_body"] = {"reasoning_effort": effort}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if effort and any(s in msg for s in ("reasoning_effort", "unsupported",
                                             "unknown parameter", "extra")):
            kwargs.pop("extra_body", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            _record(task_id, model, f"error:{type(exc).__name__}", attempt, None)
            raise
    _record(task_id, model, "ok", attempt, getattr(resp, "usage", None))
    text = (resp.choices[0].message.content or "").strip()
    if _UNCLOSED_THINK.search(text):
        return ""  # cut off mid-reasoning: no answer in this response at all
    text = _THINK.sub("", text).strip()
    if re.search(r"(?i)</(?:think|thought)>", text):
        text = _TRAILING_CLOSE.sub("", text).strip()
    return text

# ------------------------------------------------------------------- solve --


def _solve_task(client, task_id, prompt, category, chain):
    """Contract call -> validate -> corrective re-ask on the same model ->
    cross-model -> (hard categories) reasoning mode. Returns the best answer
    seen; escalation happens only on validated failure."""
    system, budget = CONTRACTS.get(category, CONTRACTS["factual"])
    best = ""

    attempts = [(chain[0], "none", 0.0, budget),
                (chain[0], "none", 0.0, budget)]      # corrective re-ask
    for extra in chain[1:]:
        attempts.append((extra, "none", 0.0, budget))  # different model family
    if category in ("math", "logic", "code_debug"):
        # A genuinely different mode, not a reroll: reasoning ON, larger budget.
        attempts.append((chain[0], None, 0.3, max(budget * 3, 4000)))

    requirement = None
    for i, (model, effort, temp, cap) in enumerate(attempts):
        if i == 1 and requirement is None:
            continue  # first answer validated: the re-ask slot is skipped
        ask = prompt if not requirement else f"{prompt}\n\nRequirement: {requirement}."
        try:
            answer = _call(client, task_id, model, system, ask, cap, i,
                           temperature=temp, effort=effort)
        except Exception:
            continue  # transport failure: walk the chain
        problem = validate(category, prompt, answer)
        if answer:
            best = answer
        if problem is None:
            return answer
        requirement = problem
    return best

# -------------------------------------------------------------------- main --


def run_simple(input_path="/input/tasks.json", output_path="/output/results.json",
               max_workers=None, per_task_max_tokens=None):
    max_workers = int(os.environ.get("WORKERS", max_workers or 6))
    deadline_s = float(os.environ.get("DEADLINE_S", "510"))
    started = time.monotonic()

    answers: dict[str, str] = {}
    tasks = _read(input_path, answers)
    _write(output_path, answers)
    if not tasks:
        return 0

    total = len(tasks)
    solver_hits = 0

    # Deterministic solvers: exact answers at zero tokens, no API needed.
    if os.environ.get("SOLVERS", "1") != "0":
        from .solvers import solve_any

        for tid, prompt in tasks:
            try:
                hit = solve_any(prompt)
            except Exception:
                hit = None
            if hit is not None:
                answers[tid] = hit[0]
                solver_hits += 1
        if solver_hits:
            _write(output_path, answers)
        tasks = [(tid, p) for tid, p in tasks if not answers.get(tid)]

    key = os.environ.get("FIREWORKS_API_KEY")
    base = os.environ.get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
    allowed = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]

    if tasks and key and allowed:
        from .backends.fireworks import normalize_base_url  # noqa
        from .classify import classify as _classify
        from .tasks import Task

        client = _client(key, normalize_base_url(base))
        gen_model = _pick(allowed, _GENERAL_MODELS) or allowed[0]
        code_model = _pick(allowed, _CODE_MODELS) or gen_model
        extra_model = _pick(allowed, _OPPORTUNISTIC)

        def chain_for(category: str) -> list[str]:
            # Known-available models carry the chain; the opportunistic tier
            # is a last resort only. No model appears twice.
            if category in ("code_gen", "code_debug"):
                chain = [code_model, gen_model]
            else:
                chain = [gen_model, code_model]
            if extra_model and extra_model not in chain:
                chain.append(extra_model)
            return [m for m in chain if m]

        def work(task):
            tid, prompt = task
            try:
                category = _classify(Task(id=tid, input=prompt))
            except Exception:
                category = "factual"
            try:
                answers[tid] = _solve_task(client, tid, prompt, category,
                                           chain_for(category))
            except Exception as exc:
                print(f"task {tid} unhandled: {type(exc).__name__}: {exc}", file=sys.stderr)
            _write(output_path, answers)

        # Watchdog: results are flushed and the process exits 0 before the
        # wall no matter what wedges. Unanswered ids are logged loudly.
        def _watchdog():
            time.sleep(max(5.0, started + deadline_s - time.monotonic()))
            empty = [t for t, a in answers.items() if not a.strip()]
            if empty:
                print(f"watchdog: {len(empty)} unanswered at the wall: {empty}",
                      file=sys.stderr)
            _write(output_path, answers)
            _write_ledger(output_path, total, solver_hits, started)
            os._exit(0)

        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(work, tasks))
        except Exception as exc:
            print(f"pool error: {type(exc).__name__}: {exc}", file=sys.stderr)
    elif tasks:
        print("missing FIREWORKS_API_KEY or ALLOWED_MODELS", file=sys.stderr)

    _write(output_path, answers)
    _write_ledger(output_path, total, solver_hits, started)
    return 0


def _write_ledger(output_path, total, solver_hits, started) -> None:
    summary = {
        "tasks": total,
        "solver_answered": solver_hits,
        "model_calls": len(LEDGER),
        "prompt_tokens": sum(e["prompt_tokens"] for e in LEDGER),
        "completion_tokens": sum(e["completion_tokens"] for e in LEDGER),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    print(json.dumps(summary), file=sys.stderr)
    try:
        Path(output_path).with_name("inference_log.json").write_text(
            json.dumps({"summary": summary, "calls": LEDGER}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass


def _read(input_path, answers):
    try:
        raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read {input_path}: {type(exc).__name__}", file=sys.stderr)
        return []
    tasks, seen = [], set()
    for i, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            print(f"skipping malformed task record #{i}", file=sys.stderr)
            continue
        tid = str(item.get("task_id", f"task-{i}"))
        if tid in seen:
            print(f"duplicate task_id {tid!r}: keeping the first occurrence",
                  file=sys.stderr)
            continue
        seen.add(tid)
        answers[tid] = ""
        tasks.append((tid, str(item.get("prompt", ""))))
    return tasks


_write_lock = threading.Lock()


def _write(output_path, answers):
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results = [{"task_id": k, "answer": v} for k, v in answers.items()]
        with _write_lock:
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
    except Exception as exc:
        print(f"write failed (will retry on next flush): {exc}", file=sys.stderr)
