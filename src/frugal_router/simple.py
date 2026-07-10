"""The scored path: accuracy-first gate baseline.

This module is what the submitted image runs — one consolidated pipeline, no
dead branches. Design rules, in order:

1. Known-available CHAT models only in the critical path (non-chat entries in
   ALLOWED_MODELS are filtered out). The Gemma tiers are opportunistic
   last-resort fallbacks, never load-bearing.
2. Deterministic solvers first: prove-or-defer, exact by construction.
3. One primary call per task with a category contract, reasoning suppressed;
   a generation cut at max_tokens is retried once with a doubled budget.
4. VALIDATE the answer. Checks are correctness-bearing where a regex can be:
   an explicit final value for math/logic, entity lines for NER, a fenced
   block for code, length constraints for summaries. Well-shaped answers to
   the hard categories are additionally CONFIRMED by a second, reasoning-mode
   opinion; a disagreement goes to a cross-model tiebreak and the majority's
   full text is emitted. Answers are never rewritten.
5. Escalation on validated failure walks: corrective re-ask -> other model
   family -> opportunistic tier, bounded and deadline-aware per task.
6. Never exit non-zero; never blow the 10-minute wall; every request lands in
   a run-scoped inference ledger with category, duration, and finish reason.
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
_OPPORTUNISTIC = ["gemma-4-31b-it", "gemma"]  # never load-bearing
# Entries in ALLOWED_MODELS that cannot serve chat completions at all.
_NON_CHAT = re.compile(r"(?i)flux|image|audio|whisper|embed|tts|guard|moderat|rerank|ocr|video")

_FENCE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")
_UNCLOSED_THINK = re.compile(r"(?i)<(?:think|thought)>(?!.*</(?:think|thought)>)", re.DOTALL)
_TRAILING_CLOSE = re.compile(r"(?is)^.*?</(?:think|thought)>\s*")
_ANSWER_LINE = re.compile(r"(?im)^\s*\**answer\**\s*[:=]\s*(.+?)\s*$")
_NER_LINE = re.compile(r"(?im)^\s*\**\s*(person|organization|location|date)s?\s*\**\s*[:=]\s*\S")
_NON_ANSWER = re.compile(r"(?i)^\s*(i (do not|don't) know|unclear|it is unclear|cannot determine|"
                         r"not enough information|as an ai)")

# ----------------------------------------------------------------- ledger ---

LEDGER: list[dict] = []
_ledger_lock = threading.Lock()


def _record(task_id, category, model, status, attempt, usage, finish, dur_ms) -> None:
    entry = {"task_id": task_id, "category": category, "model": model,
             "status": status, "attempt": attempt, "finish_reason": finish,
             "duration_ms": int(dur_ms),
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
    if _NON_ANSWER.match(a):
        return "commit to a concrete answer to the task"

    if category == "math":
        m = _ANSWER_LINE.search(a)
        if not m or not re.search(r"-?\d[\d,]*(?:\.\d+)?", m.group(1)):
            return "end with 'Answer: <numeric value>' on its own line"

    elif category == "sentiment":
        labels = {l for l in ("positive", "negative", "neutral")
                  if re.search(rf"\b{l}\b", low)}
        if len(labels) != 1:
            return ("state exactly one sentiment label - positive, negative, or "
                    "neutral - followed by a brief justification")
        if len(a.split()) < 4:
            return "add one short sentence justifying the label"

    elif category == "ner":
        if len(_NER_LINE.findall(a)) < 2:
            return ("list each entity as 'label: value' on its own line using the "
                    "labels person, organization, location, date")

    elif category == "summarization":
        if len(a.split()) < 5:
            return "write a complete summary sentence"
        m = re.search(r"(?i)\bin\s+(one|a single|two|three|\d+)\s+sentences?\b", prompt)
        if m:
            word = m.group(1).lower()
            want = {"one": 1, "a single": 1, "two": 2, "three": 3}.get(word)
            want = want if want is not None else int(word)
            sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", a) if s.strip()]
            if len(sentences) > want:
                return f"the summary must be at most {want} sentence(s)"
        m = re.search(r"(?i)\bin\s+(\d+)\s+words?\b", prompt)
        if m and len(a.split()) > int(m.group(1)) * 1.15:
            return f"the summary must be at most {m.group(1)} words"

    elif category in ("code_gen", "code_debug"):
        block = _FENCE.search(a)
        if block is None:
            return "provide the complete code in one fenced code block"
        code = block.group(1)
        is_python = bool(re.search(r"(?i)\bpython\b", prompt)) or bool(
            re.search(r"^\s*(def |import |from |class )", code, re.MULTILINE))
        if is_python:
            try:
                ast.parse(code)
            except SyntaxError:
                return "provide complete, syntactically valid code in one fenced block"

    elif category == "logic":
        if re.search(r"(?i)\b(yes or no|true or false)\b", prompt):
            m = _ANSWER_LINE.search(a)
            scope = (m.group(1) if m else a).lower()
            found = {w for w in ("yes", "no", "true", "false")
                     if re.search(rf"\b{w}\b", scope)}
            # Without an explicit Answer line, a bare fragment ("no idea") is
            # not a committed answer - demand substance alongside the label.
            if len(found) != 1 or (not m and len(a.split()) < 4):
                return ("commit to exactly one final answer: end with "
                        "'Answer: yes' or 'Answer: no' on its own line")
        elif not _ANSWER_LINE.search(a):
            return "end with 'Answer: <value>' on its own line"

    elif category == "factual":
        if len(a) < 15:
            return "answer the question in one to three complete sentences"

    return None


def _final_value(category: str, text: str) -> str | None:
    """Extract the final value for cross-attempt AGREEMENT comparison only
    (never for emission). Math -> last number on the Answer line / tail;
    logic -> yes/no/true/false or the Answer-line value."""
    a = (text or "").strip()
    if not a:
        return None
    m = _ANSWER_LINE.search(a)
    scope = m.group(1) if m else a[-120:]
    if category == "math":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", scope.replace("$", "").replace("%", ""))
        if not nums:
            return None
        try:
            return f"{float(nums[-1].replace(',', '')):g}"
        except ValueError:
            return None
    yn = re.findall(r"(?i)\b(yes|no|true|false)\b", scope)
    if yn:
        return yn[-1].lower()
    return re.sub(r"[^a-z0-9]+", "", scope.lower()) or None

# ------------------------------------------------------------------ client --


def _client(key, base):
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base,
                  timeout=float(os.environ.get("TIMEOUT_S", "35")), max_retries=1)


def _pick(allowed, hints):
    for h in hints:
        for m in allowed:
            if h.lower() in m.lower():
                return m
    return ""


def _call(client, task_id, category, model, system, prompt, max_tokens, attempt,
          temperature=0.0, effort="none"):
    """One completion, reasoning suppressed by default. A generation cut at
    max_tokens is retried once with a doubled budget. Returns cleaned text
    ('' for an all-thought reply). Raises on transport errors."""
    cap = max_tokens
    for round_ in range(2):
        kwargs = dict(model=model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": prompt}],
                      temperature=temperature, max_tokens=cap)
        if effort:
            kwargs["extra_body"] = {"reasoning_effort": effort}
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if effort and any(s in msg for s in ("reasoning_effort", "unsupported",
                                                 "unknown parameter", "extra")):
                kwargs.pop("extra_body", None)
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    _record(task_id, category, model, f"error:{type(exc2).__name__}",
                            attempt, None, None, (time.monotonic() - t0) * 1000)
                    raise
            else:
                _record(task_id, category, model, f"error:{type(exc).__name__}",
                        attempt, None, None, (time.monotonic() - t0) * 1000)
                raise
        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        _record(task_id, category, model, "ok", attempt,
                getattr(resp, "usage", None), finish, (time.monotonic() - t0) * 1000)
        text = (choice.message.content or "").strip()
        if finish == "length" and round_ == 0:
            cap = cap * 2  # cut mid-generation: one retry with real room
            continue
        if _UNCLOSED_THINK.search(text):
            return ""  # all-thought reply: no answer in this response
        text = _THINK.sub("", text).strip()
        if re.search(r"(?i)</(?:think|thought)>", text):
            text = _TRAILING_CLOSE.sub("", text).strip()
        return text
    return ""

# -------------------------------------------------------------- local tier --


_LOCAL_SPENT = {"s": 0.0}


def _try_local(task_id, category, prompt, wall):
    """Answer via the baked local model when every gate agrees; None escalates.
    Gates: category eligibility, wall margin, a cumulative local-time budget
    (a slow box can never starve the remote fallback), format validation,
    label agreement across two samples (sentiment), and self-verification."""
    from . import local_tier

    if category not in local_tier.CATEGORIES:
        return None
    # Local generation is serialized and slow on the judge box: only start it
    # while there is comfortably enough wall left for the remote fallback too.
    if time.monotonic() > wall - 240:
        return None
    if _LOCAL_SPENT["s"] > float(os.environ.get("LOCAL_TIME_BUDGET", "300")):
        return None
    if not local_tier.available():
        return None

    system, _ = CONTRACTS[category]
    cap = local_tier.CAPS[category]
    t0 = time.monotonic()
    answer = local_tier.generate(system, prompt, cap)
    if not answer or validate(category, prompt, answer) is not None:
        return None
    if category == "sentiment":
        # agreement gate: a second sample at a different temperature must
        # land on the same label, or the task escalates
        second = local_tier.generate(system, prompt, cap, temperature=0.6)
        labels = []
        for text in (answer, second):
            found = [l for l in ("positive", "negative", "neutral") if l in text.lower()]
            labels.append(found[0] if len(found) == 1 else None)
        if labels[0] is None or labels[0] != labels[1]:
            return None
    if not local_tier.verify(prompt, answer):
        _LOCAL_SPENT["s"] += time.monotonic() - t0
        return None
    dur = time.monotonic() - t0
    _LOCAL_SPENT["s"] += dur
    _record(task_id, category, "local", "ok", 0, None, "stop", dur * 1000)
    return answer


# ------------------------------------------------------------------- solve --


def _confirm_hard_answer(client, task_id, category, prompt, chain, answer,
                         answered_by, wall):
    """Math/logic answers that LOOK right still get one reasoning-mode second
    opinion — run on the model that actually produced the answer (the head of
    the chain may be the model that just failed). Agreement -> keep the
    primary text. Disagreement -> tiebreak from a DIFFERENT healthy model; the
    majority's own full text is emitted, never a rewrite."""
    system, budget = CONTRACTS[category]
    primary_val = _final_value(category, answer)
    if primary_val is None or time.monotonic() > wall - 60:
        return answer

    try:
        confirm = _call(client, task_id, category, answered_by, system, prompt,
                        max(budget * 3, 4000), attempt=90, effort=None)
    except Exception:
        return answer
    confirm_val = _final_value(category, confirm)
    if confirm_val is None or confirm_val == primary_val:
        return answer

    # Disagreement: a third opinion from a different model family breaks it.
    tiebreak, tiebreak_val = "", None
    others = [m for m in chain if m != answered_by]
    if others and time.monotonic() < wall - 45:
        try:
            tiebreak = _call(client, task_id, category, others[0], system, prompt,
                             budget, attempt=91)
            tiebreak_val = _final_value(category, tiebreak)
        except Exception:
            pass
    if tiebreak_val == primary_val:
        return answer
    if tiebreak_val == confirm_val and validate(category, prompt, confirm) is None:
        return confirm
    # No majority: prefer the reasoning-mode answer when it validates.
    if validate(category, prompt, confirm) is None:
        return confirm
    return answer


def _solve_task(client, task_id, prompt, category, chain, wall):
    """Contract call -> validate -> corrective re-ask -> other families, all
    deadline-aware. The first non-empty answer is the fallback: a later failed
    retry never replaces an earlier one."""
    system, budget = CONTRACTS.get(category, CONTRACTS["factual"])
    first_nonempty = ""

    plan = [(chain[0], budget), (chain[0], budget)]          # primary + corrective
    plan += [(m, budget) for m in chain[1:]]                  # other families
    requirement = None
    for i, (model, cap) in enumerate(plan):
        if i == 1 and requirement is None:
            continue  # primary validated: corrective slot unused
        # Margin: a request can cost timeout x (1 + SDK retry). The first
        # attempt gets a slimmer allowance (something beats a guaranteed
        # blank); later attempts must clear the full worst case.
        remaining = wall - time.monotonic()
        if remaining < (40 if i == 0 else 80):
            break     # leave room for the flush; a partial beats a blank batch
        ask = prompt if not requirement else f"{prompt}\n\nRequirement: {requirement}."
        try:
            answer = _call(client, task_id, category, model, system, ask, cap, i)
        except Exception:
            continue
        problem = validate(category, prompt, answer)
        if answer and not first_nonempty:
            first_nonempty = answer
        if problem is None:
            if category in ("math", "logic"):
                return _confirm_hard_answer(client, task_id, category, prompt,
                                            chain, answer, model, wall)
            return answer
        requirement = problem
    return first_nonempty

# -------------------------------------------------------------------- main --


def run_simple(input_path="/input/tasks.json", output_path="/output/results.json",
               max_workers=None, per_task_max_tokens=None):
    max_workers = int(os.environ.get("WORKERS", max_workers or 6))
    deadline_s = float(os.environ.get("DEADLINE_S", "510"))
    started = time.monotonic()
    wall = started + deadline_s
    LEDGER.clear()  # run-scoped audit trail
    _LOCAL_SPENT["s"] = 0.0

    answers: dict[str, str] = {}
    tasks = _read(input_path, answers)
    _write(output_path, answers)
    if not tasks:
        return 0

    total = len(tasks)
    solver_hits = 0

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
    allowed_all = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    allowed = [m for m in allowed_all if not _NON_CHAT.search(m)]
    if allowed_all and not allowed:
        # Fail closed: calling a non-chat model is a guaranteed failure that
        # can also invalidate the submission. Solver answers stand; the rest
        # are preserved (and loudly logged) rather than burned on a bad call.
        print(f"ERROR: no chat-capable model in ALLOWED_MODELS={allowed_all}",
              file=sys.stderr)

    if tasks and key and allowed:
        from .backends.fireworks import normalize_base_url  # noqa
        from .classify import classify as _classify
        from .tasks import Task

        client = _client(key, normalize_base_url(base))
        general_hints = [h.strip() for h in os.environ.get(
            "GENERAL_HINTS",
            "minimax,kimi-k2p7-code,kimi,gemma-4-31b-it,gemma").split(",") if h.strip()]
        gen_model = _pick(allowed, general_hints) or allowed[0]
        code_model = _pick(allowed, _CODE_MODELS) or gen_model
        extra_model = _pick(allowed, _OPPORTUNISTIC)

        def chain_for(category: str) -> list[str]:
            if category in ("code_gen", "code_debug"):
                chain = [code_model, gen_model]
            else:
                chain = [gen_model, code_model]
            if extra_model:
                chain.append(extra_model)
            deduped: list[str] = []
            for m in chain:
                if m and m not in deduped:
                    deduped.append(m)
            return deduped

        def work(task):
            tid, prompt = task
            try:
                category = _classify(Task(id=tid, input=prompt))
            except Exception:
                category = "factual"
            if os.environ.get("LOCAL", "0") == "1":
                try:
                    local = _try_local(tid, category, prompt, wall)
                except Exception as exc:
                    print(f"task {tid} local tier: {type(exc).__name__}", file=sys.stderr)
                    local = None
                if local is not None:
                    answers[tid] = local
                    _write(output_path, answers)
                    return
            try:
                answers[tid] = _solve_task(client, tid, prompt, category,
                                           chain_for(category), wall)
            except Exception as exc:
                print(f"task {tid} unhandled: {type(exc).__name__}: {exc}", file=sys.stderr)
            _write(output_path, answers)

        def _watchdog():
            time.sleep(max(5.0, wall - time.monotonic()))
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
            print(f"WARNING: skipping malformed task record #{i}: {item!r}", file=sys.stderr)
            continue
        tid = str(item.get("task_id", f"task-{i}"))
        if tid in seen:
            print(f"WARNING: duplicate task_id {tid!r}: keeping the first occurrence",
                  file=sys.stderr)
            continue
        prompt = str(item.get("prompt", ""))
        if not prompt.strip():
            print(f"WARNING: task {tid!r} has an empty prompt", file=sys.stderr)
        seen.add(tid)
        answers[tid] = ""
        tasks.append((tid, prompt))
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
