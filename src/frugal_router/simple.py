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

# Per-category caps accept env overrides so a trimmed image is a build-arg
# flip, not a code change: BUDGET_CODE (code_gen/code_debug), BUDGET_REASON
# (math/logic), BUDGET_GENERAL (everything else).
_BUDGET_ENV = {"code_gen": "BUDGET_CODE", "code_debug": "BUDGET_CODE",
               "math": "BUDGET_REASON", "logic": "BUDGET_REASON"}


def _contract(category: str) -> tuple[str, int]:
    system, cap = CONTRACTS.get(category, CONTRACTS["factual"])
    env = _BUDGET_ENV.get(category, "BUDGET_GENERAL")
    override = os.environ.get(env)
    if override:
        try:
            cap = min(cap, int(override)) if int(override) > 0 else cap
        except ValueError:
            pass
    return system, cap


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


_POT_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_POT_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _run_pot(code: str, timeout: float = 6.0) -> str | None:
    """Execute a tiny generated program in a bare sandbox; last printed number."""
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        out = subprocess.run(["python3", "-I", path], capture_output=True,
                             text=True, timeout=timeout, env={})
        os.unlink(path)
        nums = _POT_NUM.findall(out.stdout or "")
        return nums[-1] if nums else None
    except Exception:
        return None


def _try_math_pot(task_id, prompt, wall):
    """Program-of-thought math: the local model writes a program, WE execute
    it, and the result must agree with the model's independent direct answer.
    Execution makes the arithmetic exact; agreement makes it trustworthy.
    Any mismatch, parse failure, or timeout defers to remote."""
    from . import local_tier

    if time.monotonic() > wall - 240:
        return None
    if _LOCAL_SPENT["s"] > float(os.environ.get("LOCAL_TIME_BUDGET", "300")):
        return None
    if not local_tier.available():
        return None
    t0 = time.monotonic()
    try:
        gen = local_tier.generate(
            "Write a short Python 3 program that computes the answer and prints "
            "ONLY the final numeric value. No comments, no explanation, one "
            "fenced code block.", prompt, 220)
        m = _POT_FENCE.search(gen or "")
        if not m:
            return None
        value = _run_pot(m.group(1))
        if value is None:
            return None
        system, _ = CONTRACTS["math"]
        direct = local_tier.generate(system, prompt, local_tier.CAPS.get("math", 120) if hasattr(local_tier, "CAPS") else 120)
        d = _ANSWER_LINE.search(direct or "")
        direct_val = None
        if d:
            nums = _POT_NUM.findall(d.group(1))
            direct_val = nums[-1] if nums else None
        if direct_val is None:
            nums = _POT_NUM.findall(direct or "")
            direct_val = nums[-1] if nums else None
        if direct_val is None or abs(float(value) - float(direct_val)) > 1e-6:
            return None                      # no independent agreement -> remote
        v = float(value)
        v = int(v) if v.is_integer() else v
        answer = f"Computed by executing a program.\nAnswer: {v}"
        if validate("math", prompt, answer) is not None:
            return None
        _record(task_id, "math", "local-pot", "ok", 0, None, "stop",
                (time.monotonic() - t0) * 1000)
        return answer
    finally:
        _LOCAL_SPENT["s"] += time.monotonic() - t0


def _try_code_exec(task_id, prompt, wall):
    """code_gen at 0 tokens ONLY when a local generation passes tests extracted
    from the prompt. No tests / fail / timeout -> defer to remote unchanged."""
    from . import code_verify, local_tier
    if not local_tier.available() or time.monotonic() > wall - 240:
        return None
    if _LOCAL_SPENT["s"] > float(os.environ.get("LOCAL_TIME_BUDGET", "300")):
        return None
    t0 = time.monotonic()
    try:
        code = code_verify.verify_code_gen(
            prompt, local_tier.generate,
            cap=int(os.environ.get("BUDGET_CODE", "400")))
    except Exception:
        code = None
    finally:
        _LOCAL_SPENT["s"] += time.monotonic() - t0
    if code and validate("code_gen", prompt, code) is None:
        _record(task_id, "code_gen", "local-exec", "ok", 0, None, "stop",
                (time.monotonic() - t0) * 1000)
        return code
    return None


def _try_local(task_id, category, prompt, wall):
    """Answer via the baked local model when every gate agrees; None escalates.
    Gates: category eligibility, wall margin, a cumulative local-time budget
    (a slow box can never starve the remote fallback), format validation,
    label agreement across two samples (sentiment), and self-verification."""
    from . import local_tier

    # Provable zero-token tiers first: spaCy spans verified verbatim against
    # the source; math via executed-program + direct-answer agreement.
    if category == "ner":
        from . import ner_local
        spans = ner_local.extract(prompt)
        if spans is not None and validate("ner", prompt, spans) is None:
            _record(task_id, "ner", "local-spacy", "ok", 0, None, "stop", 0)
            return spans
        # fall through to the model tier below
    if category == "math":
        return _try_math_pot(task_id, prompt, wall)
    if category == "code_gen":
        return _try_code_exec(task_id, prompt, wall)

    if category not in local_tier.CATEGORIES:
        return None
    # Factual splits into two populations: concept/definition explanations,
    # which a small model answers reliably, and trivia lookups (who/when/
    # where/which + names and dates), where it hallucinates confidently and
    # self-approves. Measured on real trivia: half the kept answers were
    # wrong. Only the explanation style stays local.
    if category == "factual":
        if re.search(r"(?i)\b(who|when|where|which|whom)\b", prompt) or \
                not re.match(r"(?i)\s*(explain|describe|define|what is|what are|how do|how does|why)", prompt.strip()):
            return None
    # Logic stays local ONLY for explicit yes/no-style questions, where the
    # answer space is binary and the validator demands a committed label;
    # ordering/constraint puzzles escalate (solvers or remote).
    if category == "logic" and not re.search(r"(?i)\b(yes or no|true or false)\b", prompt):
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
    if category in ("sentiment", "logic"):
        # agreement gate: a second sample at a different temperature must
        # land on the same label, or the task escalates
        vocab = ("positive", "negative", "neutral") if category == "sentiment" \
            else ("yes", "no", "true", "false")
        second = local_tier.generate(system, prompt, cap, temperature=0.6)
        labels = []
        for text in (answer, second):
            found = [l for l in vocab if re.search(rf"\b{l}\b", text.lower())]
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
    system, budget = _contract(category)
    primary_val = _final_value(category, answer)
    if primary_val is None or time.monotonic() > wall - 60:
        return answer

    # CONFIRM modes: "reason" (default) = reasoning-mode second opinion on the
    # answering model - most thorough, but hidden reasoning is billed in full;
    # "cheap" = an effort-none second opinion from the other model family -
    # an agreement check at a fraction of the cost; "off" = trust validation.
    mode = os.environ.get("CONFIRM", "reason")
    if mode == "off":
        return answer
    try:
        if mode == "cheap":
            other = next((m for m in chain if m != answered_by), answered_by)
            confirm = _call(client, task_id, category, other, system, prompt,
                            budget, attempt=90)
        else:
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
    system, budget = _contract(category)
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

# --------------------------------------------------------- full-local mode --

_FORCE_SUFFIX = (" You must commit to your single best final answer now. "
                 "Never reply UNSURE.")


def _solve_local_only(task_id, category, prompt, wall):
    """Lane B runner: every category answered locally. Abstention is a pure
    point leak when no escalation target exists, so UNSURE or a failed
    validation triggers a forced-answer re-ask; the emitted answer is always
    a committed attempt, never UNSURE, never empty (unless the model is)."""
    from . import local_tier

    system, _ = _contract(category)
    cap = local_tier.CAPS.get(category, 160) if category in local_tier.CATEGORIES \
        else {"math": 220, "logic": 220, "code_gen": 400, "code_debug": 400}.get(category, 160)

    candidates = []
    plan = [(system, prompt, 0.0)]
    for i, (sys_p, ask, temp) in enumerate(plan):
        if time.monotonic() > wall - 20:
            break
        answer = local_tier.generate(sys_p, ask, cap, temperature=temp)
        unsure = answer.strip().upper().startswith("UNSURE")
        bad = validate(category, prompt, answer) if not unsure else "committed answer required"
        if answer and not unsure:
            candidates.append(answer)
        if not unsure and bad is None:
            _record(task_id, category, "local", "ok", i, None, "stop", 0)
            return answer
        if i == 0 and time.monotonic() < wall - 30:
            # one forced retry, then one diverse sample
            plan.append((system + _FORCE_SUFFIX, prompt, 0.0))
            plan.append((system + _FORCE_SUFFIX, prompt, 0.6))
    for c in candidates:  # best committed attempt, validated or not
        if c.strip():
            _record(task_id, category, "local", "forced", 99, None, "stop", 0)
            return c
    return ""


_BATCHABLE = ("sentiment", "ner", "logic", "factual", "math")
_ITEM_HDR = re.compile(r"(?im)^\s*#{0,3}\s*ITEM\s+(\d+)\s*[:.)\-]?\s*")


def _solve_batch(client, items, category, model, wall):
    """One grouped remote call for same-category SHORT-prompt tasks: numbered
    items in, '### ITEM n'-delimited answers out. Each block is validated
    individually; anything missing or invalid falls back to the solo path.
    Amortizes the per-call system/framing overhead across items."""
    system, _ = _contract(category)
    n = len(items)
    sys_b = (system + f" You will receive {n} numbered items. Answer EVERY item. "
             "Start each answer with '### ITEM <number>' on its own line, then "
             "give the answer in the required format.")
    user = "\n\n".join(f"### ITEM {i + 1}\n{p}" for i, (_t, p) in enumerate(items))
    cap = min(120 * n + 100, 1800)
    out: dict[str, str] = {}
    text = _call(client, f"batch-{category}", category, model, sys_b, user, cap, 0)
    if not text:
        return out
    blocks = _ITEM_HDR.split(text)
    # split-with-capture: [pre, '1', block1, '2', block2, ...]
    for j in range(1, len(blocks) - 1, 2):
        try:
            idx = int(blocks[j]) - 1
        except ValueError:
            continue
        if 0 <= idx < n:
            tid, prompt = items[idx]
            ans = blocks[j + 1].strip()
            if ans and validate(category, prompt, ans) is None:
                out[tid] = ans
    return out


def _best_guess(prompt: str) -> str:
    """Absolute last resort when no tier produced anything: a formatted,
    non-empty default. For classification categories the majority label carries
    real nonzero expected accuracy; an empty cell is a guaranteed zero."""
    p = prompt.lower()
    if "sentiment" in p or "positive or negative" in p:
        return "Neutral."
    if re.search(r"\byes or no\b|\btrue or false\b", p):
        return "Yes."
    return "Unknown."


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
    all_prompts = dict(tasks)          # original tid->prompt for the local safety net
    _write(output_path, answers)
    if not tasks:
        return 0

    total = len(tasks)
    solver_hits = 0

    # Watchdog first: EVERYTHING after this line (solvers, the local tier,
    # remote calls) runs under the wall. A wedged local generation before the
    # remote branch used to run unguarded - that is a TIMEOUT verdict.
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

    # Local tier runs as a bounded SEQUENTIAL pre-pass (like the solvers):
    # cheapest categories first, wall margin and the cumulative time budget
    # re-checked between tasks, and the remote pool afterwards gets full
    # parallelism for everything the tier skipped or rejected. Locals must
    # never queue inside pool workers - that serializes the whole run.
    if tasks and os.environ.get("LOCAL", "0") == "1":
        from . import local_tier
        from .classify import classify as _classify0
        from .tasks import Task as _Task0

        if local_tier.available():
            cheap_first = {"ner": 0, "sentiment": 1, "summarization": 2, "factual": 3}
            cats = {}
            for tid, prompt in tasks:
                try:
                    cats[tid] = _classify0(_Task0(id=tid, input=prompt))
                except Exception:
                    cats[tid] = "factual"
            handled = 0
            for tid, prompt in sorted(tasks, key=lambda t: cheap_first.get(cats[t[0]], 9)):
                # math rides the pre-pass too (program-of-thought, executed +
                # agreement-gated); spaCy ner is cheapest of all (no LLM).
                if cats[tid] not in local_tier.CATEGORIES and cats[tid] not in ("math", "code_gen"):
                    continue
                try:
                    local = _try_local(tid, cats[tid], prompt, wall)
                except Exception as exc:
                    print(f"task {tid} local tier: {type(exc).__name__}", file=sys.stderr)
                    local = None
                if local is not None:
                    answers[tid] = local
                    handled += 1
                    _write(output_path, answers)
            if handled:
                print(f"local tier answered {handled} task(s) in "
                      f"{_LOCAL_SPENT['s']:.0f}s", file=sys.stderr)
            tasks = [(tid, p) for tid, p in tasks if not answers.get(tid)]

    # Lane B: FULL_LOCAL=1 answers every remaining task with the baked model
    # and never touches the network. Sequential (one llama context), deadline-
    # aware, and it never emits UNSURE - a committed answer for every task.
    if tasks and os.environ.get("FULL_LOCAL", "0") == "1":
        from . import local_tier
        from .classify import classify as _classify1
        from .tasks import Task as _Task1

        if local_tier.available():
            for tid, prompt in tasks:
                try:
                    category = _classify1(_Task1(id=tid, input=prompt))
                except Exception:
                    category = "factual"
                try:
                    answers[tid] = _solve_local_only(tid, category, prompt, wall)
                except Exception as exc:
                    print(f"task {tid} full-local: {type(exc).__name__}", file=sys.stderr)
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

        # Grouped calls first (BATCH=1): short-prompt categories share one call
        # per ~8 items; validated hits are kept, the rest fall to the solo pool.
        if os.environ.get("BATCH", "0") == "1":
            groups: dict[str, list] = {}
            for tid, prompt in tasks:
                try:
                    c = _classify(Task(id=tid, input=prompt))
                except Exception:
                    c = "factual"
                if c in _BATCHABLE:
                    groups.setdefault(c, []).append((tid, prompt))
            for c, items in groups.items():
                if len(items) < 2:
                    continue
                for s in range(0, len(items), 8):
                    if time.monotonic() > wall - 90:
                        break
                    chunk = items[s:s + 8]
                    try:
                        got = _solve_batch(client, chunk, c, chain_for(c)[0], wall)
                    except Exception as exc:
                        print(f"batch {c}: {type(exc).__name__}", file=sys.stderr)
                        got = {}
                    answers.update(got)
                    if got:
                        _write(output_path, answers)
            tasks = [(tid, p) for tid, p in tasks if not answers.get(tid)]

        def work(task):
            tid, prompt = task
            try:
                category = _classify(Task(id=tid, input=prompt))
            except Exception:
                category = "factual"
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

    # Final safety net: any task still blank -- empty ALLOWED_MODELS, remote
    # failures, or a category every tier skipped -- gets a forced local answer
    # instead of a guaranteed-zero blank. Local is 0-token and strong on
    # code/logic/math; a committed guess always beats an empty cell.
    if os.environ.get("LOCAL_FALLBACK", "1") == "1":
        blanks = [(tid, all_prompts[tid]) for tid in all_prompts
                  if not answers.get(tid, "").strip()]
        if blanks:
            from . import local_tier
            if local_tier.available():
                from .classify import classify as _classifyf
                from .tasks import Task as _Taskf
                print(f"local fallback: forcing {len(blanks)} unanswered task(s)",
                      file=sys.stderr)
                for tid, prompt in blanks:
                    if time.monotonic() > wall - 8:
                        break
                    try:
                        category = _classifyf(_Taskf(id=tid, input=prompt))
                    except Exception:
                        category = "factual"
                    try:
                        forced = _solve_local_only(tid, category, prompt, wall)
                        if forced and forced.strip():
                            answers[tid] = forced
                            _write(output_path, answers)
                    except Exception as exc:
                        print(f"task {tid} local-fallback: {type(exc).__name__}",
                              file=sys.stderr)

    # Absolute guarantee: never hand the judge a blank cell. Anything still
    # empty (total tier failure, local load/OOM) gets a formatted best-guess.
    for tid in all_prompts:
        if not answers.get(tid, "").strip():
            answers[tid] = _best_guess(all_prompts[tid])

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
