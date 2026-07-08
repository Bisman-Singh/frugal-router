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
_GENERAL_MODELS = ["gemma-4-31b-it", "gemma", "minimax"]


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
            return code_model, 2500
        if _REASON_HINT.search(prompt):
            # Reasoning model, reasoning left ON, room for thought + answer.
            return reason_model, 4000
        return gen_model, per_task_max_tokens

    def solve(task):
        tid, prompt = task
        model, budget = route(prompt)
        try:
            answers[tid] = _call(client, model, prompt, budget)
        except Exception as exc:
            print(f"task {tid} failed on {model}: {type(exc).__name__}", file=sys.stderr)
            # one fallback on the general model before giving up
            other = gen_model if model != gen_model else code_model
            try:
                answers[tid] = _call(client, other, prompt, budget)
            except Exception:
                pass
        if not (answers.get(tid) or "").strip():
            # a truncated/empty reply is a guaranteed zero; one retry on general
            try:
                answers[tid] = _call(client, gen_model, prompt, per_task_max_tokens)
            except Exception:
                pass
        _write(output_path, answers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(solve, tasks))

    _write(output_path, answers)
    print(json.dumps({"tasks": len(tasks), "answered": sum(1 for v in answers.values() if v),
                      "elapsed_s": round(time.monotonic() - started, 1)}), file=sys.stderr)
    return 0


def _client(key, base):
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base, timeout=28.0, max_retries=1)


_THINK = re.compile(r"(?s)<(?:think|thought)>.*?(?:</(?:think|thought)>|\Z)\s*")


def _call(client, model, prompt, max_tokens):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
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


def _write(output_path, answers):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = [{"task_id": k, "answer": v} for k, v in answers.items()]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
