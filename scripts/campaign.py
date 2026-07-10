#!/usr/bin/env python3
"""Robustness/cost campaign: many 19-task runs through the scored pipeline.

Each run gets the same envelope as a real evaluation (fresh process, the
shipped entrypoint function, per-run watchdog and local budget) and is graded
deterministically. The output is a DISTRIBUTION - accuracy and token spend per
run - because a submission is judged by its worst plausible draw, not its
best.

Usage:
  python scripts/campaign.py --pool data/eval_pool.jsonl --runs 48 \
      --parallel 3 --out runs/campaign
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_eval2 import grade  # noqa: E402

RUN_SNIPPET = """
import sys
sys.path.insert(0, {src!r})
from frugal_router.simple import run_simple
raise SystemExit(run_simple({inp!r}, {outp!r}))
"""


def one_run(idx: int, tasks: list[dict], args) -> dict:
    run_dir = Path(args.out) / f"run{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    inp, outp = run_dir / "tasks.json", run_dir / "results.json"
    inp.write_text(json.dumps(
        [{"task_id": t["id"], "prompt": t["prompt"]} for t in tasks]))

    env = dict(os.environ)
    env.update({
        "FIREWORKS_API_KEY": args.api_key,
        "ALLOWED_MODELS": args.models,
        "LOCAL": "1" if args.local else "0",
        "LOCAL_MODEL_PATH": str(Path("models/local.gguf.next").resolve()),
        "LOCAL_THREADS": "2",
        "DEADLINE_S": "510",
    })
    t0 = time.monotonic()
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = RUN_SNIPPET.format(src=src, inp=str(inp), outp=str(outp))
    try:
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=560)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -9  # a wedged run is a finding to record, never a campaign crash
    wall = time.monotonic() - t0

    results = {r["task_id"]: r["answer"]
               for r in json.loads(outp.read_text())} if outp.exists() else {}
    correct = empty = 0
    for t in tasks:
        a = results.get(t["id"], "")
        if not a.strip():
            empty += 1
        elif grade(t, a):
            correct += 1
    log = {}
    logp = outp.with_name("inference_log.json")
    if logp.exists():
        log = json.loads(logp.read_text()).get("summary", {})
    row = {
        "run": idx, "n": len(tasks), "correct": correct, "empty": empty,
        "acc": round(correct / max(1, len(tasks)), 3),
        "tokens": (log.get("prompt_tokens", 0) or 0) + (log.get("completion_tokens", 0) or 0),
        "local": None, "solver": log.get("solver_answered"),
        "wall_s": round(wall, 1), "exit": rc,
    }
    # count locally-answered from the full ledger
    if logp.exists():
        calls = json.loads(logp.read_text()).get("calls", [])
        row["local"] = sum(1 for c in calls if c.get("model") == "local")
    (run_dir / "metrics.json").write_text(json.dumps(row))
    print(f"  run{idx:03d}: acc={row['acc']:.2f} tokens={row['tokens']} "
          f"local={row['local']} solver={row['solver']} empty={empty} wall={row['wall_s']}s",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/eval_pool.jsonl")
    ap.add_argument("--runs", type=int, default=48)
    ap.add_argument("--batch", type=int, default=19)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--local", action="store_true", default=True)
    ap.add_argument("--no-local", dest="local", action="store_false")
    ap.add_argument("--out", default="runs/campaign")
    ap.add_argument("--api-key", default=os.environ.get("FIREWORKS_API_KEY", ""))
    ap.add_argument("--models", default=(
        "accounts/fireworks/models/deepseek-v4-pro,"
        "accounts/fireworks/models/gpt-oss-120b,"
        "accounts/fireworks/models/kimi-k2p6"))
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("FIREWORKS_API_KEY required")

    pool = [json.loads(l) for l in Path(args.pool).read_text().splitlines()]
    batches = []
    for i in range(args.runs):
        start = (i * args.batch) % len(pool)
        batch = (pool[start:] + pool[:start])[: args.batch]
        batches.append(batch)

    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool_exec:
        for row in pool_exec.map(lambda p: one_run(*p, args), enumerate(batches)):
            rows.append(row)

    rows.sort(key=lambda r: r["run"])
    accs = sorted(r["acc"] for r in rows)
    toks = sorted(r["tokens"] for r in rows)
    walls = sorted(r["wall_s"] for r in rows)
    n = len(rows)
    summary = {
        "runs": n,
        "acc_mean": round(sum(accs) / n, 3), "acc_min": accs[0],
        "acc_p10": accs[max(0, n // 10 - 1)],
        "runs_at_or_above_842": sum(1 for a in accs if a >= 0.842),
        "tokens_mean": round(sum(toks) / n), "tokens_max": toks[-1],
        "wall_mean_s": round(sum(walls) / n, 1), "wall_max_s": walls[-1],
        "empties_total": sum(r["empty"] for r in rows),
        "nonzero_exit": sum(1 for r in rows if r["exit"] != 0),
    }
    Path(args.out, "summary.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print("\n=== CAMPAIGN SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
