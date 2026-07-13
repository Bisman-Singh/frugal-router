#!/usr/bin/env python3
"""Run the repeatable 19-task local-image rehearsal at the judge envelope.

This is a development smoke, not a hidden-benchmark score.  It proves that
the real onv29-layered image starts and writes complete output with no egress
under the same 2-vCPU / 4-GB resource limits as the judge.  Its fixture also
includes solver-uncovered math plus code_gen/code_debug examples so those
execution-grounded paths are attempted by the baked local model.

Build first, then run, for example:
    docker build -f Dockerfile.onv29 -t frugal-zerogaps:rehearsal .
    python scripts/rehearse_onv29.py --image frugal-zerogaps:rehearsal

The script exits non-zero for an incomplete result, a non-zero remote-token
ledger, or a run that exceeds its outer wall timeout.  It prints measured wall
time and local-lane acceptance counts; it deliberately does not claim accuracy
for the model-backed prompts.  Pass ``--report`` to retain the final result
outside the temporary Docker mounts (useful for long-running CI/terminal jobs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "rehearsal_local_19.json"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc


def _validate_fixture(tasks: object) -> list[dict]:
    if not isinstance(tasks, list) or len(tasks) != 19:
        raise RuntimeError("the rehearsal fixture must contain exactly 19 tasks")
    ids = [task.get("task_id") for task in tasks if isinstance(task, dict)]
    if len(ids) != 19 or any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise RuntimeError("every rehearsal task needs a non-empty task_id")
    if len(set(ids)) != len(ids):
        raise RuntimeError("rehearsal task ids must be unique")
    return tasks


def _write_report(path: Path | None, payload: dict) -> None:
    """Best-effort durable result for a run whose temporary mounts are removed."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"warning: could not write report {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="already-built Docker image tag")
    parser.add_argument("--timeout-s", type=float, default=660,
                        help="outer wall timeout in seconds (default: 660)")
    parser.add_argument("--report", type=Path,
                        help="write a durable JSON result report at this path")
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")

    def finish(status: str, message: str, elapsed_s: float | None = None,
               **extra: object) -> int:
        report = {
            "status": status,
            "message": message,
            "image": args.image,
            "limits": {"cpus": 2, "memory": "4g", "memory_swap": "4g",
                       "network": "none"},
            **extra,
        }
        if elapsed_s is not None:
            report["elapsed_s"] = round(elapsed_s, 1)
        _write_report(args.report, report)
        print(f"REHEARSAL {status.upper()}: {message}",
              file=sys.stderr if status == "fail" else sys.stdout)
        return 0 if status == "pass" else 1

    try:
        tasks = _validate_fixture(_read_json(FIXTURE))
    except RuntimeError as exc:
        return finish("fail", str(exc))
    with tempfile.TemporaryDirectory(prefix="frugal-rehearsal-") as td:
        root = Path(td)
        inp, out = root / "input", root / "output"
        inp.mkdir()
        out.mkdir()
        (inp / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")

        command = [
            "docker", "run", "--rm",
            "--cpus", "2",
            "--memory", "4g",
            "--memory-swap", "4g",
            "--network", "none",
            "-e", "LOCAL=1",
            "-e", "FULL_LOCAL=0",
            "-e", "FIREWORKS_API_KEY=",
            "-e", "ALLOWED_MODELS=",
            "-v", f"{inp}:/input:ro",
            "-v", f"{out}:/output",
            args.image,
        ]
        started = time.monotonic()
        try:
            run = subprocess.run(command, check=False, text=True,
                                 timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            return finish("fail", f"exceeded outer {args.timeout_s:g}s timeout",
                          time.monotonic() - started)
        elapsed = time.monotonic() - started
        if run.returncode:
            return finish("fail", f"docker exited {run.returncode}", elapsed,
                          docker_returncode=run.returncode)

        try:
            results = _read_json(out / "results.json")
            ledger = _read_json(out / "inference_log.json")
        except RuntimeError as exc:
            return finish("fail", str(exc), elapsed)
        by_id = {row.get("task_id"): row.get("answer") for row in results
                 if isinstance(row, dict)} if isinstance(results, list) else {}
        wanted = {task["task_id"] for task in tasks}
        if set(by_id) != wanted or any(not isinstance(by_id[task_id], str) or
                                       not by_id[task_id].strip() for task_id in wanted):
            return finish("fail", "results must contain one non-empty answer per task",
                          elapsed, result_rows=len(by_id),
                          nonempty_answers=sum(bool(answer and answer.strip())
                                               for answer in by_id.values()))
        summary = ledger.get("summary", {}) if isinstance(ledger, dict) else {}
        remote = (summary.get("prompt_tokens", 0), summary.get("completion_tokens", 0))
        if remote != (0, 0):
            return finish("fail", f"expected zero remote tokens, got {remote}", elapsed,
                          summary=summary)
        calls = ledger.get("calls", []) if isinstance(ledger, dict) else []
        lanes = Counter(call.get("model") for call in calls if isinstance(call, dict))
        lane_text = ", ".join(f"{lane}={count}" for lane, count in sorted(lanes.items()))
        _write_report(args.report, {
            "status": "pass",
            "message": "19/19 non-empty answers and zero remote tokens",
            "image": args.image,
            "limits": {"cpus": 2, "memory": "4g", "memory_swap": "4g",
                       "network": "none"},
            "elapsed_s": round(elapsed, 1),
            "summary": summary,
            "local_lane_acceptances": dict(sorted(lanes.items())),
        })
        print(f"REHEARSAL PASS: 19/19 non-empty, 0 remote tokens, {elapsed:.1f}s wall")
        print(f"local lane acceptances: {lane_text or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
