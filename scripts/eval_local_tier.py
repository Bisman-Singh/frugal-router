#!/usr/bin/env python3
"""GO/NO-GO measurement for the local tier on the real GGUF.

Runs the exact shipped gate (_try_local) over the generated eval set's four
local categories and reports, per category: how many the tier KEPT, how many
kept answers are CORRECT (confident-but-wrong is the only accuracy risk),
how many escalated, and generation speed.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from frugal_router import simple                      # noqa: E402
from run_eval2 import grade                           # noqa: E402

tasks_path = sys.argv[2] if len(sys.argv) > 2 else "data/eval_gen.jsonl"
tasks = [json.loads(l) for l in Path(tasks_path).read_text().splitlines()]
from frugal_router import local_tier as _lt
tasks = [t for t in tasks if t["category"] in _lt.CATEGORIES]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if limit:
    by = {}
    for t in tasks:
        by.setdefault(t["category"], []).append(t)
    tasks = [t for c in sorted(by) for t in by[c][: limit // 4]]

wall = time.monotonic() + 10**9
stats = {}
t_start = time.monotonic()
for i, t in enumerate(tasks, 1):
    cat = t["category"]
    s = stats.setdefault(cat, {"n": 0, "kept": 0, "kept_ok": 0})
    s["n"] += 1
    a = simple._try_local(t["id"], cat, t["prompt"], wall)
    if a is not None:
        s["kept"] += 1
        s["kept_ok"] += bool(grade(t, a))
    if i % 10 == 0:
        print(f"  {i}/{len(tasks)}", file=sys.stderr)

print(f"\n{'category':<15}{'n':>4}{'kept':>6}{'kept-ok':>9}{'CBW':>6}{'esc%':>6}")
tot_kept = tot_ok = 0
for cat, s in sorted(stats.items()):
    cbw = s["kept"] - s["kept_ok"]
    tot_kept += s["kept"]; tot_ok += s["kept_ok"]
    print(f"{cat:<15}{s['n']:>4}{s['kept']:>6}{s['kept_ok']:>9}{cbw:>6}"
          f"{100 * (1 - s['kept'] / s['n']):>5.0f}%")
print(f"kept accuracy: {tot_ok}/{tot_kept}  |  confident-but-wrong: {tot_kept - tot_ok}")
print(f"elapsed: {time.monotonic() - t_start:.0f}s for {len(tasks)} tasks")
