# frugal-router

Track 1 entry for the AMD Developer Hackathon Act II (Hybrid Token-Efficient
Routing Agent).

## The scored path

The submitted image runs `frugal simple` — one consolidated pipeline:

```
/input/tasks.json
      |
  deterministic solvers      prove-or-defer; a hit is exact by construction
      |                      and costs zero tokens
  classify (free)            8 published categories, priority-ordered regexes
      |
  contract call              per-category instruction + budget, reasoning
      |                      suppressed; a generation cut at max_tokens is
      |                      retried once with a doubled budget
  validate                   category acceptance checks: final numeric value,
      |                      one sentiment label, entity lines, summary length,
      |                      fenced parseable code, explicit yes/no
      |-- pass (math/logic)  confirmed by a reasoning-mode second opinion;
      |                      disagreement -> cross-model tiebreak, majority's
      |                      own text is emitted (answers are never rewritten)
      |-- fail               corrective re-ask -> other model family ->
      |                      opportunistic tier, deadline-aware
      |
/output/results.json + /output/inference_log.json (per-call ledger)
```

Model policy: chains are carried by the known-available chat models resolved
from `ALLOWED_MODELS` at runtime (non-chat entries are filtered); the Gemma
tiers are opportunistic last-resort fallbacks, never load-bearing. A watchdog
flushes results and exits 0 before the 10-minute wall; unanswered ids are
logged loudly, never hidden.

## Rules notes (as clarified by organizers during the event)

- Local inference and deterministic solvers are legal and count as zero tokens.
- The accuracy gate is 80% (16 of the 19 fixed tasks); passers are ranked
  ascending by total tokens at the judging proxy.
- The LLM judge is not perfectly deterministic run to run, so a submission
  must be re-saved to re-evaluate and single scores carry variance.

## Develop and test

```bash
python -m venv .venv && .venv/bin/pip install -e .[ml,dev]
PYTHONPATH=src .venv/bin/python -m pytest -q          # full suite
./scripts/smoke_image.sh <image:tag>                  # real Docker image e2e
python scripts/gen_eval.py --n-per-cat 25             # labeled eval set
python scripts/run_eval2.py --backend fw --judge      # per-category diagnostics
```

The historical local-model / confidence-router machinery in this repository
(`configs/`, the `harness`, `eval`, `train-predictor`, and `sweep`
subcommands) is development history, not part of the scored image; in the
image those subcommands fail loudly instead of producing blank output.

## Submission

See `SUBMISSION.md` for the current image tag, the tag taxonomy, and the
operational protocol (re-save each cycle, confirm the registry pull counter
moved, judge configs by their lower-bound score across draws).
