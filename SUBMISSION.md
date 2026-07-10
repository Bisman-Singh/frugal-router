# Submission checklist (lablab.ai, Track 1)

Deadline: Saturday, July 11, 2026, 16:00 UTC. Submissions are rate-limited to
10/hour per team. The accuracy gate is 80% (16 of 19 fixed tasks); passers are
ranked ascending by total tokens at the judging proxy. The LLM judge is not
perfectly deterministic, so a submission must be re-saved to re-evaluate, and
scores from a single run carry roughly one task of variance.

## Docker image (mandatory field)

Gate baseline (accuracy-first, the current submission):
```
docker.io/bismansinghmadaan/frugal-router:v22
```
The image is public, linux/amd64, ~160MB, remote-only (no local weights), and
runs `frugal simple` — solvers first, one contract call per task, validation
with escalation on failure, watchdog flush before the 10-minute wall, and an
inference ledger at /output/inference_log.json.

Tag taxonomy (honest purposes):
- `gate-baseline` (v22+): accuracy-first on known-available models.
- `token-optimized`: only after the gate passes repeatedly.
- `local-candidate`: only if a build actually bakes local weights.

## Operational protocol

1. Bind the image tag, save the submission, and confirm the registry pull
   counter moved before interpreting the next score.
2. Re-save every scoring cycle; treat single scores statistically and judge a
   config by its lower bound across draws, not its best draw.
3. Freeze the submission the moment a draw passes the gate.
