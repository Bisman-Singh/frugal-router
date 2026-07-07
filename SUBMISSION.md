# Submission checklist (lablab.ai, Track 1)

Deadline: Saturday, July 11, 2026, 16:00 UTC. Submissions are rate-limited to
10 per hour per team, and the leaderboard ranks by tokens among submissions
that pass the accuracy gate, so submit a safe baseline early and iterate.

## Basic information

- **Project title**: frugal-router: a local-first cascade that spends remote
  tokens only when it must
- **Short description** (draft): A hybrid routing agent that answers with a
  free local Gemma model when self-consistency and logprob signals prove the
  answer trustworthy, and makes exactly one Fireworks call when they do not.
  Wall-clock aware, judge-shaped answers, zero cached responses.
- **Long description**: expand from README sections "Scoring rule and
  strategy" and "Architecture". Include the measured numbers from the final
  eval run (accuracy per category, escalation rate, total remote tokens) once
  the operating point is locked.
- **Tags**: AI Agents, Routing, llama.cpp, Fireworks AI, Gemma, Token
  Efficiency, Python, Docker

## Media

- **Cover image**: a diagram of the cascade (local attempt, confidence gate,
  single escalation). Keep the token counter visible.
- **Video presentation**: 2 to 3 minutes. Suggested cuts: the scoring rule and
  why cascade beats pre-routing (30s), live run of the container on a task
  file with the ledger output (60s), the sweep plot showing the chosen
  operating point on the accuracy/token frontier (30s), Gemma-everywhere
  story for the partner prize (20s).
- **Slides**: problem, scoring asymmetry, architecture diagram, confidence
  stack evidence, results table, compliance notes.

## Code and hosting

- **Public GitHub repository**: push this repo public. README already covers
  setup and usage (submission requirement). Verify no `.env`, no keys, no
  cluster data in history before flipping to public.
- **Demo**: the Docker image on a public registry is the deliverable the
  judging harness pulls. `docker buildx build --platform linux/amd64` is
  mandatory, the judging VM is linux/amd64 and a missing manifest scores zero.
- **Application URL**: the public registry image URL (plus the repo).

## Pre-submit verification (every submission)

1. `bash scripts/download_model.sh` so the GGUF is baked in. No downloads at
   startup, the container must be ready within 60 seconds.
2. `docker buildx build --platform linux/amd64 -t <registry>/frugal-router:vN .`
3. Compressed image size under 10GB (`docker images`).
4. Clean-machine pull test: `docker pull` then run with a sample
   `/input/tasks.json` and dummy env vars. Confirm `/output/results.json` is
   valid JSON, every task_id present, exit code 0.
5. Whole-batch runtime under 10 minutes on CPU with the expected task count.
6. Confirm no call bypasses `FIREWORKS_BASE_URL` and no model ID outside
   `ALLOWED_MODELS` is ever requested (both invalidate the submission).
7. Tokens telemetry printed at the end predicts the leaderboard number;
   reconcile after each live submission.
