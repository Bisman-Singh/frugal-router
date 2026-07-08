# Submission checklist (lablab.ai, Track 1)

Deadline: Saturday, July 11, 2026, 16:00 UTC. Resubmission is 10/hour, and the
leaderboard ranks by tokens among submissions that pass the accuracy gate.

## Docker image (mandatory field)

Primary (aggressive, local-first, ~half the field's tokens):
```
docker.io/bismansinghmadaan/frugal-router:v4
```
Fallback if v4 returns ACCURACY_GATE_FAILED (safe, 100% accuracy):
```
docker.io/bismansinghmadaan/frugal-router:v3
```
Both are public, linux/amd64, under the 10GB limit (v4 2.2GB, v3 0.7GB).

## Basic information

- **Title**: `Frugal Router, a Token-Efficient AI Agent`
- **Short description**:
  > A general-purpose AI agent that answers eight task categories using the fewest Fireworks tokens possible. A small local model and deterministic solvers answer what they reliably can at zero tokens; only the hard cases escalate to Fireworks AI.
- **Long description**: see docs/deck.md story; full text in the git history of this file.
- **Event track**: Track 1 — Hybrid Token-Efficient Routing Agent (only that one)
- **Technologies**: Gemma, OpenAI (SDK), Qwen (local model), and add custom
  tag "Fireworks AI"; Docker/Python/llama.cpp if free-text tags are allowed.
- **Categories**: Developer Tools, Utility and Tools, Project FromScratch

## Application (Step 3)

- **GitHub Repository**: `https://github.com/Bisman-Singh/frugal-router`
- **Demo Application Platform**: Other
- **Demo Application URL**: `https://hub.docker.com/r/bismansinghmadaan/frugal-router`
- **Additional Information**:
  > Frugal Router is a containerized batch agent, not a hosted web app, so the demo is the public Docker image the judging harness runs.
  >
  > docker pull bismansinghmadaan/frugal-router:v4
  >
  > docker run --rm -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS -v ./in:/input:ro -v ./out:/output bismansinghmadaan/frugal-router:v4
  >
  > It reads /input/tasks.json, answers each task with a local model, deterministic solvers, or a Fireworks call as cheaply as possible, and writes /output/results.json. Local and deterministic answers cost zero Fireworks tokens. A narrated walkthrough is at docs/frugal-router-video.mp4 and slides at docs/frugal-router-slides.pdf.

## Media

- **Cover image**: docs/cover.png
- **Slides**: docs/frugal-router-slides.pdf
- **Video**: docs/frugal-router-video.mp4 — upload to YouTube/Loom, paste link.
  NOTE: the current deck/video describe the earlier Fireworks-first strategy;
  regenerate for the v4 local-first story if you want full consistency (matters
  mainly for the human-judged Gemma prize, not the leaderboard rank).

## Pre-submit verification (done 2026-07-08)

- Repo public, clean, synced, no secrets in history.
- v4 + v3 images public, anon-pullable, linux/amd64, under 10GB.
- v4 validated in 4GB/2vCPU container: 48-task batch in 424s, no OOM, all answered.
- Held-out (LLM-judged, same model for all): v4 100% @ 1785 tokens vs tokenopt
  3404, v3 3571.
