# Video presentation script and storyboard

Target length 2 to 3 minutes. Record with QuickTime, Loom, or OBS. Upload to
YouTube or Loom and paste the link into the lablab "Video Presentation" field.

Structure: talking head or voiceover over the slide deck
(`docs/frugal-router-slides.pdf`), with one live terminal demo in the middle.
Narration is written to be read aloud as is. It avoids jargon dumps and keeps
each beat short.

---

## Segment 1 — Hook (0:00 to 0:20)
On screen: Slide 1 (title), then Slide 2 (the rule).

> "This is Frugal Router, my entry for Track 1 of the AMD Developer Hackathon.
> The track has one rule that decides everything. Pass the accuracy gate, then
> win on the fewest Fireworks tokens. So accuracy above the bar is wasted money.
> The entire design is about landing just above the bar at the lowest possible
> cost."

## Segment 2 — The insight (0:20 to 0:45)
On screen: Slide 3 (not a routing problem).

> "The organizers confirmed that every scored answer has to come from a
> Fireworks call. That changes the problem. It is not about choosing between a
> local model and a remote model. Every answer is remote. The real question is
> how cheap I can make each mandatory call. So I let a free local model do the
> thinking, and Fireworks do the answering."

## Segment 3 — Architecture (0:45 to 1:15)
On screen: Slide 4 (pipeline), then Slide 5 (confidence stack).

> "Here is the pipeline. Each task is classified into one of eight categories
> for free. A local model drafts an answer and measures its own confidence with
> self-consistency voting and a log probability signal. When it is confident, I
> send a compact confirmation call that costs about ten tokens. When it is not,
> I pay for a full remote solve, and only then. The confidence signals are the
> ones that actually hold up on small models. I dropped self verification
> because small models say yes to almost everything."

## Segment 4 — Live demo (1:15 to 1:55)
On screen: your terminal. Run the demo block below live.

> "Here it is running the real submission path. This is the container the judges
> run, reading a task file and writing answers, calling Fireworks through the
> harness environment."

Run this on camera (key already exported, or paste it):

```bash
cd frugal-router
printf '[{"task_id":"a","prompt":"What is 15 percent of 240?"},{"task_id":"b","prompt":"What is the sentiment of this review? Justify in one sentence: I loved the crisp display but the battery dies by noon."},{"task_id":"c","prompt":"Write a Python function named add(a, b) that returns their sum."}]' > /tmp/in/tasks.json

docker run --rm \
  -e FIREWORKS_API_KEY -e ALLOWED_MODELS \
  -v /tmp/in:/input:ro -v /tmp/out:/output \
  frugal-router

cat /tmp/out/results.json
```

> "Three tasks, three correct answers, a valid results file, and a clean exit.
> The math is right, the sentiment answer carries its justification, and the
> code task returns a runnable function."

## Segment 5 — Results and token engineering (1:55 to 2:30)
On screen: Slide 6 (token engineering), then Slide 8 (results).

> "On a live evaluation across all eight categories the agent reached full
> accuracy while spending under two hundred tokens per task. Capping reasoning
> effort alone cut completion tokens by about a third with no loss of accuracy.
> Remote calls run in parallel so a large batch clears the ten minute budget.
> Nothing is cached or hardcoded, no credentials ship in the image, and it
> always writes a valid results file even under a timeout."

## Segment 6 — Gemma and close (2:30 to 2:50)
On screen: Slide 9 (Gemma), then Slide 11 (close).

> "Gemma runs on both sides. It is the preferred remote model and the
> recommended local drafting model, so one model family carries the free path
> and the paid path. That is Frugal Router. Free intelligence, paid answers,
> minimum tokens. Thank you."

---

## Recording checklist
- Deck open at `docs/frugal-router-slides.pdf` in full screen
- Terminal font large, dark theme, window clean of other tabs
- Export `FIREWORKS_API_KEY` and `ALLOWED_MODELS` before recording so the demo
  is one clean run
- Do a silent dry run of the demo once so the Docker image is warm and the call
  is fast on camera
- Keep it under three minutes, upload unlisted or public, paste the link

## Optional B-roll
- The passing test suite: `.venv/bin/pytest -q`
- The token breakdown from a dev eval run, showing tokens per category
