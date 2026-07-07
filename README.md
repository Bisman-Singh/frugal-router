# frugal-router

A confidence-driven routing agent for the AMD Developer Hackathon Act II,
Track 1 (Hybrid Token-Efficient Routing Agent). Organizers clarified during
the event that every scored answer must come from a Fireworks call through
FIREWORKS_BASE_URL; local inference is free but cannot be the answer source.
So the local model is the intelligence, not the mouthpiece. It classifies,
drafts, and measures its own confidence for free, and that confidence decides
how cheap the mandatory Fireworks call gets: a trusted draft rides along and
the remote model confirms it in a handful of tokens, while an untrusted task
gets a full remote solve with reasoning. The original zero-token local mode
survives behind `router.answer_source: local` in case the ruling moves again.

## Scoring rule and strategy

Track 1 scores in two stages. An LLM judge grades each answer against expected
intent, and submissions below a hidden accuracy threshold are excluded. The
survivors are ranked ascending by total tokens recorded by the judging proxy.
That asymmetry drives every design choice here. Failing the gate is total loss,
overspending tokens only costs rank, so answers are intent-complete rather than
minimal, and remote calls happen exactly once per escalated task.

## Architecture

```
/input/tasks.json
      |
  classify (free)          8 published categories, heuristics on the prompt
      |
  scheduler                wall clock is the binding constraint, not tokens:
      |                    full -> greedy -> remote_direct as budget burns
  local attempt (free)     llama.cpp GGUF, adaptive self-consistency voting,
      |                    early stop on unanimity
  confidence gate          agreement + pessimistic logprob quantile
      |                    (+ optional learned failure predictor)
      +-- confident -----> ONE cheap Fireworks call: draft rides along,
      |                    remote confirms in ~10 output tokens (no CoT bill)
      |
      +-- not confident -> ONE full Fireworks call    model resolved from
              |            (CoT where the category     ALLOWED_MODELS at runtime
              |             needs it)
              +---------> emit judge-shaped answer
                           (malformed output repaired locally, never re-asked)
/output/results.json       every task_id answered, always, exit 0
```

The confidence stack is deliberately small. Self-consistency agreement is the
primary signal (the strongest cheap correctness predictor for small models),
an answer-span logprob quantile is the tiebreaker, and yes/no self-verification
is intentionally absent because small instruct models rate almost everything
"yes". A learned failure predictor (TF-IDF plus logistic regression, trained on
the local model's actual mistakes) can be added per category once eval records
exist, and stays out of the loop until then.

## Quickstart

```bash
# 1. Install (Python 3.10+)
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev,local]"

# 2. Get a local model (swap repo/file for the model you settle on)
bash scripts/download_model.sh

# 3. Set the API key for development
cp .env.example .env  # then edit; or export FIREWORKS_API_KEY=...

# 4. Solve one task
.venv/bin/frugal solve --input "What is 15 percent of 200?"

# 5. Run the dev evaluation
.venv/bin/frugal eval --dataset data/dev_tasks.jsonl --out runs/latest
```

### Container (submission form)

```bash
bash scripts/download_model.sh                 # bake the GGUF into the image
docker buildx build --platform linux/amd64 -t <registry>/frugal-router:latest .
docker push <registry>/frugal-router:latest

# Local dry run of the judging contract
mkdir -p io/input io/output
echo '[{"task_id":"t1","prompt":"What is 15 percent of 200?"}]' > io/input/tasks.json
docker run --rm \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  -v "$PWD/io/input:/input:ro" -v "$PWD/io/output:/output" \
  <registry>/frugal-router:latest
cat io/output/results.json
```

The container reads `/input/tasks.json` (`[{"task_id","prompt"}]`), writes
`/output/results.json` (`[{"task_id","answer"}]`), and exits 0. It never
crashes and never writes partial output. If everything inside fails, every
task still gets an answer entry.

## Tuning loop

The router is an afternoon, the tuning loop is the week. Signals are collected
once, then thresholds are swept offline at zero token cost.

```bash
# 1. Collect signals plus remote counterfactuals (bills dev tokens once)
.venv/bin/frugal eval --dataset data/dev_tasks.jsonl --out runs/collect --collect-remote

# 2. Sweep thresholds offline and pick the operating point
.venv/bin/frugal sweep --records runs/collect/records.jsonl --target-acc 0.85 --margin 0.05

# 3. Optional: train the per-category failure predictor from the same records
.venv/bin/frugal train-predictor --records runs/collect/records.jsonl
```

Pick the operating point conservatively. The accuracy threshold is unpublished,
so the sweep targets comfortable headroom above any plausible gate, and a
held-out set you never tune on estimates real generalization.

## Configuration

Everything model-specific lives in `configs/default.yaml`: the local GGUF path,
per-category sample counts, escalation thresholds, remote `max_tokens`, and
`remote_model_hints` (substrings resolved against the runtime `ALLOWED_MODELS`
list, never hardcoded model IDs). The judging harness injects
`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`; all three are
read from the environment at run time and override the config file.

## Compliance notes

- Every scored answer originates from a Fireworks response
  (`router.answer_source: fireworks`, the default). The local model only
  classifies, drafts, compresses, and repairs formatting, per the organizer
  clarification that all scoring inference goes through Fireworks.
- No answers are hardcoded or cached anywhere. Every response is generated
  fresh per run, as the event rules require.
- All remote calls go through `FIREWORKS_BASE_URL` with the harness-provided
  key. Retries fire only on transport failures (max 1), because a retried
  request that reached the proxy is billed twice.
- No `.env`, keys, or credentials are baked into the image.
- Model IDs come from `ALLOWED_MODELS` at run time.

## Gemma usage

The default `remote_model_hints` prefer Gemma variants from `ALLOWED_MODELS`
for every non-code category, and the recommended local model is Google's
Gemma 3 4B QAT Q4_0 GGUF, making the escalation path and the free local path
both Gemma. See `configs/default.yaml`.

## Tests

```bash
.venv/bin/pytest -q   # 66 tests, all offline via mock backends
```

## License

Apache-2.0
