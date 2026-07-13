# LOCAL-ZERO gaps plan (v2 — re-scoped against main @ b67669a)

The original from-scratch M1–M6 plan is **superseded**: `main` independently
built the local-zero architecture (llama-cpp-python `local_tier.py`, PoT math,
execution-verified `code_verify.py`, spaCy NER, broad math solvers,
`Dockerfile.local`/`Dockerfile.onv29`, a full LoRA training kit). This branch
(`feat/zero-gaps`, cut from `origin/main`) adds only the **four genuine gaps**
main does not yet cover, delivered as a thin source layer.

## Non-negotiable constraints inherited from main's hard-won experience

1. **~60 s readiness contract on the judge.** Heavy cold-start deps are
   banned (spaCy's venv unpack "broke the 60 s readiness contract — the
   smoking gun", so it is coded but disabled in the image). Everything here is
   **stdlib-only** (sandbox, CSP solver, code_debug) or dev-only (eval).
2. **From-scratch image rebuilds fail on the judge.** The proven deploy path
   is `Dockerfile.onv29`, which layers ONLY `src/` on the published
   `bismansinghmadaan/frugal-router:v29` image. Nothing here may add a runtime
   dependency — it must import and run under the frozen `v29-freeze.txt` env.
3. **Network audit → DQ.** Any egress that is not Fireworks disqualifies. The
   hardened sandbox (Gap A) exists to guarantee model-written code cannot open
   a socket even by accident.
4. **prove-or-defer** for solvers; **verify-or-defer** for local lanes. A
   wrongly accepted answer costs the accuracy gate; a wrongly rejected one
   costs a few remote tokens. Always err toward defer/escalate.
5. `frugal simple` stays the single scored entrypoint. New lanes hang off the
   existing `LOCAL=1` pre-pass and env flags; do not fork a parallel pipeline.
6. Full suite stays green (`141 passed` baseline); no test skips added.

## Gaps (build order)

### Gap A — Hardened sandbox (`src/frugal_router/sandbox.py`, `tests/test_sandbox.py`)

Today `simple._run_pot` and `code_verify._run` cage model code with
`python3 -I` + `env={}` but **no rlimits and no socket block** — a
model-authored program that opens a socket is a live DQ risk under the network
audit. Provide one hardened primitive and route both existing callers through
it (no behavioural change to their happy paths).

API (stdlib only):
```python
@dataclass
class RunResult:
    stdout: str; stderr: str; returncode: int; timed_out: bool

def run_python(code: str, *, timeout_s: float = 6.0, memory_mb: int = 512,
               argv: list[str] | None = None) -> RunResult
```
Requirements: `-I`; `preexec_fn` rlimits (RLIMIT_CPU, RLIMIT_AS, RLIMIT_NOFILE,
RLIMIT_FSIZE) — guarded so it degrades to no-rlimits on platforms without
`resource` (macOS dev is fine; Linux judge gets the limits); a prepended
**prelude** that replaces `socket.socket` and `_socket.socket` with a raiser
before user code runs; `env={}`, cwd=tmp; stdout/stderr capped at 64 KB; temp
cleanup; **never raises** — every failure returns a RunResult.

Then: `code_verify._run` and `simple._run_pot` call `sandbox.run_python`.
Preserve their existing return contracts (`code_verify._run` → bool on
`ALLPASS`; `_run_pot` → last printed number or None). The socket prelude must
not perturb ordinary arithmetic programs.

**Accept:** tests prove (a) happy-path stdout, (b) timeout → `timed_out=True`
in < timeout+2 s, (c) a `socket.socket()`/connect attempt fails inside the
sandbox, (d) a memory bomb dies without taking down the test process,
(e) output flood truncated to the cap, (f) the two rewired callers still pass
their existing tests. Full suite green.

### Gap B — CSP brute-force logic solver (`src/frugal_router/solvers.py`, tests)

`_LOGIC_SOLVERS` is still `(_syllogism, _ordering)` — no constraint-puzzle
enumerator, the highest-value logic archetype ("each person has a different X;
clues; who has Y?"). Add `_assignment_csp` (stdlib `itertools.permutations`):

- Parse N named entities and one attribute set of size N (e.g. people ↔
  {red, blue, green}); support the common pairwise clue forms: "A has/is/likes
  X", "A does not have X", "A's X is not B's" style negations, and
  "the one with X is X2" links across two attributes when ≤ 5×5.
- Enumerate assignments; keep only those satisfying **every parsed clue**.
- Emit an answer ONLY if (i) exactly one assignment survives AND (ii) every
  clue sentence in the prompt was consumed by the parser (same completeness
  guard `_ordering` uses — an unparsed clue means the model of the puzzle is
  incomplete and nothing is proven). Otherwise return None (defer).
- Bound: ≤ 5 entities × ≤ 5 attributes (≤ 120 perms). Anything larger defers.

**Accept:** positive tests on classic small puzzles; **paraphrased-variant**
tests (names/attributes/order randomized) proving it's parametric, not
memorized; defer tests (unparsed clue, >5 entities, multiple solutions,
mixed dimensions) all → None. Extend `test_solvers_adversarial.py`. Green.

### Gap C — Execution-grounded code_debug lane (`src/frugal_router/code_verify.py` or new `code_debug.py`, wired in `simple.py`)

`code_debug` currently goes fully remote. Add a zero-token local lane that is
execution-grounded, so the shipped cause line is observed, not guessed:

1. Extract the buggy snippet + any worked example/expected behaviour from the
   prompt (reuse `code_verify.extract_tests` + fenced-block extraction).
2. Run the snippet in `sandbox.run_python`; capture the real exception /
   wrong output. If it does not actually misbehave against the extracted
   expectation, defer (nothing to ground a fix on).
3. Ask the local model for the corrected code, prompted WITH the observed
   failure. Gate the fix: parses, runs clean, and reproduces the expected
   outputs. Emit "<observed error, one sentence>\n```python\n<fix>\n```".
4. Any missing evidence / failed gate → defer to remote unchanged.

Wire as `_try_code_debug(task_id, prompt, wall)` alongside `_try_code_exec`,
called from `_try_local` for `category == "code_debug"`, and add `code_debug`
to the pre-pass eligibility set. Same wall-margin / `LOCAL_TIME_BUDGET` guards.

**Accept:** unit test with a mock `generate_fn`: a snippet with a real bug +
an in-prompt example → the observed traceback appears in the model prompt, and
a correct mocked fix is accepted only after it executes green; a mocked fix
that still fails is rejected (deferred). No-evidence prompt → None. Green.

### Gap D — Archetype-variant eval (`scripts/gen_variants.py`, `run_eval2.py` wiring)

main evaluates on real benchmarks (`eval_bench.jsonl`, `eval_pool.jsonl`).
Add a generator for **randomized variants of the ~8 public task archetypes**
(the shape final scoring re-randomizes), so we can measure gate robustness of
the zero-token tiers specifically:

- Seeded RNG; per-archetype template with randomized numbers/names/entities/
  orderings and 2–3 surface phrasings; ≥ 10 variants per archetype, labeled
  with the deterministically-computed expected answer where one exists
  (math/logic/ner), else a rubric string.
- Emit JSONL compatible with `run_eval2.py`. Add a `--variants` flag (or a new
  small `scripts/eval_variants.py`) that runs solvers + the local tiers over
  the pool and reports a per-category exact-match / gate table. Judge calls
  stay dev-only and clearly outside the scored path.

**Accept:** generator is deterministic under a fixed seed and re-randomizes
under a new one; produces ≥ 80 labeled tasks across categories; the eval
prints a per-category table; committed sample output in the progress log.
No new runtime dependency (dev-only script). Green.

## Reviewer verification checklist

1. `PYTHONPATH=src .venv/bin/python -m pytest -q` — green, ≥ 141 + new tests,
   no skips added.
2. Sandbox: a `socket`-opening payload through `sandbox.run_python` fails;
   `_run_pot` and `code_verify._run` still pass their existing tests.
3. CSP solver: `git grep` shows no public-task literal/answer; variant tests
   actually randomize inputs; ambiguous/oversized puzzles defer.
4. code_debug lane: observed error is execution-derived (grep the prompt built
   in the test), fixes are execution-gated, no-evidence defers.
5. Variant eval reproduces within noise on a fresh seed; per-category table
   present.
6. No new entry in the runtime import graph beyond stdlib (`python -c "import
   frugal_router.simple, frugal_router.code_verify, frugal_router.solvers,
   frugal_router.sandbox"` under the base env); onv29-layerable.
7. `frugal simple` behaviour unchanged when `LOCAL=0` (existing scored-path
   tests still pass).

## Progress log (append below)

### Gap A — Hardened sandbox — DONE

- New `src/frugal_router/sandbox.py`: `run_python()` → `-I` isolated, `env={}`,
  cwd=tmp, POSIX rlimits (CPU/AS/DATA/FSIZE/NOFILE/NPROC, each best-effort),
  a prepended prelude neutering `socket.socket`/`_socket.socket`/
  `create_connection`, 64 KB stream caps, wall timeout, never raises. Uses
  `sys.executable` (not bare `python3`, which was unreliable under `env={}`).
- Rewired both existing callers through it with contracts preserved:
  `code_verify._run` (→ bool on `ALLPASS`) and `simple._run_pot` (→ last
  printed number or None). Dropped their ad-hoc `subprocess`/`tempfile` code.
- Tests: `tests/test_sandbox.py` (8) proves happy path, argv, prompt timeout
  (<3s), socket create + connect blocked, memory-bomb parent-survival, 64 KB
  truncation, garbage-never-raises. `tests/test_code_verify.py` (6) locks the
  rewired `_run`/`verify_code_gen` contract (had no test before).
- Evidence: `pytest -q` → **155 passed** (141 baseline + 14 new), 0 skips.
  `test_provable_tiers.py` (the `_run_pot` regression) green.
  `python -c "import frugal_router.sandbox, .code_verify, .simple"` → OK, no
  new runtime dependency (stdlib only; onv29-layerable).

### Gap B — CSP brute-force logic solver — DONE

- New `_assignment_csp` in `solvers.py`, added to `_LOGIC_SOLVERS`. Parses the
  bijective-assignment archetype ("N names each ... a different <noun>: v1, v2,
  v3", then pairwise pin/forbid clues, then a who/which question), enumerates
  the ≤ 5! permutations, and answers ONLY when exactly one assignment survives
  every clue AND every clue sentence parsed cleanly. Stdlib `itertools` only.
- prove-or-defer guards: requires the word "different" (bijection signal);
  defers on 2-name relational clues (unparsed), missing bijection signal,
  > 5 entities, multiple solutions, contradictions, name/value token overlap,
  or a second question.
- Fully parametric — no stored answers (grep audit: the only `return "yes"` is
  the pre-existing `_syllogism`, a computed conclusion).
- Tests: `tests/test_csp.py` (9): canonical pet puzzle + 3 randomized variants
  (names/values/order changed, "who" and "which" question forms, 4 entities) +
  5 defer cases. Extended `test_solvers_adversarial.py` with 4 CSP traps.
- Evidence: solver suites 27 passed; full suite **164 passed**, 0 skips.

### Gap C — Execution-grounded code_debug lane — DONE

- New `src/frugal_router/code_debug.py`: `verify_code_debug()` extracts the
  buggy fenced snippet + prompt examples (reusing `code_verify`), runs the
  snippet in the hardened sandbox to OBSERVE the real failure (raised
  exception / wrong return / parse error via `observe_failure` + `_probe`),
  prompts the model WITH that evidence, and keeps the fix only when it parses
  and passes every example (`code_verify._run`). Emits "Bug: <observed>.\n```
  python\n<fix>\n```". No evidence / no snippet / unreproducible / still-failing
  fix → None (defer). Stdlib-only.
- Wired into `simple.py`: `_try_code_debug` (same wall-margin / LOCAL_TIME_BUDGET
  guards as `_try_code_exec`), dispatched from `_try_local` for `code_debug`,
  and `code_debug` added to the pre-pass eligibility set.
- Tests: `tests/test_code_debug.py` (9): observe wrong-return / exception /
  syntax-error / correct-code-None; verify grounds the prompt with the observed
  failure and accepts a passing fix; rejects a still-failing fix; defers with no
  example (never calls the model) and with no snippet; `_try_local` wiring.
- Evidence: full suite **173 passed**, 0 skips. Import check: new modules are
  stdlib-only (onv29-layerable, no cold-start cost).

### Gap D — Archetype-variant eval — DONE

- New `scripts/gen_variants.py`: seeded generator for randomized variants of
  the 8 public task archetypes (math families, assignment-CSP + ordering logic,
  code_gen/code_debug with worked examples, sentiment/ner/summarization/
  factual). Ground truth is computed INDEPENDENTLY of the solvers (CSP puzzles
  built from a fixed random assignment, uniqueness enforced by adding positive
  pins), so it measures the code, not itself. `--per 12` → 96 labeled tasks.
- New `scripts/eval_variants.py`: runs the zero-token deterministic tiers
  (`solvers.solve_any`) over a variant file and prints a per-category
  coverage/accuracy table; flags model-tier categories (no deterministic path)
  and lists any solver miss (must be empty — a wrong zero-token answer costs
  the gate). Dev-only, stdlib + solvers; runs with no model/network.
- Measured (seed 7, and reproduced on seed 42):
  ```
  category          total  answered  correct   cover%    acc%
  logic                12        12       12     100%    100%
  math                 12        12       12     100%    100%
  (code_*/ner/sentiment/summarization/factual: 0% — model tier)
  DETERMINISTIC (math+logic): 24/24 answered (100% coverage), 24/24 correct
  ```
  So the deterministic layer alone carries ~25% of tasks (math+logic) at zero
  tokens with zero wrong answers under randomization. Model-backed lanes are
  measured with the baked GGUF via the image smoke (needs a box with the model
  + llama-cpp), out of scope for this model-less environment.
- Tests: `tests/test_variants.py` (5): generator determinism under seed,
  re-randomization on a new seed, shape (≥80 tasks, all 8 categories, grader
  types), the **safety test** (every math/logic solver hit matches the label
  across seeds 1/7/42/99 — prove-or-defer correctness under randomization),
  and full math+logic coverage at the default seed.
- Evidence: full suite **178 passed**, 0 skips. Committed `data/variants.jsonl`
  (seed 7) as a reference sample.

### Review pass (Fable) — hardening + fixes

- **Sandbox escape closed.** The socket monkeypatch was parent-process only; a
  model program could spawn a fresh interpreter (`subprocess`/`os.system`) with
  working sockets — on the Linux judge NPROC=256 permits that spawn. Prelude
  now also neuters `_posixsubprocess.fork_exec` + `os.{system,popen,fork,
  posix_spawn,exec*,spawn*}`. Verified: spawn/os.system escapes blocked, PoT
  arithmetic + code asserts still run. Docstring made honest: this is
  defence-in-depth; `--network none` is the primary guarantee. +2 tests.
- **Variant grammar fixed.** CSP verbs were past/base-tense ("ordered"→
  "ordereds"); switched to base form so "{verb}s" stays grammatical. Prompts
  now read correctly; eval unchanged (24/24, 100%/100%).
- Re-verified: full suite **180 passed**; deterministic eval reproduces
  24/24 at 100% coverage / 100% accuracy.

## Summary

All four gaps landed on `feat/zero-gaps` (off `origin/main`), full suite
**178 passed** (141 baseline + 37 new), every new module stdlib-only and
onv29-layerable, no new runtime dependency, no stored answers. Ready for
reviewer verification (checklist above).
