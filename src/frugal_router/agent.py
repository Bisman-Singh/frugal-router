"""The cascade agent: answer locally for free, escalate only on low confidence.

Fail safe, never silent: a local crash escalates, a remote crash falls back to
the best local answer. The agent always returns an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import prompts
from .classify import classify
from .compress import maybe_compress
from .confidence import ConfidenceReport, combine, logprob_quantile, vote
from .contracts import STYLE_LINE, style_of
from .extract import final_answer, is_valid_answer, vote_key
from .ledger import Ledger
from .policy import Policy, PolicyBook
from .tasks import Task


@dataclass
class SolveResult:
    task_id: str
    answer: str
    source: str  # local | remote | fallback
    task_type: str
    local_answer: str | None = None
    confidence: ConfidenceReport | None = None
    remote_prompt_tokens: int = 0
    remote_completion_tokens: int = 0
    decision_path: list[str] = field(default_factory=list)


class RoutingAgent:
    def __init__(
        self,
        local,
        remote,
        policies: PolicyBook,
        *,
        default_remote_model: str = "",
        allowed_models: list[str] | None = None,
        predictor=None,
        ledger: Ledger | None = None,
        weights: dict | None = None,
    ):
        self.local = local
        self.remote = remote
        self.policies = policies
        self.default_remote_model = default_remote_model
        self.allowed_models = allowed_models or []
        self.predictor = predictor
        self.ledger = ledger
        self.weights = weights

    def solve(self, task: Task, mode: str = "full") -> SolveResult:
        """mode is the scheduler's degradation lever: "full" allows voting,
        "greedy" forces a single local sample, "remote_direct" skips local."""
        path: list[str] = []
        task_type = classify(task)
        path.append(f"type:{task_type}")
        policy = self.policies.for_type(task_type)
        if mode == "greedy" and policy.n_samples > 1:
            policy = replace(policy, n_samples=1)
            path.append("scheduler:greedy")

        report = None
        local_answer = None
        if self.local is None:
            path.append("no_local_backend")
        elif policy.always_remote or mode == "remote_direct":
            path.append("skip_local:policy" if policy.always_remote else "skip_local:scheduler")
        else:
            try:
                report, local_answer = self._local_attempt(task, task_type, policy, path)
            except Exception as exc:
                path.append(f"local_error:{type(exc).__name__}")

        if report is not None and not self._should_escalate(report, policy, path):
            return self._finish(
                task, task_type, local_answer or "", "local", report, local_answer, 0, 0, path
            )

        pt = ct = 0
        try:
            answer, pt, ct = self._remote_answer(task, task_type, policy, path)
            if (answer or "").strip():
                return self._finish(
                    task, task_type, answer, "remote", report, local_answer, pt, ct, path
                )
            path.append("remote_empty")  # an empty answer must not beat a local one
        except Exception as exc:
            path.append(f"remote_error:{type(exc).__name__}")

        if not local_answer and self.local is not None:
            # Last-ditch free attempt: a low-confidence local answer still beats "".
            try:
                _, local_answer = self._local_attempt(
                    task, task_type, replace(policy, n_samples=1), path
                )
                path.append("late_local_attempt")
            except Exception:
                pass
        path.append("fallback")
        return self._finish(
            task, task_type, local_answer or "", "fallback", report, local_answer, pt, ct, path
        )

    # -- local -----------------------------------------------------------

    def _local_attempt(
        self, task: Task, task_type: str, policy: Policy, path: list[str]
    ) -> tuple[ConfidenceReport, str | None]:
        question = prompts.question_text(task.rendered_input(), task.context)
        system, user = prompts.local_solve(question, task_type, policy.few_shot)

        # Adaptive self-consistency: a small first window, extended to the full
        # sample budget only when the window disagrees. Unanimity stops early.
        window = min(policy.n_samples, 3)
        gens = self.local.generate(
            system,
            user,
            n=window,
            temperature=policy.sample_temperature if window > 1 else 0.0,
            max_tokens=policy.local_max_tokens,
        )
        keys = _vote_keys(gens, task_type)
        candidate, agreement = vote(keys)
        if agreement < 1.0 and policy.n_samples > window:
            extra = self.local.generate(
                system,
                user,
                n=policy.n_samples - window,
                temperature=policy.sample_temperature,
                max_tokens=policy.local_max_tokens,
            )
            gens += extra
            keys += _vote_keys(extra, task_type)
            candidate, agreement = vote(keys)

        best = _representative(gens, keys, candidate)
        answer = final_answer(best.text, task_type) if best else None
        report = ConfidenceReport(
            candidate=candidate,
            agreement=agreement,
            n_samples=len(gens),
            logprob=_best_logprob(gens, keys, candidate),
            p_fail=self.predictor.p_fail(task.input) if self.predictor else None,
            format_valid=is_valid_answer(answer, task_type),
        )
        report.score = combine(report, self.weights)
        path.append(f"local:n={len(gens)},agreement={agreement:.2f},score={report.score:.2f}")
        return report, answer

    @staticmethod
    def _should_escalate(report: ConfidenceReport, policy: Policy, path: list[str]) -> bool:
        if not report.format_valid:
            path.append("escalate:invalid_format")
            return True
        if report.p_fail is not None and report.p_fail > policy.p_fail_cutoff:
            path.append(f"escalate:p_fail={report.p_fail:.2f}")
            return True
        if report.score < policy.escalation_threshold:
            path.append(f"escalate:score={report.score:.2f}<{policy.escalation_threshold}")
            return True
        path.append("accept_local")
        return False

    # -- remote ----------------------------------------------------------

    def remote_answer(self, task: Task, task_type: str | None = None):
        """Direct remote query, used by the harness to collect counterfactuals.

        Returns (answer, prompt_tokens, completion_tokens).
        """
        task_type = task_type or classify(task)
        policy = self.policies.for_type(task_type)
        return self._remote_answer(task, task_type, policy, [])

    def _remote_answer(
        self, task: Task, task_type: str, policy: Policy, path: list[str]
    ) -> tuple[str, int, int]:
        if self.remote is None:
            raise RuntimeError("no remote backend configured")
        context = task.context
        if policy.allow_compression:
            context, compressed = maybe_compress(
                self.local, task.rendered_input(), task.context, policy.compress_over_chars
            )
            if compressed:
                path.append("compressed_context")
        question = prompts.question_text(task.rendered_input(), context)
        prompt = prompts.remote_solve(question, task_type, cot=policy.remote_cot)
        model = policy.remote_model or pick_model(
            self.allowed_models, policy.remote_model_hints, self.default_remote_model
        )
        if not model:
            raise RuntimeError("no remote model configured")

        gen = self.remote.generate(
            None, prompt, model=model, temperature=0.0, max_tokens=policy.remote_max_tokens
        )
        path.append(f"remote:{model}")
        if gen.finish_reason == "length":
            path.append("remote_truncated")  # billed in full, answer likely judge-failing
        answer = self._parse_remote(gen.text, task_type, path)
        return answer, gen.prompt_tokens, gen.completion_tokens

    def _parse_remote(self, text: str, task_type: str, path: list[str]) -> str:
        answer = final_answer(text, task_type)
        if (
            not is_valid_answer(answer, task_type)
            and answer
            and self.local is not None
            and style_of(task_type) == STYLE_LINE
        ):
            # Free local repair instead of a second billed call.
            try:
                system, user = prompts.reformat(answer, task_type)
                gens = self.local.generate(system, user, n=1, temperature=0.0, max_tokens=128)
                reshaped = final_answer(gens[0].text, task_type)
                if is_valid_answer(reshaped, task_type):
                    path.append("local_reformat")
                    answer = reshaped
            except Exception:
                pass
        return answer or (text or "").strip()

    def _finish(
        self, task, task_type, answer, source, report, local_answer, pt, ct, path
    ) -> SolveResult:
        result = SolveResult(
            task_id=task.id,
            answer=answer or "",
            source=source,
            task_type=task_type,
            local_answer=local_answer,
            confidence=report,
            remote_prompt_tokens=pt,
            remote_completion_tokens=ct,
            decision_path=path,
        )
        if self.ledger is not None:
            entry = {
                "task_id": result.task_id,
                "type": task_type,
                "answer": result.answer,
                "source": source,
                "remote_prompt_tokens": pt,
                "remote_completion_tokens": ct,
                "decision_path": path,
            }
            if report:
                entry["confidence"] = report.to_dict()
            self.ledger.record(entry)
        return result


def pick_model(allowed: list[str], hints: list[str], fallback: str) -> str:
    """Resolve the escalation model against the runtime ALLOWED_MODELS list.

    Model IDs must never be hardcoded (guide rule), so hints are substrings
    matched against whatever the harness injects.
    """
    if not allowed:
        return fallback
    for hint in hints:
        for model in allowed:
            if hint.lower() in model.lower():
                return model
    return fallback if fallback in allowed else allowed[0]


def _vote_keys(gens, task_type):
    """Truncated samples must not win the vote; their key is unusable."""
    return [
        None if g.finish_reason == "length" else vote_key(g.text, task_type) for g in gens
    ]


def _rank_by_confidence(gen) -> float:
    quantile = logprob_quantile(gen.token_logprobs)
    return quantile if quantile is not None else float("-inf")


def _representative(gens, keys, candidate):
    """The candidate-matching sample with the most confident answer tokens."""
    if candidate is None:
        return gens[0] if gens else None
    matching = [g for g, k in zip(gens, keys) if k == candidate]
    if not matching:
        return gens[0] if gens else None
    return max(matching, key=_rank_by_confidence)


def _best_logprob(gens, keys, candidate) -> float | None:
    """Pessimistic quantile logprob of the most confident candidate-matching sample."""
    if candidate is None:
        return None
    values = [
        quantile
        for g, k in zip(gens, keys)
        if k == candidate
        for quantile in [logprob_quantile(g.token_logprobs)]
        if quantile is not None
    ]
    return max(values) if values else None
