"""The cascade agent: answer locally for free, escalate only on low confidence.

Fail safe, never silent: a local crash escalates, a remote crash falls back to
the best local candidate. The agent always returns an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .cache import ResponseCache
from .classify import classify
from .compress import maybe_compress
from .confidence import ConfidenceReport, combine, vote
from .extract import extract_answer, is_valid, normalize
from .ledger import Ledger
from .policy import Policy, PolicyBook
from .tasks import Task


@dataclass
class SolveResult:
    task_id: str
    answer: str
    source: str  # local | remote | cache | fallback
    task_type: str
    local_candidate: str | None = None
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
        predictor=None,
        cache: ResponseCache | None = None,
        ledger: Ledger | None = None,
        weights: dict | None = None,
    ):
        self.local = local
        self.remote = remote
        self.policies = policies
        self.default_remote_model = default_remote_model
        self.predictor = predictor
        self.cache = cache
        self.ledger = ledger
        self.weights = weights

    def solve(self, task: Task) -> SolveResult:
        path: list[str] = []
        task_type = classify(task)
        path.append(f"type:{task_type}")
        policy = self.policies.for_type(task_type)

        report = None
        if self.local is None:
            path.append("no_local_backend")
        elif policy.always_remote:
            path.append("policy:always_remote")
        else:
            try:
                report = self._local_attempt(task, task_type, policy, path)
            except Exception as exc:
                path.append(f"local_error:{type(exc).__name__}")

        if report is not None and not self._should_escalate(report, policy, path):
            return self._finish(
                task, task_type, report.candidate or "", "local", report, 0, 0, path
            )

        try:
            answer, pt, ct, cached = self._remote_answer(task, task_type, policy, path)
            source = "cache" if cached else "remote"
            if cached:
                pt = ct = 0  # cached responses bill nothing
            return self._finish(task, task_type, answer, source, report, pt, ct, path)
        except Exception as exc:
            path.append(f"remote_error:{type(exc).__name__}")
            fallback = report.candidate if report and report.candidate else ""
            path.append("fallback")
            return self._finish(task, task_type, fallback, "fallback", report, 0, 0, path)

    # -- local -----------------------------------------------------------

    def _local_attempt(
        self, task: Task, task_type: str, policy: Policy, path: list[str]
    ) -> ConfidenceReport:
        question = prompts.question_text(task.rendered_input(), task.context)
        spec = prompts.format_spec_for(task_type, policy.format_spec or None)
        system, user = prompts.local_solve(question, task_type, spec, policy.few_shot)
        gens = self.local.generate(
            system,
            user,
            n=policy.n_samples,
            temperature=policy.sample_temperature if policy.n_samples > 1 else 0.0,
            max_tokens=policy.local_max_tokens,
        )
        normalized = [normalize(extract_answer(g.text), task_type) for g in gens]
        candidate, agreement = vote(normalized)
        verify = None
        if policy.use_verification and candidate:
            verify = self.local.yes_probability(prompts.verification(question, candidate))
        p_fail = self.predictor.p_fail(task.input) if self.predictor else None
        report = ConfidenceReport(
            candidate=candidate,
            agreement=agreement,
            n_samples=policy.n_samples,
            mean_logprob=_best_logprob(gens, normalized, candidate),
            verify_yes_prob=verify,
            p_fail=p_fail,
            format_valid=is_valid(candidate, task_type),
        )
        report.score = combine(report, self.weights)
        path.append(f"local:agreement={agreement:.2f},score={report.score:.2f}")
        return report

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

        Returns (answer, prompt_tokens, completion_tokens, from_cache); on a
        cache hit the tokens are the original call's would-be cost.
        """
        task_type = task_type or classify(task)
        policy = self.policies.for_type(task_type)
        return self._remote_answer(task, task_type, policy, [])

    def _remote_answer(
        self, task: Task, task_type: str, policy: Policy, path: list[str]
    ) -> tuple[str, int, int, bool]:
        if self.remote is None:
            raise RuntimeError("no remote backend configured")
        spec = prompts.format_spec_for(task_type, policy.format_spec or None)
        context, compressed = maybe_compress(
            self.local, task.rendered_input(), task.context, policy.compress_over_chars
        )
        if compressed:
            path.append("compressed_context")
        question = prompts.question_text(task.rendered_input(), context)
        build = prompts.remote_cot if policy.remote_cot else prompts.remote_minimal
        prompt = build(question, spec)
        model = policy.remote_model or self.default_remote_model
        if not model:
            raise RuntimeError("no remote model configured")

        key = None
        if self.cache is not None:
            key = ResponseCache.key(model=model, prompt=prompt, max_tokens=policy.remote_max_tokens)
            hit = self.cache.get(key)
            if hit is not None:
                path.append("cache_hit")
                answer = self._parse_remote(hit["text"], task_type, spec, path)
                return answer, hit.get("prompt_tokens", 0), hit.get("completion_tokens", 0), True

        gen = self.remote.generate(
            None, prompt, model=model, temperature=0.0, max_tokens=policy.remote_max_tokens
        )
        if self.cache is not None and key is not None:
            self.cache.put(
                key,
                {
                    "text": gen.text,
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                },
            )
        path.append(f"remote:{model}")
        answer = self._parse_remote(gen.text, task_type, spec, path)
        return answer, gen.prompt_tokens, gen.completion_tokens, False

    def _parse_remote(self, text: str, task_type: str, spec: str, path: list[str]) -> str:
        raw = (extract_answer(text) or text or "").strip()
        answer = normalize(raw, task_type)
        if not is_valid(answer, task_type) and self.local is not None and raw:
            # Free local reformat instead of a second billed call.
            try:
                system, user = prompts.reformat(raw, spec)
                gens = self.local.generate(system, user, n=1, temperature=0.0, max_tokens=64)
                answer = normalize(extract_answer(gens[0].text), task_type)
                path.append("local_reformat")
            except Exception:
                pass
        if is_valid(answer, task_type):
            return answer
        return answer or raw or ""

    def _finish(self, task, task_type, answer, source, report, pt, ct, path) -> SolveResult:
        result = SolveResult(
            task_id=task.id,
            answer=answer or "",
            source=source,
            task_type=task_type,
            local_candidate=report.candidate if report else None,
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


def _best_logprob(gens, normalized, candidate) -> float | None:
    if candidate is None:
        return None
    values = [
        g.mean_logprob
        for g, n in zip(gens, normalized)
        if n == candidate and g.mean_logprob is not None
    ]
    return max(values) if values else None
