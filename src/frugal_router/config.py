"""YAML-backed settings and agent assembly."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .policy import PolicyBook


@dataclass
class LocalConfig:
    model_path: str = ""
    n_ctx: int = 8192
    n_threads: int = 0
    n_gpu_layers: int = 0
    chat_format: str | None = None


@dataclass
class RemoteConfig:
    base_url: str = "https://api.fireworks.ai/inference/v1"
    default_model: str = ""
    timeout_s: float = 25.0
    max_retries: int = 1


@dataclass
class SchedulerConfig:
    time_budget_s: float = 570.0  # the harness allows 600s for the whole batch
    est_full_s: float = 40.0      # voting attempt on CPU
    est_greedy_s: float = 12.0    # single local sample
    est_remote_s: float = 5.0     # one proxied remote call
    max_workers: int = 4          # parallel remote solving when no local model runs


@dataclass
class Settings:
    local: LocalConfig
    remote: RemoteConfig
    policies: PolicyBook
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    weights: dict | None = None
    answer_source: str = "fireworks"  # event rule: scored answers come from Fireworks
    solver_mode: str = "confirm"  # deterministic tier: confirm | direct | off
    predictor_path: str = "artifacts/predictor.joblib"


def load_settings(path: str | Path) -> Settings:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    policies_raw = raw.get("policies") or {}
    router = raw.get("router") or {}
    return Settings(
        local=LocalConfig(**(raw.get("local") or {})),
        remote=RemoteConfig(**(raw.get("remote") or {})),
        policies=PolicyBook(policies_raw.get("defaults"), policies_raw.get("per_type")),
        scheduler=SchedulerConfig(**(raw.get("scheduler") or {})),
        weights=router.get("weights"),
        answer_source=router.get("answer_source", "fireworks"),
        solver_mode=router.get("solver_mode", "confirm"),
        predictor_path=router.get("predictor_path", "artifacts/predictor.joblib"),
    )


def allowed_models_from_env() -> list[str]:
    """The judging harness injects ALLOWED_MODELS as a comma-separated list.
    Model IDs must come from here, never from code (guide rule)."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def build_agent(settings: Settings, *, ledger=None):
    """Assemble the agent from settings. Missing pieces degrade gracefully:
    no GGUF file means remote-only, no API key means local-only."""
    from .agent import RoutingAgent
    from .predictor import FailurePredictor

    local = None
    if settings.local.model_path and Path(settings.local.model_path).exists():
        # A broken local backend must never take the remote path down with it.
        try:
            from .backends.llama_local import LlamaLocalBackend

            local = LlamaLocalBackend(
                model_path=settings.local.model_path,
                n_ctx=settings.local.n_ctx,
                n_threads=settings.local.n_threads or None,
                n_gpu_layers=settings.local.n_gpu_layers,
                chat_format=settings.local.chat_format,
            )
        except Exception as exc:
            import sys

            print(f"local backend unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)

    remote = None
    if os.environ.get("FIREWORKS_API_KEY"):
        from .backends.fireworks import FireworksBackend

        remote = FireworksBackend(
            base_url=settings.remote.base_url,
            timeout=settings.remote.timeout_s,
            max_retries=settings.remote.max_retries,
        )

    return RoutingAgent(
        local,
        remote,
        settings.policies,
        default_remote_model=settings.remote.default_model,
        allowed_models=allowed_models_from_env(),
        answer_source=settings.answer_source,
        solver_mode=settings.solver_mode,
        predictor=FailurePredictor.load(settings.predictor_path),
        ledger=ledger,
        weights=settings.weights,
    )
