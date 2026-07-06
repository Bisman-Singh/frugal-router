"""YAML-backed settings and agent assembly."""
from __future__ import annotations

import os
from dataclasses import dataclass
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
    timeout_s: float = 60.0
    max_retries: int = 2


@dataclass
class Settings:
    local: LocalConfig
    remote: RemoteConfig
    policies: PolicyBook
    weights: dict | None = None
    cache_path: str = "artifacts/remote_cache.sqlite"
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
        weights=router.get("weights"),
        cache_path=router.get("cache_path", "artifacts/remote_cache.sqlite"),
        predictor_path=router.get("predictor_path", "artifacts/predictor.joblib"),
    )


def build_agent(settings: Settings, *, ledger=None):
    """Assemble the agent from settings. Missing pieces degrade gracefully:
    no GGUF file means remote-only, no API key means local-only."""
    from .agent import RoutingAgent
    from .cache import ResponseCache
    from .predictor import FailurePredictor

    local = None
    if settings.local.model_path and Path(settings.local.model_path).exists():
        from .backends.llama_local import LlamaLocalBackend

        local = LlamaLocalBackend(
            model_path=settings.local.model_path,
            n_ctx=settings.local.n_ctx,
            n_threads=settings.local.n_threads or None,
            n_gpu_layers=settings.local.n_gpu_layers,
            chat_format=settings.local.chat_format,
        )

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
        predictor=FailurePredictor.load(settings.predictor_path),
        cache=ResponseCache(settings.cache_path),
        ledger=ledger,
        weights=settings.weights,
    )
