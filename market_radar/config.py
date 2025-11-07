"""Configuration helpers for the news aggregator service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
import os


@dataclass
class TimeWindowConfig:
    """Time window configuration for RSS fetching."""

    since: str = "7d"

    def as_timedelta(self) -> timedelta:
        value = self.since.strip().lower()
        if not value:
            raise ValueError("time window 'since' must not be empty")
        number = ""
        unit = ""
        for ch in value:
            if ch.isdigit():
                number += ch
            else:
                unit += ch
        if not number or not unit:
            raise ValueError(f"invalid time window value: {self.since!r}")
        amount = int(number)
        unit = unit.strip()
        mapping = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
        }
        if unit not in mapping:
            raise ValueError(f"unsupported time unit: {unit}")
        return mapping[unit]


@dataclass
class FetcherConfig:
    """Configuration for the RSS fetcher."""

    sources_path: Path
    min_chars: int = 400
    max_per_source: int = 200
    concurrency: int = 8
    timeout: int = 30
    user_agent: str = "news-aggregator/1.0"
    feed_retries: int = 2


@dataclass
class AggregatorConfig:
    """Top-level configuration for the service."""

    fetch_interval_seconds: int = 1800
    cleanup_interval_seconds: int = 3600
    retention_window: TimeWindowConfig = field(default_factory=TimeWindowConfig)
    model_name: str = "Qwen/Qwen3-0.6B"
    model_port: int = 8001
    model_host: str = "127.0.0.1"
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", "8080"))
    summary_max_tokens: int = 256
    rerank_max_tokens: int = 512
    summary_max_chars: int = 4000

    @property
    def base_url(self) -> str:
        return f"http://{self.model_host}:{self.model_port}"


def load_config() -> AggregatorConfig:
    """Load configuration from environment variables with sensible defaults."""

    fetch_interval = int(os.getenv("FETCH_INTERVAL_SECONDS", "1800"))
    cleanup_interval = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))
    retention = os.getenv("RETENTION_WINDOW", "7d")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
    model_port = int(os.getenv("MODEL_PORT", "8001"))
    model_host = os.getenv("MODEL_HOST", "127.0.0.1")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("PORT", "8080"))
    summary_max_tokens = int(os.getenv("SUMMARY_MAX_TOKENS", "256"))
    rerank_max_tokens = int(os.getenv("RERANK_MAX_TOKENS", "512"))
    summary_max_chars = int(os.getenv("SUMMARY_MAX_CHARS", "4000"))

    retention_window = TimeWindowConfig(since=retention)

    return AggregatorConfig(
        fetch_interval_seconds=fetch_interval,
        cleanup_interval_seconds=cleanup_interval,
        retention_window=retention_window,
        model_name=model_name,
        model_port=model_port,
        model_host=model_host,
        api_host=api_host,
        api_port=api_port,
        summary_max_tokens=summary_max_tokens,
        rerank_max_tokens=rerank_max_tokens,
        summary_max_chars=summary_max_chars,
    )


__all__ = [
    "AggregatorConfig",
    "FetcherConfig",
    "TimeWindowConfig",
    "load_config",
]
