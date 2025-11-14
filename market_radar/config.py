"""Configuration helpers for the news aggregator service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional
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


def _looks_like_model_dir(path: Path) -> bool:
    """Return True if the given directory resembles a Hugging Face model export."""

    if not path.is_dir():
        return False

    config_file = path / "config.json"
    has_safetensors = any(path.glob("*.safetensors"))
    has_bin = any(path.glob("*.bin"))

    return config_file.exists() and (has_safetensors or has_bin)


@dataclass
class AggregatorConfig:
    """Top-level configuration for the service."""

    fetch_interval_seconds: int = 1800
    cleanup_interval_seconds: int = 3600
    retention_window: TimeWindowConfig = field(default_factory=TimeWindowConfig)
    cache_path: Path = Path("data/news_cache.json")
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    model_quantization: Optional[str] = "fp8"
    model_variant: str = "fp8"
    model_port: int = 8001
    model_host: str = "127.0.0.1"
    model_local_path: Optional[Path] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", "8080"))
    summary_max_tokens: int = 768
    rerank_max_tokens: int = 512
    summary_max_chars: int = 10000
    advice_max_tokens: int = 512

    @property
    def base_url(self) -> str:
        return f"http://{self.model_host}:{self.model_port}"

    @property
    def local_model_path(self) -> Optional[Path]:
        """Return the local model path if it looks like a model directory."""

        if not self.model_local_path:
            return None

        candidate = self.model_local_path.expanduser()
        if _looks_like_model_dir(candidate):
            return candidate

        # Allow pointing to a parent directory that contains named subfolders.
        model_dir_name = self.model_name.split("/")[-1]
        nested_candidate = candidate / model_dir_name
        if _looks_like_model_dir(nested_candidate):
            return nested_candidate

        # Some local exports may omit the quantisation suffix.
        if model_dir_name.endswith("-FP8"):
            alt_name = model_dir_name[:-4]
            alt_candidate = candidate / alt_name
            if _looks_like_model_dir(alt_candidate):
                return alt_candidate

        return None

    @property
    def model_identifier(self) -> str:
        """Return the identifier that should be passed to vLLM."""

        local_path = self.local_model_path
        if local_path is not None:
            return str(local_path)
        return self.model_name

    @property
    def using_local_model(self) -> bool:
        return self.local_model_path is not None


def _resolve_model_settings() -> tuple[str, Optional[str], str]:
    """Resolve the desired model name/quantisation from environment variables."""

    variant_env = os.getenv("MODEL_VARIANT")
    presets = {
        "fp8": ("Qwen/Qwen3-4B-Instruct-2507-FP8", "fp8"),
        "instruct": ("Qwen/Qwen3-4B-Instruct-2507", None),
        "base": ("Qwen/Qwen3-4B-Instruct-2507", None),
    }

    if variant_env:
        key = variant_env.strip().lower()
        if key not in presets:
            raise ValueError(
                "Unsupported MODEL_VARIANT value. Available options: "
                + ", ".join(sorted(presets))
            )
        name, quant = presets[key]
        return name, quant, key

    model_name_env = os.getenv("MODEL_NAME")
    if model_name_env:
        model_name = model_name_env
        quant_env = os.getenv("MODEL_QUANTIZATION")
        model_quantization = quant_env or None
    else:
        model_name = "Qwen/Qwen3-4B-Instruct-2507-FP8"
        model_quantization = os.getenv("MODEL_QUANTIZATION", "fp8") or None

    variant = "fp8"
    if model_quantization and model_quantization.lower() != "fp8":
        variant = model_quantization.lower()
    elif not model_quantization:
        variant = "instruct"
    elif not model_name.lower().endswith("-fp8"):
        variant = "instruct"

    return model_name, model_quantization, variant


def load_config() -> AggregatorConfig:
    """Load configuration from environment variables with sensible defaults."""

    fetch_interval = int(os.getenv("FETCH_INTERVAL_SECONDS", "1800"))
    cleanup_interval = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))
    retention = os.getenv("RETENTION_WINDOW", "7d")
    cache_path = Path(os.getenv("CACHE_PATH", "data/news_cache.json"))
    model_name, model_quantization, model_variant = _resolve_model_settings()
    model_port = int(os.getenv("MODEL_PORT", "8001"))
    model_host = os.getenv("MODEL_HOST", "127.0.0.1")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("PORT", "8080"))
    summary_max_tokens = int(os.getenv("SUMMARY_MAX_TOKENS", "256"))
    rerank_max_tokens = int(os.getenv("RERANK_MAX_TOKENS", "512"))
    summary_max_chars = int(os.getenv("SUMMARY_MAX_CHARS", "4000"))
    advice_max_tokens = int(os.getenv("ADVICE_MAX_TOKENS", "512"))
    model_local_path_env = os.getenv("MODEL_LOCAL_PATH")
    gpu_memory_utilization_env = os.getenv("GPU_MEMORY_UTILIZATION")
    max_model_len_env = os.getenv("MAX_MODEL_LEN")
    gpu_memory_utilization: Optional[float]
    if gpu_memory_utilization_env is not None:
        gpu_memory_utilization = float(gpu_memory_utilization_env)
    else:
        gpu_memory_utilization = None
    max_model_len: Optional[int]
    if max_model_len_env is not None:
        max_model_len = int(max_model_len_env)
    else:
        max_model_len = None

    default_local_dir = Path("/models")
    model_local_path: Optional[Path]
    if model_local_path_env:
        model_local_path = Path(model_local_path_env).expanduser()
    elif default_local_dir.exists():
        model_local_path = default_local_dir
    else:
        model_local_path = None

    retention_window = TimeWindowConfig(since=retention)

    return AggregatorConfig(
        fetch_interval_seconds=fetch_interval,
        cleanup_interval_seconds=cleanup_interval,
        retention_window=retention_window,
        cache_path=cache_path,
        model_name=model_name,
        model_port=model_port,
        model_host=model_host,
        model_quantization=model_quantization,
        model_variant=model_variant,
        model_local_path=model_local_path,
        api_host=api_host,
        api_port=api_port,
        summary_max_tokens=summary_max_tokens,
        rerank_max_tokens=rerank_max_tokens,
        summary_max_chars=summary_max_chars,
        advice_max_tokens=advice_max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


__all__ = [
    "AggregatorConfig",
    "FetcherConfig",
    "TimeWindowConfig",
    "load_config",
]
