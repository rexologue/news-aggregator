"""Configuration models and loaders for the Market Radar pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TimeWindowConfig:
    """Configuration for the time window used to collect news."""

    since: str
    timezone: Optional[str] = None


@dataclass
class FetcherConfig:
    """Configuration for RSS/news fetching."""

    sources_path: Path
    min_chars: int = 400
    max_per_source: int = 200
    concurrency: int = 8
    timeout: int = 30
    user_agent: str = "market-radar/1.0"
    feed_retries: int = 2


@dataclass
class DensityConfig:
    """Configuration for the density estimator."""

    model_id: str
    model_cache_dir: Optional[Path] = None
    title_score: float = 0.7
    content_score: float = 0.3
    content_chars: int = 300
    batch_size: int = 64
    window_hours: int = 24
    deduplicate: bool = True
    deduplication_threshold: float = 0.92


@dataclass
class SummarizerConfig:
    """Configuration for the LLM summarizer."""

    model: str
    temperature: float = 0.2
    timeout: int = 60
    api_key: Optional[str] = None
    fallback_summary: bool = True


@dataclass
class HotnessWeights:
    """Weights applied when calculating the final hotness score."""

    time: float
    density: float
    domain: float


@dataclass
class HotnessConfig:
    """Configuration of hotness calculation."""

    weights: HotnessWeights
    time_decay: float = 4.0


@dataclass
class OutputConfig:
    """Configuration for pipeline output."""

    path: Path


@dataclass
class PipelineConfig:
    """Root configuration for the orchestrator."""

    time_window: TimeWindowConfig
    fetcher: FetcherConfig
    density: DensityConfig
    summarizer: SummarizerConfig
    hotness: HotnessConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Instantiate :class:`PipelineConfig` from a raw dictionary."""

        time_window = TimeWindowConfig(**data["time_window"])

        fetcher_cfg_raw = {**data.get("fetcher", {})}
        if "sources_path" in fetcher_cfg_raw:
            fetcher_cfg_raw["sources_path"] = Path(fetcher_cfg_raw["sources_path"])
        fetcher = FetcherConfig(**fetcher_cfg_raw)

        density_cfg_raw = {**data.get("density", {})}
        if "model_cache_dir" in density_cfg_raw and density_cfg_raw["model_cache_dir"]:
            density_cfg_raw["model_cache_dir"] = Path(density_cfg_raw["model_cache_dir"])
        density = DensityConfig(**density_cfg_raw)

        summarizer_cfg_raw = {**data.get("summarizer", {})}
        summarizer = SummarizerConfig(**summarizer_cfg_raw)

        hotness_raw = {**data.get("hotness", {})}
        weights_raw = hotness_raw.get("weights", {})
        weights = HotnessWeights(**weights_raw)
        hotness_raw["weights"] = weights
        hotness = HotnessConfig(**hotness_raw)

        output_cfg_raw = {**data.get("output", {})}
        output_cfg_raw["path"] = Path(output_cfg_raw["path"])
        output = OutputConfig(**output_cfg_raw)

        return cls(
            time_window=time_window,
            fetcher=fetcher,
            density=density,
            summarizer=summarizer,
            hotness=hotness,
            output=output,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("YAML config must contain a mapping at the root")
        return cls.from_dict(data)

    @classmethod
    def from_yaml_string(cls, content: str) -> "PipelineConfig":
        """Load configuration from a raw YAML string."""

        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("YAML config must contain a mapping at the root")
        return cls.from_dict(data)


__all__ = [
    "PipelineConfig",
    "TimeWindowConfig",
    "FetcherConfig",
    "DensityConfig",
    "SummarizerConfig",
    "HotnessConfig",
    "HotnessWeights",
    "OutputConfig",
]
