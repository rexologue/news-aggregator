"""FastAPI application exposing the Market Radar pipeline over HTTP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .config import PipelineConfig
from .orchestrator import NewsPipelineOrchestrator

DEFAULT_CONFIG_ENV = "MARKET_RADAR_CONFIG"
DEFAULT_CONFIG_FALLBACK = "config.example.yaml"
MODEL_CACHE_ENV = "MARKET_RADAR_MODEL_CACHE"

_DEFAULT_CONFIG_PATH: Path | None = None


def configure_default_config_path(path: Path | str) -> None:
    """Set the default configuration file path used by the API factory."""

    global _DEFAULT_CONFIG_PATH
    _DEFAULT_CONFIG_PATH = Path(path).expanduser()


def create_app(config_path: Path | str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    default_config_path = _determine_config_path(config_path)

    app = FastAPI(
        title="Market Radar API",
        version="1.0.0",
        description=(
            "HTTP wrapper for the Market Radar pipeline. Runtime overrides are limited "
            "to the 'since' window while responses return the generated JSON file."
        ),
    )

    @app.get("/healthz", summary="Health check")
    async def health_check() -> dict[str, str]:
        """Return a simple health indicator."""

        return {"status": "ok"}

    @app.post(
        "/pipeline",
        summary="Run the Market Radar pipeline",
        response_class=FileResponse,
    )
    async def run_pipeline(
        since: Optional[str] = Query(
            None,
            description="Override the time window 'since' value, e.g. '6h'",
        ),
    ) -> FileResponse:
        """Execute the pipeline with optional overrides and return the output file."""

        try:
            response = await run_in_threadpool(
                _execute_pipeline,
                default_config_path,
                since,
            )
        except FileNotFoundError as exc:  # pragma: no cover - simple mapping
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:  # pragma: no cover - validation mapping
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return FileResponse(
            response,
            media_type="application/json",
            filename=response.name,
        )

    return app


def _execute_pipeline(
    default_config_path: Path,
    since: Optional[str],
) -> Path:
    """Load configuration, apply overrides, and run the pipeline synchronously."""

    config = _load_config(default_config_path)

    if since:
        config.time_window.since = since

    config.output.path = _ensure_parent(_resolve_output_path(config, default_config_path))

    model_cache = os.getenv(MODEL_CACHE_ENV)
    if model_cache and config.density.model_cache_dir is None:
        cache_dir = Path(model_cache).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        config.density.model_cache_dir = cache_dir

    orchestrator = NewsPipelineOrchestrator(config)
    orchestrator.run()

    return config.output.path


def _determine_config_path(config_path: Path | str | None) -> Path:
    """Resolve the configuration path from CLI, module, or environment."""

    candidates: List[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path).expanduser())
    if _DEFAULT_CONFIG_PATH is not None:
        candidates.append(_DEFAULT_CONFIG_PATH)

    env_path = os.getenv(DEFAULT_CONFIG_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    if not candidates:
        candidates.append(Path(DEFAULT_CONFIG_FALLBACK).expanduser())

    return candidates[0]


def _load_config(default_config_path: Path) -> PipelineConfig:
    """Load configuration from the default path defined at application startup."""

    config_path = _validate_path(default_config_path, "configuration file")
    return PipelineConfig.from_yaml(config_path)


def _resolve_output_path(config: PipelineConfig, default_config_path: Path) -> Path:
    """Resolve the output path defined in the configuration file."""

    output_path = config.output.path
    if not output_path.is_absolute():
        output_path = (default_config_path.parent / output_path).resolve()
    return output_path


def _validate_path(path_input: Path | str, label: str, *, must_exist: bool = True) -> Path:
    """Validate and normalise file paths coming from the request."""

    path = Path(path_input).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label.capitalize()} not found: {path}")
    return path


def _ensure_parent(path: Path) -> Path:
    """Ensure the parent directory exists before writing output files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "configure_default_config_path",
    "create_app",
]
