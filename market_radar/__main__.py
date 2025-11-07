"""Entrypoint for running the Market Radar API with ``python -m market_radar``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import uvicorn

from .api import configure_default_config_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for launching the API service."""

    parser = argparse.ArgumentParser(description="Run the Market Radar API service")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file used by the pipeline",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Hostname or IP address to bind the API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the API server (defaults to $PORT or 8000)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the FastAPI application with Uvicorn."""

    args = parse_args(argv)
    configure_default_config_path(Path(args.config))

    port = args.port if args.port is not None else int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "market_radar.api:create_app",
        host=args.host,
        port=port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
