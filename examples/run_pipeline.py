"""Command-line helper that runs the Market Radar pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from market_radar.config import PipelineConfig
from market_radar.orchestrator import NewsPipelineOrchestrator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Market Radar pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.example.yaml"),
        help=(
            "Path to the YAML configuration file. Defaults to ./config.example.yaml."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PipelineConfig.from_yaml(args.config)
    orchestrator = NewsPipelineOrchestrator(config)
    results = orchestrator.run()

    output_path = config.output.path
    if results:
        print(f"Wrote {len(results)} articles ordered by hotness.")
        print(f"Output file: {output_path}")
    else:
        print("No articles found in the configured window. Output file still created.")
        print(f"Output file: {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
