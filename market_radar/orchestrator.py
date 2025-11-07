"""Pipeline orchestrator tying together fetching, density, summarization and hotness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from .config import PipelineConfig
from .deduplication import Deduplicator, DeduplicationSettings
from .density_estimator import DensityEstimator
from .fetching import NewsFetcher
from .hotness import HotnessCalculator
from .models import Article
from .progress import PipelineProgress
from .summarizer import Summarizer

if TYPE_CHECKING:
    from .progress import StageHandle


class NewsPipelineOrchestrator:
    """High-level orchestrator that runs the Market Radar pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.start_time = datetime.now(timezone.utc)
        self.time_window_delta = NewsFetcher.parse_since(config.time_window.since)

        self.fetcher = NewsFetcher(config.fetcher, config.time_window)
        self.density_estimator = DensityEstimator(config.density)
        dedup_settings = DeduplicationSettings(
            enabled=config.density.deduplicate,
            threshold=config.density.deduplication_threshold,
        )
        self.deduplicator = Deduplicator(dedup_settings)
        self.summarizer = Summarizer(config.summarizer)
        self.hotness = HotnessCalculator(config.hotness, self.start_time, self.time_window_delta)

    def run(self) -> List[Dict[str, object]]:
        with PipelineProgress() as progress:
            with progress.stage("Fetch") as stage:
                articles = self.fetcher.fetch(self.start_time, stage=stage)

            if not articles:
                with progress.stage("Output") as stage:
                    stage.set_total(1)
                    self._write_output([])
                    stage.advance(1)
                return []

            with progress.stage("Density") as stage:
                density_scores = self.density_estimator.estimate(articles, stage=stage)
            title_embeddings = self.density_estimator.get_title_embeddings()

            with progress.stage("Deduplicate") as stage:
                articles, density_values = self.deduplicator.apply(
                    articles, density_scores, title_embeddings, stage=stage
                )

            if not articles:
                with progress.stage("Output") as stage:
                    stage.set_total(1)
                    self._write_output([])
                    stage.advance(1)
                return []

            for art, coef in zip(articles, density_values):
                art.density_coef = coef

            with progress.stage("Summaries") as stage:
                self.summarizer.summarize(articles, stage=stage)

            with progress.stage("Hotness") as stage:
                self.hotness.apply(articles, stage=stage)

            with progress.stage("Output") as stage:
                stage.set_total(len(articles) + 1)
                output = self._build_output(articles, stage=stage)
                self._write_output(output)
                stage.advance(1)

        return output

    def _build_output(
        self,
        articles: Sequence[Article],
        stage: Optional["StageHandle"] = None,
    ) -> List[Dict[str, object]]:
        def _to_iso(dt: datetime) -> str:
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        payload: List[Dict[str, object]] = []
        for art in articles:
            published = art.published_at or art.crawled_at
            payload.append(
                {
                    "source": art.source_id,
                    "source_domain": art.source_domain,
                    "published_at": _to_iso(published),
                    "url": art.url,
                    "title": art.title,
                    "summary": art.summary,
                    "time_coef": round(float(art.time_coef or 0.0), 6),
                    "density_coef": round(float(art.density_coef or 0.0), 6),
                    "domain_coef": round(float(art.domain_coef or 0.0), 6),
                    "hotness": round(float(art.hotness or 0.0), 6),
                }
            )
            if stage is not None:
                stage.advance(1)

        payload.sort(key=lambda item: item["hotness"], reverse=True)
        return payload

    def _write_output(self, data: Sequence[Dict[str, object]]) -> None:
        path = self.config.output.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_from_config(path: Path) -> List[Dict[str, object]]:
    config = PipelineConfig.from_yaml(path)
    orchestrator = NewsPipelineOrchestrator(config)
    return orchestrator.run()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Market Radar pipeline orchestrator")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_from_config(config_path)
