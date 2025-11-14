"""Core news aggregation service."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .config import AggregatorConfig, FetcherConfig, TimeWindowConfig
from .fetching import NewsFetcher
from .model_worker import ModelWorker
from .models import Article, NewsReport

LOGGER = logging.getLogger(__name__)


class NewsAggregator:
    """Fetch, summarize, store, and serve news reports."""

    def __init__(
        self,
        config: AggregatorConfig,
        model_worker: ModelWorker,
        sources_path: Path,
    ) -> None:
        self.config = config
        self.model_worker = model_worker
        if not sources_path.exists():
            raise FileNotFoundError(f"sources.json not found at {sources_path}")
        self.sources_path = sources_path
        self._reports: Dict[str, NewsReport] = {}
        self._cache_path = self.config.cache_path
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_refresh: Optional[datetime] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._http_client = httpx.Client(timeout=15)

        fetcher_cfg = FetcherConfig(sources_path=sources_path)
        self._fetcher = NewsFetcher(fetcher_cfg, TimeWindowConfig(since=config.retention_window.since))
        self._load_cached_reports()

    def start(self) -> None:
        LOGGER.info("Starting news aggregator service")
        self._stop_event.clear()
        self._initial_refresh()
        self._fetch_thread.start()
        self._cleanup_thread.start()

    def stop(self) -> None:
        LOGGER.info("Stopping news aggregator service")
        self._stop_event.set()
        self._fetch_thread.join(timeout=5)
        self._cleanup_thread.join(timeout=5)
        self._http_client.close()
        self._save_cached_reports()

    def _initial_refresh(self) -> None:
        try:
            self.refresh_news()
        except Exception as exc:  # pragma: no cover - startup resilience
            LOGGER.exception("Initial news refresh failed: %s", exc)

    def _fetch_loop(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            try:
                self.refresh_news()
            except Exception as exc:  # pragma: no cover - resilience
                LOGGER.exception("Periodic news refresh failed: %s", exc)
            elapsed = time.time() - start
            wait_time = max(0.0, self.config.fetch_interval_seconds - elapsed)
            self._stop_event.wait(wait_time)

    def _cleanup_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.config.cleanup_interval_seconds)
            try:
                self.remove_stale_reports()
            except Exception as exc:  # pragma: no cover - resilience
                LOGGER.exception("Cleanup failed: %s", exc)

    def refresh_news(self) -> None:
        LOGGER.info("Refreshing news feeds")
        cutoff = self._last_refresh
        articles = self._fetcher.fetch(cutoff=cutoff)
        LOGGER.info("Fetched %d articles", len(articles))
        for article in articles:
            self._process_article(article)
        self._last_refresh = datetime.now(timezone.utc)
        self.remove_stale_reports()
        self._save_cached_reports()

    def remove_stale_reports(self) -> None:
        cutoff = datetime.now(timezone.utc) - self.config.retention_window.as_timedelta()
        with self._lock:
            before = len(self._reports)
            self._reports = {
                url: report
                for url, report in self._reports.items()
                if report.published_at >= cutoff
            }
            after = len(self._reports)
        if after != before:
            LOGGER.info("Removed %d stale reports", before - after)
            self._save_cached_reports()

    def rebuild_reports(self) -> int:
        """Clear all cached reports and rebuild them from scratch."""

        LOGGER.info("Manual rebuild of all reports requested")
        with self._lock:
            self._reports.clear()
        self.refresh_news()
        with self._lock:
            rebuilt = len(self._reports)
        LOGGER.info("Rebuild complete with %d reports", rebuilt)
        return rebuilt

    def _process_article(self, article: Article) -> None:
        if article.content is None and not article.extras.get("newsplease", {}).get("description"):
            return
        published = article.best_timestamp()
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        with self._lock:
            if article.url in self._reports:
                return
        try:
            summary = self._summarize_article(article)
        except Exception as exc:  # pragma: no cover - LLM failure resilience
            LOGGER.warning("Summary generation failed for %s: %s", article.url, exc)
            summary = "Summary unavailable."
        image_b64 = self._download_image(article)
        report = NewsReport(
            agency=article.source_id,
            title=article.title,
            summary=summary,
            image_base64=image_b64,
            url=article.url,
            published_at=published,
            crawled_at=article.crawled_at,
        )
        with self._lock:
            self._reports[article.url] = report
        LOGGER.debug("Stored report for %s", article.url)

    def _summarize_article(self, article: Article) -> str:
        content = article.content or article.extras.get("newsplease", {}).get("description") or ""
        text = content.strip()
        if not text:
            return "Summary unavailable."
        text = text[: self.config.summary_max_chars]
        return self.model_worker.summarize(article.title, text, self.config.summary_max_tokens)

    def _download_image(self, article: Article) -> Optional[str]:
        image_url = (
            article.extras.get("newsplease", {}).get("image_url") if article.extras else None
        )
        if not image_url:
            return None
        try:
            response = self._http_client.get(image_url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "application/octet-stream")
            encoded = base64.b64encode(response.content).decode("ascii")
            return f"data:{content_type};base64,{encoded}"
        except Exception:  # pragma: no cover - network errors
            LOGGER.debug("Failed to download image for %s", article.url, exc_info=True)
            return None

    def list_reports(self) -> List[NewsReport]:
        with self._lock:
            return list(self._reports.values())

    def top_reports(self, topics: List[str], top_n: int) -> List[NewsReport]:
        reports = self.list_reports()
        if not reports:
            return []
        try:
            scores = self.model_worker.rerank(topics, reports, self.config.rerank_max_tokens)
        except Exception as exc:
            LOGGER.warning("Falling back to recency ordering due to rerank failure: %s", exc)
            reports.sort(key=lambda report: report.published_at, reverse=True)
            return reports[:top_n]
        scored: List[tuple[float, NewsReport]] = []
        for idx, report in enumerate(reports, start=1):
            score = scores.get(str(idx))
            if score is None:
                continue
            scored.append((score, report))
        if not scored:
            reports.sort(key=lambda report: report.published_at, reverse=True)
            return reports[:top_n]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [report for _, report in scored[:top_n]]

    def generate_financial_advice(self, payload_json: str) -> str:
        """Use the model worker to produce structured financial advice."""

        max_tokens = getattr(self.config, "advice_max_tokens", 512)
        return self.model_worker.financial_advice(payload_json, max_tokens)

    def _load_cached_reports(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupted cache
            LOGGER.warning("Failed to load cached reports: %s", exc)
            return

        reports: Dict[str, NewsReport] = {}
        for entry in raw.get("reports", []):
            try:
                published = datetime.fromisoformat(entry["published_at"])
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                crawled = datetime.fromisoformat(entry["crawled_at"])
                if crawled.tzinfo is None:
                    crawled = crawled.replace(tzinfo=timezone.utc)
                report = NewsReport(
                    agency=entry["agency"],
                    title=entry.get("title"),
                    summary=entry["summary"],
                    image_base64=entry.get("image_base64"),
                    url=entry["url"],
                    published_at=published,
                    crawled_at=crawled,
                )
            except Exception as exc:  # pragma: no cover - invalid entry
                LOGGER.debug("Skipping invalid cached report: %s", exc)
                continue
            reports[report.url] = report

        last_refresh_raw = raw.get("last_refresh")
        if last_refresh_raw:
            try:
                last_refresh = datetime.fromisoformat(last_refresh_raw)
                if last_refresh.tzinfo is None:
                    last_refresh = last_refresh.replace(tzinfo=timezone.utc)
                self._last_refresh = last_refresh
            except Exception:  # pragma: no cover - invalid timestamp
                LOGGER.debug("Invalid last_refresh timestamp in cache", exc_info=True)

        with self._lock:
            self._reports = reports

        self.remove_stale_reports()

    def _save_cached_reports(self) -> None:
        payload = {
            "reports": [],
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
        }
        with self._lock:
            for report in self._reports.values():
                payload["reports"].append(report.to_dict())
        try:
            self._cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - disk errors
            LOGGER.warning("Failed to persist cached reports", exc_info=True)


__all__ = ["NewsAggregator"]
