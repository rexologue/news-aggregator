"""News fetching utilities for the Market Radar pipeline."""

from __future__ import annotations

import concurrent.futures as futures
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import feedparser
import tldextract
from dateutil import parser as dtparse
from newsplease import NewsPlease

from .config import FetcherConfig, TimeWindowConfig
from .models import Article

if TYPE_CHECKING:
    from .progress import StageHandle

@dataclass
class Source:
    """Representation of a source entry from the sources JSON."""

    id: str
    type: str
    urls: Sequence[str]


class NewsFetcher:
    """Collect news articles within a configured time window."""

    def __init__(self, config: FetcherConfig, time_window: TimeWindowConfig) -> None:
        self.config = config
        self.time_window = time_window

    @staticmethod
    def parse_since(value: str) -> timedelta:
        """Parse a duration string like ``"2h"`` into :class:`timedelta`."""

        value = value.strip().lower()
        number = ""
        unit = ""
        for ch in value:
            if ch.isdigit():
                number += ch
            else:
                unit += ch
        if not number or not unit:
            raise ValueError(f"Invalid time window value: {value}")
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
            raise ValueError(f"Unsupported time unit: {unit}")
        return mapping[unit]

    @staticmethod
    def _best_entry_datetime(entry: Dict[str, object]) -> Optional[datetime]:
        for key in ("published", "pubDate", "updated", "dc_date"):
            value = entry.get(key)
            if not value:
                continue
            try:
                dt = dtparse.parse(str(value))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
        for key in ("published_parsed", "updated_parsed"):
            value = entry.get(key)
            if not value:
                continue
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except Exception:
                continue
        return None

    def _collect_feed_urls(
        self,
        source_id: str,
        urls: Sequence[str],
        since_utc: datetime,
        request_headers: Optional[Dict[str, str]] = None,
        retries: int = 2,
        log_error: Optional[Callable[[str, Optional[BaseException]], None]] = None,
    ) -> List[Tuple[str, Optional[datetime]]]:
        """Robustly collect links from RSS feeds with retries."""

        collected: List[Tuple[str, Optional[datetime]]] = []
        headers = request_headers or {}
        for feed_url in urls:
            parsed = None
            last_error: Optional[BaseException] = None
            for attempt in range(retries + 1):
                try:
                    parsed = feedparser.parse(feed_url, request_headers=headers)
                    break
                except Exception as exc:  # pragma: no cover - network failure branch
                    last_error = exc
                    if attempt < retries:
                        sleep_s = (1.5 ** attempt) + random.random() * 0.5
                        time.sleep(sleep_s)
                    else:
                        message = (
                            f"RSS fetch failed after retries | src={source_id} | url={feed_url} | err={exc!r}"
                        )
                        if log_error:
                            log_error(message, exc)
                        else:
                            print(f"[w] {message}")
            if not parsed:
                continue

            status = getattr(parsed, "status", None)
            if status and status != 200:
                message = (
                    f"non-200 RSS response | src={source_id} | url={feed_url} | status={status}"
                )
                if log_error:
                    log_error(message, last_error)
                else:
                    print(f"[w] {message}")

            try:
                entries = getattr(parsed, "entries", []) or []
            except Exception as exc:  # pragma: no cover - defensive
                if log_error:
                    log_error(f"bad RSS structure | src={source_id} | url={feed_url}", exc)
                else:
                    print(f"[w] bad RSS structure | src={source_id} | url={feed_url}")
                entries = []

            for entry in entries:
                dt = self._best_entry_datetime(entry)
                link = entry.get("link") if isinstance(entry, dict) else getattr(entry, "link", "")
                if not link:
                    continue
                if dt is None or dt >= since_utc:
                    collected.append((str(link), dt))

        unique: List[Tuple[str, Optional[datetime]]] = []
        seen = set()
        for url, dt in collected:
            if url not in seen:
                unique.append((url, dt))
                seen.add(url)
        return unique

    @staticmethod
    def _extract_article(url: str) -> Optional[NewsPlease]:
        try:
            return NewsPlease.from_url(url)
        except Exception:
            return None

    def _map_article(
        self,
        source_id: str,
        url: str,
        guess_dt: Optional[datetime],
    ) -> Optional[Article]:
        art = self._extract_article(url)
        if art is None:
            return None

        content = getattr(art, "maintext", None)
        if content and len(content) < self.config.min_chars:
            return None

        title = getattr(art, "title", None)
        ext = tldextract.extract(url)
        domain = ".".join(part for part in [ext.domain, ext.suffix] if part)

        published_dt = getattr(art, "date_publish", None)
        if isinstance(published_dt, datetime):
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
            published = published_dt.astimezone(timezone.utc)
        elif guess_dt is not None:
            published = guess_dt.astimezone(timezone.utc)
        else:
            published = None

        crawled_at = datetime.now(timezone.utc)
        language = getattr(art, "language", None)
        authors = getattr(art, "authors", None)
        if not isinstance(authors, list):
            authors = None

        extras = {
            "newsplease": {
                "date_download": str(getattr(art, "date_download", None)),
                "description": getattr(art, "description", None),
                "image_url": getattr(art, "image_url", None),
            }
        }

        return Article(
            source_id=source_id,
            source_domain=domain,
            url=url,
            title=title,
            content=content,
            published_at=published,
            crawled_at=crawled_at,
            language=language,
            authors=authors,
            extras=extras,
        )

    def _load_sources(self) -> List[Source]:
        import json

        raw = json.loads(self.config.sources_path.read_text(encoding="utf-8"))
        sources: List[Source] = []
        for entry in raw:
            if entry.get("type") != "rss":
                continue
            sources.append(
                Source(id=entry.get("id", ""), type=entry["type"], urls=entry.get("urls", []))
            )
        return sources

    def fetch(
        self,
        start_time: Optional[datetime] = None,
        stage: Optional["StageHandle"] = None,
    ) -> List[Article]:
        """Fetch news articles according to the configured window."""

        if start_time is None:
            start_time = datetime.now(timezone.utc)
        since_td = self.parse_since(self.time_window.since)
        cutoff = start_time - since_td

        socket.setdefaulttimeout(max(1, self.config.timeout))

        sources = self._load_sources()
        all_tasks: List[Tuple[str, str, Optional[datetime]]] = []
        for src in sources:
            urls = self._collect_feed_urls(
                src.id,
                src.urls,
                cutoff,
                request_headers={"User-Agent": self.config.user_agent},
                retries=self.config.feed_retries,
            )
            if self.config.max_per_source and len(urls) > self.config.max_per_source:
                urls = urls[: self.config.max_per_source]
            for url, dt in urls:
                all_tasks.append((src.id, url, dt))

        if not all_tasks:
            return []

        if stage is not None:
            stage.set_total(len(all_tasks))

        results: List[Article] = []
        lock = Lock()

        def _append(article: Optional[Article]) -> None:
            if article is None:
                return
            with lock:
                results.append(article)

        with futures.ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            future_to_url = {
                pool.submit(self._map_article, source_id, url, guess_dt): url
                for source_id, url, guess_dt in all_tasks
            }
            for fut in futures.as_completed(future_to_url):
                try:
                    article = fut.result()
                    if article is None:
                        continue
                    if article.best_timestamp() < cutoff:
                        continue
                    _append(article)
                except Exception:
                    continue
                finally:
                    if stage is not None:
                        stage.advance(1)

        return results


__all__ = ["NewsFetcher"]
