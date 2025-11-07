"""Core data models for the news aggregator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Article:
    """Representation of a parsed news article."""

    source_id: str
    source_domain: str
    url: str
    title: Optional[str]
    content: Optional[str]
    published_at: Optional[datetime]
    crawled_at: datetime
    language: Optional[str]
    authors: Optional[List[str]]
    extras: Dict[str, Any] = field(default_factory=dict)

    def best_timestamp(self) -> datetime:
        """Return the best available timestamp (published or crawled)."""

        return self.published_at or self.crawled_at


@dataclass
class NewsReport:
    """A processed and summarized news report ready for serving."""

    agency: str
    title: Optional[str]
    summary: str
    image_base64: Optional[str]
    url: str
    published_at: datetime
    crawled_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agency": self.agency,
            "title": self.title,
            "summary": self.summary,
            "image_base64": self.image_base64,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "crawled_at": self.crawled_at.isoformat(),
        }


__all__ = ["Article", "NewsReport"]
