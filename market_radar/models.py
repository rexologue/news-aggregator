"""Core data models used by the Market Radar pipeline."""

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

    summary: Optional[str] = None
    density_coef: Optional[float] = None
    domain_coef: Optional[float] = None
    time_coef: Optional[float] = None
    hotness: Optional[float] = None

    def best_timestamp(self) -> datetime:
        """Return the best available timestamp (published or crawled)."""

        return self.published_at or self.crawled_at


__all__ = ["Article"]
