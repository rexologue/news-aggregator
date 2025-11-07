"""Pydantic schemas for request and response payloads."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, validator

from .models import NewsReport


class RankingRequest(BaseModel):
    topics: List[str] = Field(..., description="List of topics to rank news against")
    top_n: int = Field(..., gt=0, description="Number of top articles to return")

    @validator("topics")
    def validate_topics(cls, value: List[str]) -> List[str]:
        cleaned = [topic.strip() for topic in value if topic and topic.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty topic is required")
        return cleaned


class ReportResponse(BaseModel):
    agency: str
    title: str | None
    summary: str
    image_base64: str | None
    url: str
    published_at: datetime
    crawled_at: datetime

    @classmethod
    def from_report(cls, report: NewsReport) -> "ReportResponse":
        return cls(
            agency=report.agency,
            title=report.title,
            summary=report.summary,
            image_base64=report.image_base64,
            url=report.url,
            published_at=report.published_at,
            crawled_at=report.crawled_at,
        )


__all__ = ["RankingRequest", "ReportResponse"]
